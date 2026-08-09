from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path

import pytest

pytest.importorskip("docling")

from sie_server.adapters.docling.adapter import DoclingAdapter
from sie_server.types.inputs import Item


@pytest.fixture(scope="module")
def loaded_adapter() -> DoclingAdapter:
    adapter = DoclingAdapter()
    adapter.load("cpu")
    return adapter


def _make_pdf_bytes() -> bytes:
    reportlab = pytest.importorskip("reportlab")
    from reportlab.pdfgen import canvas  # ty: ignore[unresolved-import]

    _ = reportlab
    buf = io.BytesIO()
    pdf = canvas.Canvas(buf)
    pdf.drawString(100, 750, "Smoke test heading")
    pdf.drawString(100, 720, "Hello from reportlab.")
    pdf.save()
    return buf.getvalue()


def _make_docx_bytes() -> bytes:
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_heading("Smoke test heading", level=1)
    document.add_paragraph("Hello from python-docx.")
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


def _make_html_bytes() -> bytes:
    return b"<html><body><h1>Smoke test heading</h1><p>Hello from HTML.</p></body></html>"


@pytest.mark.parametrize(
    ("format_hint", "maker"),
    [
        ("pdf", _make_pdf_bytes),
        ("docx", _make_docx_bytes),
        ("html", _make_html_bytes),
    ],
)
def test_extract_real_document(loaded_adapter: DoclingAdapter, format_hint: str, maker: Callable[[], bytes]) -> None:
    data = maker()
    out = loaded_adapter.extract([Item(document={"data": data, "format": format_hint})])

    assert out.batch_size == 1
    assert out.data is not None
    item = out.data[0]
    assert "error" not in item, f"adapter reported error: {item.get('error')}"
    assert "Smoke test heading" in item["text"] or "Smoke test heading" in item["markdown"]
    assert "document" in item


_PROSE = (
    "Alice was beginning to get very tired of sitting by her sister "
    "on the bank and of having nothing to do once or twice she had "
    "peeped into the book her sister was reading"
)


def _make_prose_image_bytes() -> bytes:
    pil = pytest.importorskip("PIL")
    from PIL import Image, ImageDraw, ImageFont  # ty: ignore[unresolved-import]

    _ = pil
    image = Image.new("RGB", (1400, 400), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=40)
    words = _PROSE.split()
    for row in range(0, len(words), 8):
        draw.text((40, 40 + (row // 8) * 70), " ".join(words[row : row + 8]), fill="black", font=font)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.integration
def test_served_ocr_output_keeps_word_boundaries() -> None:
    """#2919: end-to-end sanity of the ARTIFACT-BACKED served OCR path.

    This exercises the real pinned artifact revision end to end: it proves the
    revision resolves a usable RapidOCR set on the served path and returns
    segmented text, which a bare adapter on docling's own cache cannot show.

    SCOPE — measured, not assumed. This floor does NOT discriminate the
    recogniser language. Mutating the adapter to ``lang=["ch"]`` provably
    reaches a different recogniser (``ch_PP-OCRv4_rec_mobile.onnx`` instead of
    ``en_PP-OCRv4_rec_mobile.onnx``) yet yields BYTE-IDENTICAL text here
    (density 0.2741 both ways): the Chinese set reads clean, large, synthetic
    renders of Latin script perfectly. Degrading the fixture does not rescue
    the separation — across six render variants the two languages are
    non-monotonic, and on a gray/noisy variant the Chinese set scored HIGHER
    (0.2296 vs 0.1290). Threshold-fitting a synthetic image would buy a flaky
    test, not a guard.

    What actually holds the language pin is
    ``test_docling.py::TestDoclingMakeConverter`` (both cases assert
    ``ocr_options.lang == ["en"]``; flipping the adapter fails them in the
    default suite). What catches DEGRADED-but-segmented output on real pages is
    the quality floor now committed at
    ``benchmarks/ocr__Teklia__IAM-line/docling/quality/target.ocr.extract.json``
    (#2919): over 2,915 real pages the ocr profile scores mean_similarity
    0.3878 with the English recogniser and 0.3743 with the Chinese one, so a
    ``lang=["ch"]`` flip lands 3.49% under the floor and reds the gate. The
    same holds on olmOCR-bench (0.3458 vs 0.3347, -3.22%).
    ``test_benchmark_repository_contracts.py`` replays both and fails if either
    floor stops rejecting the defect.

    So: keep this as a served-path smoke over the pinned revision. Do not read
    it as evidence that the language is correct.
    """
    yaml = pytest.importorskip("yaml")
    model_yaml = Path(__file__).resolve().parents[2] / "models" / "docling.yaml"
    pinned = yaml.safe_load(model_yaml.read_text())

    adapter = DoclingAdapter(
        model_name_or_path=pinned["hf_id"],
        revision=pinned["hf_revision"],
    )
    adapter.load("cpu")
    out = adapter.extract(
        [Item(images=[{"data": _make_prose_image_bytes(), "format": "png"}])],
        options={"ocr": True},
    )

    assert out.data is not None
    assert out.errors is None, f"adapter reported errors: {out.errors}"
    item = out.data[0]
    text = item["text"]
    alpha = sum(1 for ch in text if ch.isalpha())
    whitespace = sum(1 for ch in text if ch.isspace())
    assert alpha > 40, f"OCR recognised too little text to judge: {text!r}"
    density = whitespace / alpha
    assert density > 0.10, (
        f"served OCR output is word-joined (whitespace/alpha = {density:.3f}); "
        f"the pinned artifact revision resolves no recogniser able to segment "
        f"this page at all. This is a catastrophic-failure floor, not a "
        f"language check — see the docstring: {text!r}"
    )
