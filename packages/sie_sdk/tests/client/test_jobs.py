"""client.jobs namespace — request-shape assertions against a mocked transport."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import msgpack
import pytest
from sie_sdk import RequestError, ServerError, SIEAsyncClient, SIEClient

GW = "http://gw.test:8080"
KEY = "sk-sie-testkey"

_SUBMIT_RESP = {
    "id": "job-1",
    "object": "job",
    "operation": "encode",
    "model": "BAAI/bge-m3",
    "state": "queued",
    "total_items": 2,
    "chunks": 1,
    "preflight": {"estimated_credits": 64},
}


def _result_chunk(item_id: str) -> bytes:
    return msgpack.packb(
        [
            {
                "success": True,
                "id": item_id,
                "units": {"input_tokens": 5},
                "result_msgpack": msgpack.packb(
                    {"dense": {"dims": 2, "values": [0.1, 0.2]}},
                    use_bin_type=True,
                ),
            }
        ],
        use_bin_type=True,
    )


def _result_job(*refs: str) -> dict[str, Any]:
    return {
        "id": "job-1",
        "state": "succeeded",
        "total_items": len(refs),
        "settled_credits": 5 * len(refs),
        "output": {
            "kind": "refs",
            "chunks": [{"seq": seq, "items": 1, "state": "succeeded", "ref": ref} for seq, ref in enumerate(refs)],
        },
    }


def _resp(status: int, body: Any) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.headers = {"content-type": "application/json"}
    payload = json.dumps(body).encode()
    r.content = payload
    r.json.return_value = body
    return r


class _FakeAio:
    """Minimal stand-in for the async client's _AioResponse."""

    def __init__(self, status: int, body: Any) -> None:
        self.status_code = status
        self.content = json.dumps(body).encode()
        self.headers = {"content-type": "application/json"}
        self._body = body

    def json(self) -> Any:
        return self._body


# ---------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------


def test_submit_inline_maps_items_and_posts_v1_jobs() -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.request = MagicMock(return_value=_resp(201, _SUBMIT_RESP))
        client = SIEClient(GW, api_key=KEY)
        result = client.jobs.submit(source=["a", "b"], model="BAAI/bge-m3")
        assert result["id"] == "job-1"
        method, url = mock_client.return_value.request.call_args.args
        body = mock_client.return_value.request.call_args.kwargs["json"]
        assert (method, url) == ("POST", "/v1/jobs")
        assert body == {"operation": "encode", "model": "BAAI/bge-m3", "items": [{"text": "a"}, {"text": "b"}]}
        assert "Idempotency-Key" not in mock_client.return_value.request.call_args.kwargs["headers"]
        client.close()


def test_submit_connector_job_body() -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.request = MagicMock(return_value=_resp(201, _SUBMIT_RESP))
        client = SIEClient(GW, api_key=KEY)
        client.jobs.submit(
            source="postgres://warehouse?query=x",
            model="BAAI/bge-m3",
            sink="s3://out/vecs",
            execution="plan",
            idempotency_key="plan-postgres-1",
        )
        call = mock_client.return_value.request.call_args
        body = call.kwargs["json"]
        assert body["src"] == "postgres://warehouse?query=x"
        assert body["connection"] == "warehouse"
        assert body["sink"] == "s3://out/vecs"
        assert body["sink_connection"] == "out"
        assert body["execution"] == "plan"
        assert [(key, value) for key, value in call.kwargs["headers"].items() if key.lower() == "idempotency-key"] == [
            ("Idempotency-Key", "plan-postgres-1")
        ]
        client.close()


def test_submit_connector_requires_retry_stable_idempotency_key_before_io() -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        client = SIEClient(GW, api_key=KEY)
        with pytest.raises(ValueError, match="idempotency_key"):
            client.jobs.submit(
                source="postgres://warehouse?query=x",
                model="BAAI/bge-m3",
                execution="plan",
            )
        mock_client.return_value.request.assert_not_called()
        client.close()


def test_submit_field_map_job_body() -> None:
    """field_map + output_field ride the wire; upload:// derives no connection."""
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.request = MagicMock(return_value=_resp(201, _SUBMIT_RESP))
        client = SIEClient(GW, api_key=KEY)
        client.jobs.submit(
            source="upload://file-abc?format=csv",
            model="BAAI/bge-m3",
            sink="upload://file-out",
            field_map={"id_field": "doc_id", "input_field": "text", "carry": ["source_url"], "input_type": "text"},
            output_field="embedding",
            execution="run",
            idempotency_key="run-upload-1",
        )
        body = mock_client.return_value.request.call_args.kwargs["json"]
        assert body["src"] == "upload://file-abc?format=csv"
        assert body["sink"] == "upload://file-out"
        assert "connection" not in body  # internal scheme: OUR store, no org connection
        assert body["field_map"] == {
            "id_field": "doc_id",
            "input_field": "text",
            "carry": ["source_url"],
            "input_type": "text",
        }
        assert body["output_field"] == "embedding"
        client.close()


def test_submit_score_options_query_rides_the_wire() -> None:
    """Op inputs ride `options` (op matrix): a score job's query reaches the body."""
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.request = MagicMock(return_value=_resp(201, _SUBMIT_RESP))
        client = SIEClient(GW, api_key=KEY)
        client.jobs.submit(
            source="postgres://warehouse?query=x",
            model="BAAI/bge-m3",
            operation="score",
            sink="postgres://warehouse?table=scores",
            options={"query": "rank these documents"},
            execution="run",
            idempotency_key="run-score-1",
        )
        body = mock_client.return_value.request.call_args.kwargs["json"]
        assert body["operation"] == "score"
        assert body["options"] == {"query": "rank these documents"}
        client.close()


def test_get_and_cancel_hit_expected_urls() -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.request = MagicMock(return_value=_resp(200, {"id": "job-1", "state": "cancelled"}))
        client = SIEClient(GW, api_key=KEY)
        client.jobs.get("job-1")
        assert mock_client.return_value.request.call_args.args == ("GET", "/v1/jobs/job-1")
        out = client.jobs.cancel("job-1")
        assert mock_client.return_value.request.call_args.args == ("POST", "/v1/jobs/job-1/cancel")
        assert out["state"] == "cancelled"
        client.close()


@pytest.mark.parametrize(
    ("job_id", "encoded_job_id"),
    [
        ("job-victim/cancel", "job-victim%2Fcancel"),
        ("job-victim?next=/cancel", "job-victim%3Fnext%3D%2Fcancel"),
        ("job-victim#cancel", "job-victim%23cancel"),
    ],
)
def test_sync_job_id_is_one_percent_encoded_path_segment(job_id: str, encoded_job_id: str) -> None:
    response = {"id": "job-victim", "state": "succeeded", "output": {"kind": "refs", "chunks": []}}
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.request = MagicMock(return_value=_resp(200, response))
        client = SIEClient(GW, api_key=KEY)

        client.jobs.get(job_id)
        assert mock_client.return_value.request.call_args.args == ("GET", f"/v1/jobs/{encoded_job_id}")
        client.jobs.cancel(job_id)
        assert mock_client.return_value.request.call_args.args == ("POST", f"/v1/jobs/{encoded_job_id}/cancel")
        client.jobs.execute(job_id, 3, "execute-plan-3")
        assert mock_client.return_value.request.call_args.args == ("POST", f"/v1/jobs/{encoded_job_id}/execute")
        client.jobs.repair(job_id, 3, 2, "repair-plan-3-attempt-2")
        assert mock_client.return_value.request.call_args.args == ("POST", f"/v1/jobs/{encoded_job_id}/repair")
        client.jobs.results(job_id)
        assert mock_client.return_value.request.call_args.args == ("GET", f"/v1/jobs/{encoded_job_id}")
        client.close()


def test_execute_posts_exact_revision_and_one_idempotency_key() -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.request = MagicMock(return_value=_resp(200, {"id": "job-1", "state": "queued"}))
        client = SIEClient(GW, api_key=KEY)
        result = client.jobs.execute("job-1", 3, "execute-plan-3")
        call = mock_client.return_value.request.call_args
        assert call.args == ("POST", "/v1/jobs/job-1/execute")
        assert call.kwargs["json"] == {"plan_revision": 3}
        assert [(key, value) for key, value in call.kwargs["headers"].items() if key.lower() == "idempotency-key"] == [
            ("Idempotency-Key", "execute-plan-3")
        ]
        assert result["id"] == "job-1"
        client.close()


def test_repair_posts_exact_revision_predecessor_and_one_idempotency_key() -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.request = MagicMock(return_value=_resp(202, {"id": "job-1", "state": "running"}))
        client = SIEClient(GW, api_key=KEY)
        result = client.jobs.repair("job-1", 3, 2, "repair-plan-3-attempt-2")
        call = mock_client.return_value.request.call_args
        assert call.args == ("POST", "/v1/jobs/job-1/repair")
        assert call.kwargs["json"] == {"plan_revision": 3, "recovery_attempt_ordinal": 2}
        assert [(key, value) for key, value in call.kwargs["headers"].items() if key.lower() == "idempotency-key"] == [
            ("Idempotency-Key", "repair-plan-3-attempt-2")
        ]
        assert result["id"] == "job-1"
        client.close()


def test_wait_returns_stable_planned_phase_without_polling() -> None:
    planned = {"id": "job-plan-1", "state": "queued", "execution": "plan", "phase": "planned"}
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.request = MagicMock(return_value=_resp(200, planned))
        client = SIEClient(GW, api_key=KEY)
        assert client.jobs.wait("job-plan-1", poll_s=0)["phase"] == "planned"
        mock_client.return_value.request.assert_called_once()
        client.close()


def test_list_returns_data_array() -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.request = MagicMock(
            return_value=_resp(200, {"object": "list", "data": [{"id": "job-1"}, {"id": "job-2"}]})
        )
        client = SIEClient(GW, api_key=KEY)
        jobs = client.jobs.list()
        assert [j["id"] for j in jobs] == ["job-1", "job-2"]
        assert mock_client.return_value.request.call_args.args == ("GET", "/v1/jobs")
        client.close()


def test_results_reads_local_refs_and_decodes(tmp_path: Any) -> None:
    ref = tmp_path / "chunk0.msgpack"
    ref.write_bytes(
        msgpack.packb(
            [
                {
                    "success": True,
                    "id": str(i),
                    "units": {"input_tokens": 5},
                    "result_msgpack": msgpack.packb(
                        {"dense": {"dims": 4, "values": [0.1, 0.2, 0.3, 0.4]}}, use_bin_type=True
                    ),
                }
                for i in range(3)
            ],
            use_bin_type=True,
        )
    )
    job = {
        "id": "job-1",
        "state": "succeeded",
        "total_items": 3,
        "settled_credits": 15,
        "output": {"kind": "refs", "chunks": [{"seq": 0, "items": 3, "state": "succeeded", "ref": str(ref)}]},
    }
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.request = MagicMock(return_value=_resp(200, job))
        client = SIEClient(GW, api_key=KEY)
        results = client.jobs.results("job-1")
        assert results["retrieved"] == 3
        assert results["dims"] == 4
        assert results["items"][0]["dense"].shape == (4,)
        client.close()


def test_results_refreshes_job_after_replica_miss_without_duplicating_partial_items() -> None:
    stale = _result_job("https://gw.test/stale-0", "https://gw.test/stale-1")
    fresh = _result_job("https://gw.test/fresh-0", "https://gw.test/fresh-1")
    replica_miss = RequestError("not on this replica", code="RESULT_NOT_FOUND", status_code=404)
    with patch("sie_sdk.client.sync.httpx.Client"):
        client = SIEClient(GW, api_key=KEY)
        with (
            patch.object(client.jobs, "get", side_effect=[stale, fresh]) as get_job,
            patch.object(
                client.jobs,
                "_read_ref",
                side_effect=[_result_chunk("discarded"), replica_miss, _result_chunk("0"), _result_chunk("1")],
            ) as read_ref,
        ):
            results = client.jobs.results("job-1")

        assert get_job.call_count == 2
        assert read_ref.call_count == 4
        assert results["retrieved"] == 2
        assert [item["id"] for item in results["items"]] == ["0", "1"]
        assert [chunk["ref"] for chunk in results["chunks"]] == [
            "https://gw.test/fresh-0",
            "https://gw.test/fresh-1",
        ]
        client.close()


def test_results_bounds_replica_miss_refreshes() -> None:
    replica_miss = RequestError("not on this replica", code="RESULT_NOT_FOUND", status_code=404)
    with patch("sie_sdk.client.sync.httpx.Client"):
        client = SIEClient(GW, api_key=KEY)
        with (
            patch.object(client.jobs, "get", return_value=_result_job("https://gw.test/stale")) as get_job,
            patch.object(client.jobs, "_read_ref", side_effect=replica_miss) as read_ref,
            pytest.raises(RequestError, match="not on this replica") as excinfo,
        ):
            client.jobs.results("job-1")

        expected_attempts = 4
        assert get_job.call_count == expected_attempts
        assert read_ref.call_count == expected_attempts
        assert excinfo.value is replica_miss
        client.close()


@pytest.mark.parametrize(
    "error",
    [
        RequestError("ordinary missing ref", code="NOT_FOUND", status_code=404),
        ServerError("payload store unavailable", code="STORAGE_UNAVAILABLE", status_code=503),
    ],
)
def test_results_does_not_refresh_unrelated_ref_failures(error: RequestError | ServerError) -> None:
    with patch("sie_sdk.client.sync.httpx.Client"):
        client = SIEClient(GW, api_key=KEY)
        with (
            patch.object(client.jobs, "get", return_value=_result_job("https://gw.test/ref")) as get_job,
            patch.object(client.jobs, "_read_ref", side_effect=error) as read_ref,
            pytest.raises(type(error)) as excinfo,
        ):
            client.jobs.results("job-1")

        get_job.assert_called_once_with("job-1")
        read_ref.assert_called_once_with("https://gw.test/ref")
        assert excinfo.value is error
        client.close()


# ---------------------------------------------------------------------------
# async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_submit_serializes_json_body() -> None:
    client = SIEAsyncClient(GW, api_key=KEY)
    client._post = AsyncMock(return_value=_FakeAio(201, _SUBMIT_RESP))
    result = await client.jobs.submit(source=["a"], model="m")
    assert result["id"] == "job-1"
    call = client._post.call_args
    assert call.args[0] == "/v1/jobs"
    assert json.loads(call.kwargs["data"]) == {"operation": "encode", "model": "m", "items": [{"text": "a"}]}
    assert "Idempotency-Key" not in call.kwargs["headers"]
    await client.close()


@pytest.mark.asyncio
async def test_async_wait_returns_stable_planned_phase_without_polling() -> None:
    client = SIEAsyncClient(GW, api_key=KEY)
    client._get = AsyncMock(
        return_value=_FakeAio(200, {"id": "job-plan-1", "state": "queued", "execution": "plan", "phase": "planned"})
    )
    assert (await client.jobs.wait("job-plan-1", poll_s=0))["phase"] == "planned"
    client._get.assert_awaited_once()
    await client.close()


@pytest.mark.asyncio
async def test_async_submit_floors_long_running_timeout() -> None:
    """jobs.submit floors the per-call timeout to 120s (parity with the sync client)."""
    client = SIEAsyncClient(GW, api_key=KEY)
    client._post = AsyncMock(return_value=_FakeAio(201, _SUBMIT_RESP))
    await client.jobs.submit(source=["a"], model="m")
    assert client._post.call_args.kwargs["timeout_s"] == max(client._timeout, 120.0)
    await client.close()


@pytest.mark.asyncio
async def test_async_submit_field_map_body() -> None:
    client = SIEAsyncClient(GW, api_key=KEY)
    client._post = AsyncMock(return_value=_FakeAio(201, _SUBMIT_RESP))
    await client.jobs.submit(
        source="postgres://wh?query=select id, body, source_url from docs",
        model="BAAI/bge-m3",
        sink="postgres://wh?table=doc_vectors",
        field_map={"id_field": "id", "input_field": "body", "carry": ["source_url"]},
        output_field="embedding",
        execution="plan",
        idempotency_key="async-plan-postgres-1",
    )
    body = json.loads(client._post.call_args.kwargs["data"])
    assert body["field_map"] == {"id_field": "id", "input_field": "body", "carry": ["source_url"]}
    assert body["output_field"] == "embedding"
    assert body["connection"] == "wh"
    assert body["execution"] == "plan"
    assert client._post.call_args.kwargs["headers"]["Idempotency-Key"] == "async-plan-postgres-1"
    await client.close()


@pytest.mark.asyncio
async def test_async_submit_connector_requires_retry_stable_idempotency_key_before_io() -> None:
    client = SIEAsyncClient(GW, api_key=KEY)
    client._post = AsyncMock()
    with pytest.raises(ValueError, match="idempotency_key"):
        await client.jobs.submit(
            source="postgres://wh?query=select id, body from docs",
            model="BAAI/bge-m3",
            execution="plan",
        )
    client._post.assert_not_awaited()
    await client.close()


@pytest.mark.asyncio
async def test_async_submit_extract_options_labels_body() -> None:
    """Async parity: an extract job's labels/output_schema ride `options` as-is."""
    client = SIEAsyncClient(GW, api_key=KEY)
    client._post = AsyncMock(return_value=_FakeAio(201, _SUBMIT_RESP))
    await client.jobs.submit(
        source=["some text"],
        model="urchade/gliner_small-v2.1",
        operation="extract",
        options={"labels": ["PERSON", "ORG"], "output_schema": {"type": "object"}},
    )
    body = json.loads(client._post.call_args.kwargs["data"])
    assert body["operation"] == "extract"
    assert body["options"] == {"labels": ["PERSON", "ORG"], "output_schema": {"type": "object"}}
    await client.close()


@pytest.mark.asyncio
async def test_async_list_and_cancel() -> None:
    client = SIEAsyncClient(GW, api_key=KEY)
    client._get = AsyncMock(return_value=_FakeAio(200, {"object": "list", "data": [{"id": "job-9"}]}))
    client._post = AsyncMock(return_value=_FakeAio(200, {"id": "job-9", "state": "cancelled"}))
    assert [j["id"] for j in await client.jobs.list()] == ["job-9"]
    out = await client.jobs.cancel("job-9")
    assert out["state"] == "cancelled"
    assert client._post.call_args.args[0] == "/v1/jobs/job-9/cancel"
    await client.close()


@pytest.mark.parametrize(
    ("job_id", "encoded_job_id"),
    [
        ("job-victim/cancel", "job-victim%2Fcancel"),
        ("job-victim?next=/cancel", "job-victim%3Fnext%3D%2Fcancel"),
        ("job-victim#cancel", "job-victim%23cancel"),
    ],
)
@pytest.mark.asyncio
async def test_async_job_id_is_one_percent_encoded_path_segment(job_id: str, encoded_job_id: str) -> None:
    response = {"id": "job-victim", "state": "succeeded", "output": {"kind": "refs", "chunks": []}}
    client = SIEAsyncClient(GW, api_key=KEY)
    client._get = AsyncMock(return_value=_FakeAio(200, response))
    client._post = AsyncMock(return_value=_FakeAio(200, response))

    await client.jobs.get(job_id)
    assert client._get.call_args.args[0] == f"/v1/jobs/{encoded_job_id}"
    await client.jobs.cancel(job_id)
    assert client._post.call_args.args[0] == f"/v1/jobs/{encoded_job_id}/cancel"
    await client.jobs.execute(job_id, 3, "execute-plan-3")
    assert client._post.call_args.args[0] == f"/v1/jobs/{encoded_job_id}/execute"
    await client.jobs.repair(job_id, 3, 2, "repair-plan-3-attempt-2")
    assert client._post.call_args.args[0] == f"/v1/jobs/{encoded_job_id}/repair"
    await client.jobs.results(job_id)
    assert client._get.call_args.args[0] == f"/v1/jobs/{encoded_job_id}"
    await client.close()


@pytest.mark.asyncio
async def test_async_execute_posts_exact_revision_and_one_idempotency_key() -> None:
    client = SIEAsyncClient(GW, api_key=KEY)
    client._post = AsyncMock(return_value=_FakeAio(200, {"id": "job-9", "state": "queued"}))
    result = await client.jobs.execute("job-9", 2, "async-execute-plan-2")
    call = client._post.call_args
    assert call.args[0] == "/v1/jobs/job-9/execute"
    assert json.loads(call.kwargs["data"]) == {"plan_revision": 2}
    assert [(key, value) for key, value in call.kwargs["headers"].items() if key.lower() == "idempotency-key"] == [
        ("Idempotency-Key", "async-execute-plan-2")
    ]
    assert result["id"] == "job-9"
    await client.close()


@pytest.mark.asyncio
async def test_async_repair_posts_exact_revision_predecessor_and_one_idempotency_key() -> None:
    client = SIEAsyncClient(GW, api_key=KEY)
    client._post = AsyncMock(return_value=_FakeAio(202, {"id": "job-9", "state": "running"}))
    result = await client.jobs.repair("job-9", 2, 1, "async-repair-plan-2-attempt-1")
    call = client._post.call_args
    assert call.args[0] == "/v1/jobs/job-9/repair"
    assert json.loads(call.kwargs["data"]) == {"plan_revision": 2, "recovery_attempt_ordinal": 1}
    assert [(key, value) for key, value in call.kwargs["headers"].items() if key.lower() == "idempotency-key"] == [
        ("Idempotency-Key", "async-repair-plan-2-attempt-1")
    ]
    assert result["id"] == "job-9"
    await client.close()


@pytest.mark.asyncio
async def test_async_results_refreshes_job_after_replica_miss_without_duplicating_partial_items() -> None:
    stale = _result_job("https://gw.test/stale-0", "https://gw.test/stale-1")
    fresh = _result_job("https://gw.test/fresh-0", "https://gw.test/fresh-1")
    replica_miss = RequestError("not on this replica", code="RESULT_NOT_FOUND", status_code=404)
    client = SIEAsyncClient(GW, api_key=KEY)
    with (
        patch.object(client.jobs, "get", new=AsyncMock(side_effect=[stale, fresh])) as get_job,
        patch.object(
            client.jobs,
            "_read_ref",
            new=AsyncMock(
                side_effect=[_result_chunk("discarded"), replica_miss, _result_chunk("0"), _result_chunk("1")]
            ),
        ) as read_ref,
    ):
        results = await client.jobs.results("job-1")

    assert get_job.await_count == 2
    assert read_ref.await_count == 4
    assert results["retrieved"] == 2
    assert [item["id"] for item in results["items"]] == ["0", "1"]
    assert [chunk["ref"] for chunk in results["chunks"]] == [
        "https://gw.test/fresh-0",
        "https://gw.test/fresh-1",
    ]
    await client.close()


@pytest.mark.asyncio
async def test_async_results_bounds_replica_miss_refreshes() -> None:
    replica_miss = RequestError("not on this replica", code="RESULT_NOT_FOUND", status_code=404)
    client = SIEAsyncClient(GW, api_key=KEY)
    with (
        patch.object(client.jobs, "get", new=AsyncMock(return_value=_result_job("https://gw.test/stale"))) as get_job,
        patch.object(client.jobs, "_read_ref", new=AsyncMock(side_effect=replica_miss)) as read_ref,
        pytest.raises(RequestError, match="not on this replica") as excinfo,
    ):
        await client.jobs.results("job-1")

    expected_attempts = 4
    assert get_job.await_count == expected_attempts
    assert read_ref.await_count == expected_attempts
    assert excinfo.value is replica_miss
    await client.close()


@pytest.mark.parametrize(
    "error",
    [
        RequestError("ordinary missing ref", code="NOT_FOUND", status_code=404),
        ServerError("payload store unavailable", code="STORAGE_UNAVAILABLE", status_code=503),
    ],
)
@pytest.mark.asyncio
async def test_async_results_does_not_refresh_unrelated_ref_failures(error: RequestError | ServerError) -> None:
    client = SIEAsyncClient(GW, api_key=KEY)
    with (
        patch.object(client.jobs, "get", new=AsyncMock(return_value=_result_job("https://gw.test/ref"))) as get_job,
        patch.object(client.jobs, "_read_ref", new=AsyncMock(side_effect=error)) as read_ref,
        pytest.raises(type(error)) as excinfo,
    ):
        await client.jobs.results("job-1")

    get_job.assert_awaited_once_with("job-1")
    read_ref.assert_awaited_once_with("https://gw.test/ref")
    assert excinfo.value is error
    await client.close()
