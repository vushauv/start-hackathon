"""End-to-end tests against a fake Reactor WebSocket server on localhost."""

import asyncio
import json
import re

import pytest
from websockets.asyncio.server import serve

import reactor_service as reactor
from reactor_service import ReactorSettings

REF_RE = re.compile(r"\[ref:([0-9a-f]{8})\]")


def make_settings(port: int, **kw) -> ReactorSettings:
    return ReactorSettings(
        _env_file=None,
        base_url=f"http://127.0.0.1:{port}", agent_id="agt_test", access_key="rxk_test", **kw,
    )


async def run_server(handler):
    server = await serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, port


async def test_ask_happy_path():
    seen = {}

    async def handler(ws):
        seen["auth"] = ws.request.headers.get("Authorization")
        seen["path"] = ws.request.path
        msg = json.loads(await ws.recv())
        ref = REF_RE.search(msg["text"]).group(1)
        await ws.send(json.dumps({"type": "ack", "text": msg["text"], "cycle": 3}))
        await ws.send(json.dumps({"type": "reply", "text": "stale", "cycle": 3}))
        await ws.send(json.dumps({"type": "reply", "text": "let me think", "cycle": 4}))
        await ws.send(json.dumps({
            "type": "reply", "cycle": 4,
            "text": f"notes\n# FINAL ANSWER [ref:{ref}]: June [E2]",
        }))

    server, port = await run_server(handler)
    async with server:
        s = make_settings(port)
        ev = [{"id": "s_1", "text": "May"}, {"id": "s_2", "text": "June", "date": "2023/05/27"}]
        a = await reactor.aask("When?", ev, history_id="h1", question_id="q1", settings=s)

    assert seen["auth"] == "Bearer rxk_test"
    assert seen["path"] == "/agents/agt_test/ws/chat?history=h1"
    assert a.answer == "June [E2]"
    assert a.cited_ids == ["s_2"]
    assert a.unknown_citations == []
    assert a.abstained is False
    assert a.uncited is False
    assert a.other_replies == ["let me think"]
    assert a.cycle == 4 and a.question_id == "q1" and a.history_id == "h1"
    assert json.loads(a.to_jsonl())["status"] == "answered"


async def test_ask_structured_final_answer():
    async def handler(ws):
        msg = json.loads(await ws.recv())
        ref = REF_RE.search(msg["text"]).group(1)
        await ws.send(json.dumps({"type": "ack", "text": msg["text"], "cycle": 0}))
        await ws.send(json.dumps({
            "type": "reply", "cycle": 1,
            "text": f'# FINAL ANSWER [ref:{ref}]: {{"answer":"June","citations":["s_2"]}}',
        }))

    server, port = await run_server(handler)
    async with server:
        a = await reactor.aask("When?", [{"id": "s_1", "text": "May"}, {"id": "s_2", "text": "June"}], settings=make_settings(port))
    assert a.answer == "June"
    assert a.cited_ids == ["s_2"]
    assert a.uncited is False


async def test_abstention_and_uncited():
    async def handler(ws):
        msg = json.loads(await ws.recv())
        ref = REF_RE.search(msg["text"]).group(1)
        await ws.send(json.dumps({"type": "ack", "text": msg["text"], "cycle": 0}))
        await ws.send(json.dumps({"type": "reply", "cycle": 1,
                                  "text": f"# FINAL ANSWER [ref:{ref}]: INSUFFICIENT_EVIDENCE"}))

    server, port = await run_server(handler)
    async with server:
        a = await reactor.aask("?", [{"id": "x", "text": "y"}], settings=make_settings(port))
    assert a.abstained is True and a.citations == [] and a.uncited is False


async def test_retry_on_busy_then_answer():
    calls = {"n": 0}

    async def handler(ws):
        calls["n"] += 1
        msg = json.loads(await ws.recv())
        if calls["n"] == 1:
            await ws.send(json.dumps({"type": "error", "code": "busy", "forwarded": False,
                                      "text": "another chat is open"}))
            return
        await ws.send(json.dumps({"type": "ack", "text": msg["text"], "cycle": 0}))
        await ws.send(json.dumps({"type": "reply", "cycle": 1, "text": "OK"}))

    server, port = await run_server(handler)
    async with server:
        r = await reactor.asend_text("hello", settings=make_settings(port))
    assert r.reply == "OK" and r.retries == 1 and calls["n"] == 2


async def test_non_retryable_refusal():
    async def handler(ws):
        await ws.recv()
        await ws.send(json.dumps({"type": "error", "code": "history_required",
                                  "forwarded": False, "text": "run new history"}))

    server, port = await run_server(handler)
    async with server:
        with pytest.raises(reactor.Refused) as ei:
            await reactor.asend_text("hello", settings=make_settings(port))
    assert ei.value.info.code == "history_required"
    assert ei.value.retryable is False


async def test_timeout():
    async def handler(ws):
        await ws.recv()
        await asyncio.sleep(5)

    server, port = await run_server(handler)
    async with server:
        with pytest.raises(reactor.TimedOut):
            await reactor.asend_text("hello", timeout=0.3, settings=make_settings(port))


async def test_closed_without_answer():
    async def handler(ws):
        await ws.recv()

    server, port = await run_server(handler)
    async with server:
        with pytest.raises(reactor.Closed):
            await reactor.asend_text("hello", settings=make_settings(port))


async def test_connect_403():
    from websockets.http11 import Response

    async def process_request(connection, request):
        return Response(403, "Forbidden", websockets.datastructures.Headers())

    import websockets

    server = await serve(lambda ws: None, "127.0.0.1", 0, process_request=process_request)
    port = server.sockets[0].getsockname()[1]
    async with server:
        with pytest.raises(reactor.ConnectFailed) as ei:
            await reactor.asend_text("hello", settings=make_settings(port))
    assert ei.value.status == 403


def test_message_too_large():
    s = make_settings(1, max_chars=10)
    with pytest.raises(reactor.MessageTooLarge):
        reactor.send_text("x" * 11, settings=s)


def test_sync_wrapper_in_loop_errors():
    async def inner():
        reactor.send_text("hi", settings=make_settings(1))

    with pytest.raises(reactor.ReactorError, match="async variant"):
        asyncio.run(inner())


def test_split_history_respects_limit():
    sessions = [
        {"session_id": f"s{i}", "date": "2023/01/0%d" % (i + 1),
         "turns": [{"role": "user", "content": "a" * 300}]}
        for i in range(10)
    ]
    parts = reactor.split_history(sessions, max_chars=1000)
    assert len(parts) > 1
    assert all(len(p) <= 800 for p in parts)
    assert parts[0].startswith("## session s0")


def test_settings_from_json(tmp_path):
    p = tmp_path / "creds.json"
    p.write_text(json.dumps({
        "team_name": "t", "slot": "t02",
        "base_url": "https://t02.api.reactor-hackathon.rxai.dev/",
        "agent_id": "agt_1", "access_key": "rxk_1",
    }))
    s = ReactorSettings.from_json(p)
    assert s.base_url == "https://t02.api.reactor-hackathon.rxai.dev"
    assert s.chat_url("h/1") == "wss://t02.api.reactor-hackathon.rxai.dev/agents/agt_1/ws/chat?history=h%2F1"
    assert "rxk_1" not in repr(s)


def test_settings_rejects_path_and_http():
    with pytest.raises(ValueError):
        ReactorSettings(_env_file=None, base_url="https://x.dev/api", agent_id="a", access_key="k")
    with pytest.raises(ValueError):
        ReactorSettings(_env_file=None, base_url="http://x.dev", agent_id="a", access_key="k")
