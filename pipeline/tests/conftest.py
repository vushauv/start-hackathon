"""A fake Reactor server (HTTP for new history, WebSocket for chat) in a background thread."""

from __future__ import annotations

import asyncio
import json
import re
import threading

import pytest
from reactor_service import ReactorSettings
from websockets.asyncio.server import serve
from websockets.http11 import Response
from websockets.datastructures import Headers

REF_RE = re.compile(r"\[ref:([0-9a-f]{8})\]")


class FakeReactor:
    """``answer_for(message_text) -> reply text`` decides what the agent says. Records every message."""

    def __init__(self, answer_for):
        self.answer_for = answer_for
        self.messages: list[str] = []
        self.histories = 0
        self.port = 0
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _process_request(self, connection, request):
        if request.path.endswith("/history") and request.headers.get("Authorization") == "Bearer rxk_test":
            self.histories += 1
            body = json.dumps({"history_id": f"fake-{self.histories}", "running": True}).encode()
            return Response(200, "OK", Headers([("Content-Type", "application/json")]), body)
        return None  # proceed with the WebSocket handshake

    async def _handler(self, ws):
        msg = json.loads(await ws.recv())
        text = msg["text"]
        self.messages.append(text)
        await ws.send(json.dumps({"type": "ack", "text": text, "cycle": 1}))
        m = REF_RE.search(text)
        reply = self.answer_for(text)
        if m and reply is not None:
            reply = f"notes\n# FINAL ANSWER [ref:{m.group(1)}]: {reply}"
        await ws.send(json.dumps({"type": "reply", "text": reply or "OK", "cycle": 2}))

    def _run(self):
        asyncio.set_event_loop(self._loop)

        async def main():
            async with serve(self._handler, "127.0.0.1", 0, process_request=self._process_request) as server:
                self.port = server.sockets[0].getsockname()[1]
                self._ready.set()
                await asyncio.Event().wait()

        self._loop.run_until_complete(main())

    def start(self):
        self._thread.start()
        self._ready.wait(5)
        return self

    def settings(self) -> ReactorSettings:
        return ReactorSettings(_env_file=None, base_url=f"http://127.0.0.1:{self.port}",
                               agent_id="agt_test", access_key="rxk_test", timeout=5)


@pytest.fixture
def fake_reactor():
    def factory(answer_for):
        return FakeReactor(answer_for).start()
    return factory
