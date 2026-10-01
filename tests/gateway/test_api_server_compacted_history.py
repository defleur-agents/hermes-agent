"""History requests preserve archived conversation without changing model context."""
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB


@pytest.mark.asyncio
async def test_compacted_history_is_opt_in_paginated_and_excludes_rewind(tmp_path):
    db = SessionDB(tmp_path / 'state.db')
    try:
        sid = db.create_session('history-regression', 'api_server')
        original = [
            {'role': 'user', 'content': 'original request'},
            {'role': 'assistant', 'content': 'first response'},
            {'role': 'user', 'content': 'middle follow-up'},
            {'role': 'assistant', 'content': 'middle response'},
        ]
        db.replace_messages(sid, original)
        db.archive_and_compact(sid, [{'role': 'assistant', 'content': 'latest response'}])
        removed = db.append_message(sid, 'assistant', 'rewound response')
        db._conn.execute('UPDATE messages SET active=0, compacted=0 WHERE id=?', (removed,))
        db._conn.commit()
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        adapter._session_db = db
        app = web.Application()
        app.router.add_get('/api/sessions/{session_id}/messages', adapter._handle_session_messages)
        async with TestClient(TestServer(app)) as client:
            base = f'/api/sessions/{sid}/messages'
            for suffix in ['', '?include_compacted=false']:
                response = await client.get(base + suffix)
                assert response.status == 200
                assert [m['content'] for m in (await response.json())['data']] == ['latest response']
            collected = []
            for offset in range(0, 6, 2):
                response = await client.get(base + f'?include_compacted=true&order=oldest&limit=2&offset={offset}')
                assert response.status == 200
                payload = await response.json()
                assert payload['pagination']['returned'] == len(payload['data'])
                collected.extend(m['content'] for m in payload['data'])
            assert collected == [m['content'] for m in original] + ['latest response']
            response = await client.get(base + '?include_compacted=true&order=latest&limit=2')
            assert [m['content'] for m in (await response.json())['data']] == ['middle response', 'latest response']
            response = await client.get(base + '?include_compacted=garbage')
            assert response.status == 400
        assert [m['content'] for m in db.get_messages(sid)] == ['latest response']
    finally:
        db.close()
