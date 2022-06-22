from fastapi.testclient import TestClient
from fastapi import WebSocket
from gop_server.preprocess import pcm2float
from tests.utils import create_test_fixture
import numpy as np
import soundfile as sf
import pytest
import json
import asyncio


@pytest.fixture
def app_ctx():
    return create_test_fixture()


def check_res(data):
    if 'bytes' in data:
        data = data['bytes']
        sf.write(f'tmp.wav', pcm2float(np.frombuffer(data, dtype=np.int16)), 16000, 'PCM_16')
    else:
        data = json.loads(data['text'])
        print(data)
        assert False


def test_tts_api(app_ctx):
    client: TestClient = app_ctx['client']

    with client.websocket_connect("/tts") as wb:
        wb: WebSocket = wb
        wb.send_json(dict(transcript='一本书', username=app_ctx['username'], token=app_ctx['token']))
        check_res(wb.receive())


def test_std_speech_api(app_ctx):
    client: TestClient = app_ctx['client']

    with client.websocket_connect("/std-speech") as wb:
        wb: WebSocket = wb
        wb.send_json(dict(transcript='一本书', username=app_ctx['username'], token=app_ctx['token']))
        check_res(wb.receive())


def check_listen_time(username: str, transcript: str, expected: int):
    from gop_server.db import get_user_listen_count
    q, _, _ = get_user_listen_count(username, transcript)
    assert q is not None
    assert q.count == expected


def test_listen_time_count(app_ctx):
    client: TestClient = app_ctx['client']
    username = app_ctx['username']
    token = app_ctx['token']
    transcript = '一本书'

    with client.websocket_connect("/std-speech") as wb:
        wb: WebSocket = wb
        wb.send_json(dict(transcript=transcript, username=username, token=token))
        check_res(wb.receive())
        check_listen_time(username, transcript, 1)

    with client.websocket_connect("/tts") as wb:
        wb: WebSocket = wb
        wb.send_json(dict(transcript=transcript, username=username, token=token))
        check_res(wb.receive())
        check_listen_time(username, transcript, 2)

    with client.websocket_connect("/std-speech") as wb:
        wb: WebSocket = wb
        wb.send_json(dict(transcript=transcript, username=username, token=token))
        check_res(wb.receive())
        check_listen_time(username, transcript, 3)


def test_tts():
    from gop_server.api.tts import tts
    text = '操你妈'
    loop = asyncio.get_event_loop()
    loop.run_until_complete(tts(text))
