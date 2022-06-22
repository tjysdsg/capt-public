from fastapi import WebSocket, APIRouter
from pydantic import ValidationError
import os
import json
from ws4py.client.threadedclient import WebSocketClient
import ws4py.messaging
import time
import numpy as np
from typing import Union, Tuple
from gop_server import server_config
from gop_server.error import ErrorCode
from gop_server.api.api_models import TranscriptInput, WSMessageResponse
from gop_server.db import increment_user_listen_count

router = APIRouter()


class MiniClient(WebSocketClient):
    def __init__(self, url, protocols=None, extensions=None, heartbeat_freq=None):
        super(MiniClient, self).__init__(url, protocols, extensions, heartbeat_freq)
        self.bytes = b''

    def received_message(self, msg):
        if isinstance(msg, ws4py.messaging.BinaryMessage):
            self.bytes += msg.data
        elif isinstance(msg, ws4py.messaging.TextMessage):
            msg = json.loads(msg.data.decode('utf-8'))

            if type(msg) == dict:
                print(msg)

    def get_data(self) -> np.ndarray:
        from gop_server.preprocess import pcm2float
        return pcm2float(np.frombuffer(self.bytes, dtype=np.int16))


async def tts(text: str, spkid='db4') -> Tuple[Union[bytes, dict], bool]:
    """
    :param text: Text
    :param spkid: [db4, db1, blm, ljs]
    """
    from gop_server import zh_config
    import soundfile as sf

    # cache dir
    std_pronun_dir = os.path.join(zh_config.root_path, 'std_pronun')
    os.makedirs(std_pronun_dir, exist_ok=True)

    # load cached wav file if exist
    output_path = os.path.join(std_pronun_dir, f'{text}-{spkid}.wav')
    if os.path.isfile(output_path):
        from gop_server.preprocess import float2pcm
        ret, _ = sf.read(output_path)
        ret = float2pcm(ret)
        ret = ret.tobytes()
    else:
        ws = MiniClient(server_config.tts_api_url)
        ws.connect()
        event = dict(spkid=spkid, text=text)
        ws.send(json.dumps(event))
        ws.send('<eos>')

        while not ws.client_terminated:
            time.sleep(0.5)

        sf.write(output_path, ws.get_data(), 16000, 'PCM_16')
        ret = ws.bytes

    return ret, True


@router.websocket('/tts')
async def tts_api(wb: WebSocket):
    await wb.accept()

    ret = WSMessageResponse()
    payload: dict = await wb.receive_json()

    try:
        in_: TranscriptInput = TranscriptInput.parse_obj(payload)
    except ValidationError as e:
        ret.status = ErrorCode.INVALID_INPUT
        ret.message = str(e)
        await wb.send_json(ret.dict())
        return
    data, success = await tts(in_.transcript)
    if success:
        await wb.send_bytes(data)
        increment_user_listen_count(in_.username, in_.transcript)
        return
    else:
        ret.server_response = data
        await wb.send_json(ret.dict())
        return


@router.websocket('/std-speech')
async def standard_speech(wb: WebSocket):
    from gop_server import zh_config
    import librosa
    from gop_server.preprocess import float2pcm

    await wb.accept()

    ret = WSMessageResponse()
    payload: dict = await wb.receive_json()

    try:
        in_: TranscriptInput = TranscriptInput.parse_obj(payload)
    except ValidationError as e:
        ret.status = ErrorCode.INVALID_INPUT
        ret.message = str(e)
        await wb.send_json(ret.dict())
        return

    std_speech_file = os.path.join(zh_config.root_path, 'word_list', f'{in_.transcript}.wav')
    if os.path.isfile(std_speech_file):
        data, _ = librosa.load(std_speech_file, sr=16000, mono=True)
        data = float2pcm(data).tobytes()
    else:
        data = None

    if data is not None:
        await wb.send_bytes(data)
        increment_user_listen_count(in_.username, in_.transcript)
        return
    else:
        ret.server_response = dict(message="Standard speech not available")
        await wb.send_json(ret.dict())
        return
