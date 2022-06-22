import io
import uuid
import os
import logging
from fastapi import WebSocket, APIRouter
from typing import List
import numpy as np
import librosa
from pydub import AudioSegment
import soundfile as sf
from gop_server.preprocess import float2pcm
from gop_server.error import ErrorCode

router = APIRouter()

logger = logging.getLogger('gop-server')


@router.get("/")
def root():
    return {"message": "Server is up, find API documentation at /docs"}


def wav_data_loader(data: bytes, save_path: str):
    with open(save_path, 'wb') as f:
        f.write(data)
    wav_fs = io.BytesIO(data)
    data, sr = librosa.load(wav_fs, sr=16000, mono=True)
    return float2pcm(data)


def pydub_data_loader(data: bytes, ext: str, save_path: str):
    ifs = io.BytesIO(data)
    audio = AudioSegment.from_file(ifs, format=ext)
    audio.export(save_path, format='wav')
    return np.asarray(audio.get_array_of_samples())


# TODO: simplify this
def webm_data_loader(data: bytes, save_path: str):
    ifs = io.BytesIO(data)
    audio = AudioSegment.from_file(ifs, format='webm')
    wav_fs = io.BytesIO()
    audio.export(wav_fs, format='wav')
    wav_fs.seek(0)
    data, sr = librosa.load(wav_fs, sr=16000, mono=True)
    sf.write(save_path, data, sr, 'PCM_16')
    return float2pcm(data)


async def load_audio(wb, ext: str, save_path: str):
    data = await wb.receive_bytes()
    if ext == "webm":
        data = webm_data_loader(data, save_path)
    elif ext in ['m4a', 'caf']:
        data = pydub_data_loader(data, ext, save_path)
    elif ext == "wav":
        data = wav_data_loader(data, save_path)
    else:
        assert False

    return data


def postprocess_result(res: dict):
    annotations_a: List[str] = res['annotations_a']
    tones: np.ndarray = res['tones']

    # data sent to client is (list of annotations, list of correctness, list of tone indices)
    corr: np.ndarray = res['corr']
    return [annotations_a, corr.tolist(), tones.tolist()]


def save_result_to_db(res: dict, username: str, sentence, save_path: str):
    from gop_server.db import User, Utterance, Phone

    phone_ids: np.ndarray = res['annotations']
    gop: np.ndarray = res['gop']
    tones: np.ndarray = res['tones']
    corr: np.ndarray = res['corr']

    user = User.objects(username=username)[0]
    n = len(phone_ids)
    phones = [
        Phone(
            expected_phone_id=phone_ids[i],
            phone_score=gop[i, 0], phone_correctness=corr[i, 0],
            tone_score=tones[i], tone_correctness=corr[i, 1],
        )
        for i in range(n)
    ]
    Utterance(user=user, sentence=sentence, filename=save_path, phones=phones).save()


async def send_websocket_error_response(wb, res, error_code: ErrorCode, msg: str, log=True):
    res.status = error_code
    res.message = msg
    if log:
        logger.error(msg)
    await wb.send_json(res.dict())
    return res


@router.websocket("/upload")
async def upload_audio(wb: WebSocket):
    from gop_server import lang_configs, audio_save_path
    from pydantic import ValidationError
    from gop_server.api.api_models import ComputeGOPInput, ComputeGOPResponse
    from gop_server.db import Sentence

    ret = ComputeGOPResponse(status=ErrorCode.SUCCESS)

    # 1. metadata
    await wb.accept()
    metadata: dict = await wb.receive_json()

    try:
        in_: ComputeGOPInput = ComputeGOPInput.parse_obj(metadata)
    except ValidationError as e:
        await send_websocket_error_response(wb, ret, ErrorCode.INVALID_INPUT, str(e))
        return

    # get infos
    ext = in_.ext
    sentence_id = in_.sentence_id
    sentence: Sentence = Sentence.objects(sentence_id=sentence_id).first()
    if sentence is None:
        ret.status = ErrorCode.INVALID_INPUT
        ret.message = 'Invalid sentence id'
        await wb.send_json(ret.dict())
        return
    transcript = sentence.transcript
    pinyin = sentence.pinyin.split() if sentence.pinyin != '' else None

    # check auth
    username = in_.username
    token = in_.token
    lang: str = in_.lang
    from gop_server.api.auth import verify_token_for_user
    if not verify_token_for_user(username, token):
        await send_websocket_error_response(wb, ret, ErrorCode.INVALID_INPUT, 'Invalid username or token')
        return

    # 2. load data
    # build save path
    save_dir = os.path.join(audio_save_path, lang)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{username}-{transcript}-{str(uuid.uuid4())}.wav')

    data = await load_audio(wb, ext, save_path)

    # main routine
    try:
        lang_config = lang_configs.get(lang)
        res = lang_config.handler(data, transcript, pinyin)
        ret.data = postprocess_result(res)
        save_result_to_db(res, username, sentence, save_path)
    except Exception as e:
        from gop_server.utils import get_error_str
        await send_websocket_error_response(wb, ret, ErrorCode.SERVER_ERROR, get_error_str(e))
        return
    logger.info('Computed GOP successfully')
    await wb.send_json(ret.dict())


# TODO: support input: {"pinyin": "n i_2 h ao_3", xxx}
@router.websocket("/gop")
async def compute_gop(wb: WebSocket):
    from gop_server import lang_configs, audio_save_path
    from gop_server.error import ErrorCode
    from pydantic import ValidationError
    from gop_server.api.api_models import ArbitraryGOPInput, ComputeGOPResponse

    ret = ComputeGOPResponse(status=ErrorCode.SUCCESS)

    await wb.accept()

    # 1. metadata
    metadata: dict = await wb.receive_json()
    try:
        in_: ArbitraryGOPInput = ArbitraryGOPInput.parse_obj(metadata)
    except ValidationError as e:
        await send_websocket_error_response(wb, ret, ErrorCode.INVALID_INPUT, str(e))
        return

    # get infos
    ext = in_.ext
    transcript = in_.transcript
    lang: str = in_.lang

    # 2. load data
    # build save path
    save_dir = os.path.join(audio_save_path, lang)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'tmp-{str(uuid.uuid4())}.wav')
    data = await load_audio(wb, ext, save_path)

    # main routine
    try:
        lang_config = lang_configs.get(lang)
        res = lang_config.handler(data, transcript, None)

        phones = res['annotations_a']

        gop = res['gop'][:, 0]  # not including tone gop
        gop = gop.tolist()

        corr = res['corr'][:, 0]  # not including tone
        corr = corr.tolist()

        tones = res['tones'].tolist()

        ret.data = dict(pinyin=phones, gop=gop, corr=corr, tones=tones)
    except Exception as e:
        from gop_server.utils import get_error_str
        await send_websocket_error_response(wb, ret, ErrorCode.SERVER_ERROR, get_error_str(e))
        return

    await wb.send_json(ret.dict())
