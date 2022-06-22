from fastapi.websockets import WebSocket
import os
from os.path import dirname, abspath
from gop_server.error import ErrorCode
from gop_server.db import Sentence, Lesson, Utterance
from tests.utils import create_test_fixture
import pytest

file_dir = dirname(abspath(__file__))
audio_dir = os.path.join(file_dir, 'audio')


@pytest.fixture
def app_ctx():
    return create_test_fixture()


def check_db():
    sentence = Sentence.objects(sentence_id=1)[0]
    q = Utterance.objects(sentence=sentence)[0].phones
    q = list(q)
    assert len(q) == 4
    q[0].expected_phone_id = 114  # n
    q[1].expected_phone_id = 63  # ǐ
    q[2].expected_phone_id = 59  # h
    q[3].expected_phone_id = 26  # ǎo

    for p in q:
        assert p.phone_score is not None
        assert p.tone_score is not None
        assert p.phone_correctness is not None


def do_test(app_ctx, audio_name: str, audio_fmt: str, transcript: str):
    client = app_ctx['client']
    token = app_ctx['token']
    lessons = app_ctx['lessons']

    with open(os.path.join(audio_dir, audio_name), 'rb') as audio_file:
        audio_data = audio_file.read()

    with client.websocket_connect("/upload") as wb:
        wb: WebSocket = wb
        wb.send_json(
            {'ext': audio_fmt, 'lang': 'zh', 'username': 'tjysdsg', 'token': token, 'sentence_id': lessons[transcript]})
        wb.send_bytes(audio_data)
        data: dict = wb.receive_json()
        print(data)
        assert data.get('status') == ErrorCode.SUCCESS


def test_wav(app_ctx):
    do_test(app_ctx, 'test_wav1.wav', 'wav', '你好')
    check_db()


def test_webm(app_ctx):
    do_test(app_ctx, 'test_wav1.webm', 'webm', '你好')
    check_db()


def test_m4a(app_ctx):
    do_test(app_ctx, 'test_wav1.m4a', 'm4a', '你好')
    check_db()


def test_caf(app_ctx):
    do_test(app_ctx, 'test_wav1.caf', 'caf', '你好')
    check_db()


"""
def test_nonsense(app_ctx):
    client = app_ctx['client']
    token = app_ctx['token']
    with open(os.path.join(audio_dir, 'test_nonsense.wav'), 'rb') as audio_file:
        audio_data = audio_file.read()
    with client.websocket_connect("/upload") as wb:
        wb: WebSocket = wb
        wb.send_json({'ext': 'wav', 'lang': 'zh', 'username': 'tjysdsg', 'token': token, 'sentence_id': 1})
        wb.send_bytes(audio_data)
        data: dict = wb.receive_json()
        assert data.get('status') == ErrorCode.INVALID_INPUT
"""

"""
def test_parallel(app_ctx, nj=2):  # 2 processes already consume a large amount of memory
    from multiprocessing import Process
    from gop_server.zh_gop import zh_gop_main
    from gop_server.preprocess import float2pcm
    import librosa

    def worker():
        file = os.path.join(audio_dir, '不知道天气会怎么样.wav')
        trans = '不知道天气会怎么样'
        data, sr = librosa.load(file, sr=16000, mono=True)
        data = float2pcm(data)
        zh_gop_main(data, trans)

    ps = [Process(target=worker) for _ in range(nj)]
    for p in ps:
        p.start()

    for p in ps:
        p.join()
"""


def test_arbitrary_gop(app_ctx):
    client = app_ctx['client']

    with open(os.path.join(audio_dir, 'test_wav1.wav'), 'rb') as audio_file:
        audio_data = audio_file.read()

    with client.websocket_connect("/gop") as wb:
        wb: WebSocket = wb
        wb.send_json({'ext': 'wav', 'transcript': '你好'})
        wb.send_bytes(audio_data)
        data: dict = wb.receive_json()
        print(data)
        assert data.get('status') == ErrorCode.SUCCESS


def run_with_transcript_and_pinyin(client, token: str, trans: str, pinyin: str):
    import random
    lesson = Lesson(lesson_id=random.randint(666, 48931784924729), lesson_name='test lesson').save()
    sentence = Sentence(
        sentence_id=random.randint(666, 48931784924729), transcript='送给她什么礼物', pinyin=pinyin, lesson=lesson
    ).save()

    with open(os.path.join(audio_dir, f'{trans}.wav'), 'rb') as audio_file:
        audio_data = audio_file.read()
    with client.websocket_connect("/upload") as wb:
        wb: WebSocket = wb
        wb.send_json({
            'ext': 'wav', 'lang': 'zh', 'username': 'tjysdsg', 'token': token, 'sentence_id': sentence.sentence_id
        })
        wb.send_bytes(audio_data)
        data: dict = wb.receive_json()
        assert data.get('status') == ErrorCode.SUCCESS


def test_with_trans_pinyin(app_ctx):
    client = app_ctx['client']
    token = app_ctx['token']
    run_with_transcript_and_pinyin(client, token, '送给她什么礼物', 's ong_4 g ei_3 t a_1 sh en_2 m e_0 l i_3 w u_4')
    run_with_transcript_and_pinyin(client, token, '还不能回去', 'h i_2 b u_4 n eng_2 h ui_2 q v_4')


def test_special_cases():
    from gop_server.zh_gop import zh_gop_main
    from gop_server.preprocess import float2pcm
    import librosa
    files = [
        os.path.join(audio_dir, '不知道天气会怎么样.wav'),
        os.path.join(audio_dir, '起床.wav'),
        os.path.join(audio_dir, '美国人.wav'),
        os.path.join(audio_dir, '真的啊.wav'),
        os.path.join(audio_dir, '夏天不热.wav'),
        # os.path.join(audio_dir, '李友觉得.wav'),
        # os.path.join(audio_dir, 'test-这里出去很不方便.wav'),
        # os.path.join(audio_dir, '她在学校_denoise.wav'),
    ]
    trans = [
        '不知道天气会怎么样',
        '起床',
        '美国人',
        '真的啊',
        '夏天不热',
        # '李友觉得高文中家很大',
        # '这里出去很不方便',
        # '她在学校工作',
    ]
    for f, t in zip(files, trans):
        data, sr = librosa.load(f, sr=16000, mono=True)
        data = float2pcm(data)
        zh_gop_main(data, t)
