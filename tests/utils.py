from fastapi.testclient import TestClient
from requests import Response
from gop_server.error import ErrorCode

NORMAL_TEST_USER = dict(username='tjysdsg', password='test_password', real_name='tjysdsg')
ADMIN_TEST_USER = dict(username='test', password='test-test-test', real_name='admin')


def get_test_db_url():
    import uuid
    return f'mongomock://localhost/{uuid.uuid4()}'


def get_account_token(client: TestClient, username: str, password: str) -> str:
    response = client.post("/login", json=dict(username=username, password=password))
    assert response.status_code == 200
    res = response.json()
    assert res['status'] == ErrorCode.SUCCESS
    return res['token']


def get_check_response_json(response: Response, error_code=ErrorCode.SUCCESS) -> dict:
    assert response.status_code == 200
    res = response.json()
    assert res['status'] == error_code
    return res


def init_test_lessons():
    from gop_server.db import (Sentence, Lesson)
    lesson = Lesson(lesson_id=1, lesson_name='test lesson').save()
    Sentence(sentence_id=1, transcript='你好', pinyin='n i_2 h ao_3', lesson=lesson).save(),
    Sentence(sentence_id=2, transcript='一本书', pinyin='y i_4 b en_3 sh u_1', lesson=lesson).save()

    return {
        '你好': 1,
        '一本书': 2,
    }


def create_test_fixture():
    from gop_server.app import create_app
    from gop_server.db import init_test_account

    app = create_app(db_url=get_test_db_url())

    init_test_account()
    lessons = init_test_lessons()

    client = TestClient(app)

    # setup test account
    response = client.post("/register", json=NORMAL_TEST_USER)
    assert response.status_code == 200
    res = response.json()
    assert res['status'] == ErrorCode.SUCCESS, res['message']

    ret = dict(token=res['token'], client=client, lessons=lessons)
    ret.update(NORMAL_TEST_USER)
    return ret
