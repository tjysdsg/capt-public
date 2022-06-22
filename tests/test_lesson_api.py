from fastapi.testclient import TestClient
from gop_server.error import ErrorCode
from tests.utils import (
    get_account_token, create_test_fixture, NORMAL_TEST_USER, ADMIN_TEST_USER
)
import pytest


@pytest.fixture
def app_ctx():
    ret = create_test_fixture()

    client: TestClient = ret['client']
    ret['student_token'] = get_account_token(client, NORMAL_TEST_USER['username'], NORMAL_TEST_USER['password'])
    ret['teacher_token'] = get_account_token(client, ADMIN_TEST_USER['username'], ADMIN_TEST_USER['password'])
    return ret


def test_get(app_ctx):
    client = app_ctx['client']
    response = client.get('/lessons')
    assert response.status_code == 200
    res = response.json()
    assert res['status'] == ErrorCode.SUCCESS
    lessons = res['lessons']
    assert len(lessons) == 1
    assert lessons[0]['lesson_name'] == 'test lesson'


def test_create(app_ctx):
    client = app_ctx['client']
    student_token = app_ctx['student_token']
    teacher_token = app_ctx['teacher_token']

    # test creating a lesson
    response = client.post("/lessons",
                           json=dict(
                               username='test',
                               token=teacher_token,
                               lesson_name='test 1',
                               sentences=[dict(transcript='你好'), dict(transcript='再见')],
                           ))
    assert response.status_code == 200
    res = response.json()
    assert res['status'] == ErrorCode.SUCCESS

    # check if actually created
    response = client.get('/lessons')
    res = response.json()
    lessons = res['lessons']
    assert len(lessons) == 2
    assert lessons[0]['lesson_name'] == 'test 1'  # ordered by name so "test 1" is the first one

    sentences = lessons[0]['sentences']
    assert sentences[0]['transcript'] == '你好'
    assert sentences[1]['transcript'] == '再见'

    # test creating a lesson without permission
    response = client.post("/lessons",
                           json=dict(
                               username='tjysdsg',
                               token=student_token,
                               lesson_name='test 2',
                               sentences=[dict(transcript='你好'), dict(transcript='再见')],
                           ))
    assert response.status_code == 200
    res = response.json()
    assert res['status'] == ErrorCode.PERMISSION_DENIED


def test_update(app_ctx):
    client = app_ctx['client']
    student_token = app_ctx['student_token']
    teacher_token = app_ctx['teacher_token']

    # test creating a lesson
    response = client.put("/lessons",
                          json=dict(
                              username='test',
                              token=teacher_token,
                              lesson_id=1,
                              lesson_name='updated name',
                              sentences=[dict(transcript='你好呀'), dict(transcript='再见吧')],
                          ))
    assert response.status_code == 200
    res = response.json()
    assert res['status'] == ErrorCode.SUCCESS

    # check if actually updated
    response = client.get('/lessons')
    res = response.json()
    lessons = res['lessons']
    assert len(lessons) == 1
    assert lessons[0]['lesson_name'] == 'updated name'

    sentences = lessons[0]['sentences']
    assert sentences[0]['transcript'] == '你好呀'
    assert sentences[1]['transcript'] == '再见吧'

    # test updating a lesson without permission
    response = client.put("/lessons",
                          json=dict(
                              username='tjysdsg',
                              token=student_token,
                              lesson_id=1,
                              lesson_name='updated name',
                              sentences=[dict(transcript='你好呀'), dict(transcript='再见吧')],
                          ))
    assert response.status_code == 200
    res = response.json()
    assert res['status'] == ErrorCode.PERMISSION_DENIED


def test_delete(app_ctx):
    client = app_ctx['client']
    student_token = app_ctx['student_token']
    teacher_token = app_ctx['teacher_token']

    response = client.delete("/lessons",
                             json=dict(username='test',
                                       token=teacher_token,
                                       lesson_id=1
                                       )
                             )
    assert response.status_code == 200
    res = response.json()
    assert res['status'] == ErrorCode.SUCCESS

    # check if actually deleted
    response = client.get('/lessons')
    res = response.json()
    lessons = res['lessons']
    assert len(lessons) == 0

    # test deleting a lesson without permission
    response = client.delete("/lessons",
                             json=dict(
                                 username='tjysdsg',
                                 token=student_token,
                                 lesson_id=1,
                             ))
    assert response.status_code == 200
    res = response.json()
    assert res['status'] == ErrorCode.PERMISSION_DENIED
