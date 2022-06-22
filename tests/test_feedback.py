from fastapi.testclient import TestClient
from gop_server.error import ErrorCode
from tests.utils import (
    get_account_token, get_check_response_json, create_test_fixture, NORMAL_TEST_USER,
    ADMIN_TEST_USER
)
import pytest

feedback_str = 'I hate this app'


# TODO test these:
#  - normal users can only see teacher feedbacks and their own feedbacks
#  - teacher can see all feedbacks

@pytest.fixture
def app_ctx():
    ret = create_test_fixture()

    client: TestClient = ret['client']
    ret['student_token'] = get_account_token(client, NORMAL_TEST_USER['username'], NORMAL_TEST_USER['password'])
    ret['teacher_token'] = get_account_token(client, ADMIN_TEST_USER['username'], ADMIN_TEST_USER['password'])
    return ret


def create_feedback(client, username: str, token: str):
    response = client.post("/feedback",
                           json=dict(username=username,
                                     token=token,
                                     content=feedback_str,
                                     sentence_id=1)
                           )
    get_check_response_json(response)


def test_feedback(app_ctx):
    client = app_ctx['client']
    teacher_token = app_ctx['teacher_token']

    # test creating feedback
    create_feedback(client, ADMIN_TEST_USER['username'], teacher_token)

    # test getting feedback
    response = client.get(f"/feedback?sentence_id=1&token={teacher_token}&username={ADMIN_TEST_USER['username']}")
    res = get_check_response_json(response)
    feedback = res['feedbacks'][0]
    assert feedback['content'] == feedback_str
    assert feedback['username'] == ADMIN_TEST_USER['username']

    # get feedback invalid token
    response = client.get(f"/feedback?sentence_id=1&token=fake-token-here&username={ADMIN_TEST_USER['username']}")
    get_check_response_json(response, ErrorCode.INVALID_INPUT)

    # create feedback invalid token
    response = client.post("/feedback",
                           json=dict(
                               username=ADMIN_TEST_USER['username'],
                               token='fake-token-hahaha',
                               content=feedback_str,
                               sentence_id=1)
                           )
    assert response.status_code == 200
    res = response.json()
    assert res['status'] != ErrorCode.SUCCESS


def test_student_see_own_feedback(app_ctx):
    client = app_ctx['client']
    student_token = app_ctx['student_token']

    # test creating feedback
    create_feedback(client, NORMAL_TEST_USER['username'], student_token)

    #  test if students can their own feedbacks
    response = client.get(f"/feedback?sentence_id=1&token={student_token}&username={NORMAL_TEST_USER['username']}")
    res = get_check_response_json(response)

    feedbacks = res['feedbacks']
    assert len(feedbacks) > 0
    fb = feedbacks[0]
    assert fb['content'] == feedback_str
    assert fb['username'] == NORMAL_TEST_USER['username']


def test_quick_help(app_ctx):
    from gop_server.api.feedback import quick_help_str
    client = app_ctx['client']
    token = app_ctx['teacher_token']

    # test creating feedback
    response = client.post("/quick-help", json=dict(username=ADMIN_TEST_USER['username'], token=token, sentence_id=1))
    get_check_response_json(response)

    # make sure a feedback with content "Need help" is actually created
    response = client.get(f"/feedback?sentence_id=1&token={token}&username={ADMIN_TEST_USER['username']}")
    res = get_check_response_json(response)
    feedback = res['feedbacks'][0]
    assert feedback['content'] == quick_help_str
    assert feedback['username'] == ADMIN_TEST_USER['username']
