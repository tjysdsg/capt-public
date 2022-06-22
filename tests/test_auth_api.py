from fastapi.testclient import TestClient
from gop_server.error import ErrorCode
from gop_server.app import create_app
from gop_server.api.auth import verify_token_for_user
import pytest


@pytest.fixture
def client():
    from tests.utils import get_test_db_url
    app = create_app(db_url=get_test_db_url())
    ret = TestClient(app)
    return ret


def test_register(client):
    response = client.post(
        "/register",
        json=dict(username='tjysdsg', password='tjysdsg111-000', real_name='Jiyang Tang')
    )
    assert response.status_code == 200
    res = response.json()
    print(res)
    assert res['status'] == ErrorCode.SUCCESS
    token = res['token']
    assert verify_token_for_user('tjysdsg', token)

    # invalid username
    response = client.post("/register",
                           json=dict(username='t.j*ysdsg', password='tjysdsg111-000', real_name='Jiyang Tang'))
    assert response.status_code != 200


def test_register_without_real_name(client):
    response = client.post(
        "/register",
        json=dict(username='tjysdsg', password='tjysdsg111-000')
    )
    assert response.status_code == 200
    res = response.json()
    assert res['status'] == ErrorCode.SUCCESS
    token = res['token']
    assert verify_token_for_user('tjysdsg', token)


def test_login(client):
    test_register(client)
    response = client.post("/login", json=dict(username='tjysdsg', password='tjysdsg111-000'))
    assert response.status_code == 200
    res = response.json()
    print(res)
    assert res['status'] == ErrorCode.SUCCESS
    token = res['token']
    assert verify_token_for_user('tjysdsg', token)
