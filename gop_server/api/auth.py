import uuid
from fastapi import APIRouter
from gop_server.error import ErrorCode
from gop_server.api.api_models import (
    TokenTestResponse, TokenTestInput, RegisterResponse, RegisterInput,
    LoginResponse, LoginInput
)

router = APIRouter()


def get_user_id_from_username(username: str) -> int:
    from gop_server.db import User
    user = User.objects(username=username).first()
    return user.id


def get_username_from_user_id(user_id: int) -> str:
    from gop_server.db import User
    user = User.objects(id=user_id).first()
    return user.username


def encode_password(password: str) -> str:
    from passlib.hash import bcrypt
    return bcrypt.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    from passlib.hash import bcrypt
    return bcrypt.verify(password, hashed)


def verify_token_for_user(username: str, token: str) -> bool:
    from gop_server.db import User

    user = User.objects(username=username).first()
    if user is None:
        return False
    return user.token.token == token


def verify_teacher_account(username: str, token: str) -> bool:
    from gop_server.db import User, Q

    user = User.objects(Q(username=username) & Q(role='teacher')).first()
    if user is None:
        return False
    return user.token.token == token


@router.post('/token_test', response_model=TokenTestResponse)
def token_test(in_: TokenTestInput):
    try:
        valid = verify_token_for_user(in_.username, in_.token)
        return TokenTestResponse(status=ErrorCode.SUCCESS, valid=valid)
    except Exception as e:
        from gop_server.utils import get_error_str
        msg = get_error_str(e)
        return TokenTestResponse(status=ErrorCode.SERVER_ERROR, valid=False, message=msg)


@router.post('/register', response_model=RegisterResponse)
def register(in_: RegisterInput):
    from gop_server.db import User, Token
    try:
        token_str = str(uuid.uuid4())
        User(
            username=in_.username, real_name=in_.real_name, password=encode_password(in_.password),
            token=Token(token=token_str)
        ).save()
    except Exception as e:
        from gop_server.utils import get_error_str
        msg = get_error_str(e)
        return RegisterResponse(status=ErrorCode.INVALID_INPUT, token='', message=msg)

    return RegisterResponse(status=ErrorCode.SUCCESS, token=token_str)


@router.post('/login', response_model=LoginResponse)
def login(in_: LoginInput):
    from gop_server.db import User

    ret = LoginResponse(status=ErrorCode.INVALID_INPUT, token='')

    user: User = User.objects(username=in_.username).first()
    if user is None or not verify_password(in_.password, user.password):
        return ret

    assert user.token is not None
    return LoginResponse(status=ErrorCode.SUCCESS, token=user.token.token, message='')
