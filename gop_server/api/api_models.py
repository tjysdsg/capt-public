from pydantic import BaseModel, Field, constr
from typing import List
from typing_extensions import Literal
from gop_server.db_model_config import (
    username_max_length, username_min_length, real_name_max_length,
    real_name_min_length, password_max_length, password_min_length
)
from gop_server.error import ErrorCode

username_constr = constr(min_length=username_min_length, max_length=username_max_length, regex=r'^[a-zA-Z_0-9]+$')
real_name_constr = constr(min_length=real_name_min_length, max_length=real_name_max_length)
password_constr = constr(min_length=password_min_length, max_length=password_max_length)


class BaseResponseModel(BaseModel):
    status: ErrorCode = ErrorCode.SUCCESS
    message: str = ''


class GetPinyinResponse(BaseResponseModel):
    transcript: str
    pinyin: List[str]


class Sentence(BaseModel):
    id: int
    transcript: str
    explanation: str = ''


class Lesson(BaseModel):
    id: int
    lesson_name: str
    sentences: List[Sentence]


class GetLessonsResponse(BaseResponseModel):
    lessons: List[Lesson]


class RegisterInput(BaseModel):
    username: username_constr
    real_name: real_name_constr = 'unknown'
    password: password_constr


class RegisterResponse(BaseResponseModel):
    token: str


class LoginResponse(BaseResponseModel):
    token: str


class LoginInput(BaseModel):
    username: username_constr
    password: password_constr


class Authenticated(BaseModel):
    """
    Data model that contains authentication info
    """
    username: username_constr
    token: str


class ComputeGOPInput(Authenticated):
    ext: Literal['webm', 'wav', 'm4a', 'caf'] = 'webm'
    sentence_id: int = Field(ge=0)
    lang: Literal['zh', 'eng'] = 'zh'


class ArbitraryGOPInput(BaseModel):
    ext: Literal['webm', 'wav', 'm4a', 'caf'] = 'webm'
    transcript: str
    lang: Literal['zh', 'eng'] = 'zh'


class ComputeGOPResponse(BaseResponseModel):
    data: List = []


class TokenTestInput(Authenticated):
    pass


class TokenTestResponse(BaseResponseModel):
    valid: bool


class TranscriptInput(Authenticated):
    """
    This needs authentication because we want to count a user's std speech listening count
    """
    transcript: constr(min_length=1)


class WSMessageResponse(BaseResponseModel):
    server_response: dict = None


class GetClientLatestVersionResponse(BaseModel):
    version: str


class CreateFeedbackInput(Authenticated):
    sentence_id: int
    content: str


class QuickHelpInput(Authenticated):
    sentence_id: int


class CreateFeedbackResponse(BaseResponseModel):
    pass


class _GetFeedbackResponse(BaseModel):
    username: username_constr
    content: str


class GetFeedbackResponse(BaseResponseModel):
    feedbacks: List[_GetFeedbackResponse]


class _Sentence(BaseModel):
    transcript: str


class CreateLessonInput(Authenticated):
    username: username_constr
    token: str
    lesson_name: str
    sentences: List[_Sentence]


class UpdateLessonInput(CreateLessonInput):
    lesson_id: int


class DeleteLessonInput(Authenticated):
    lesson_id: int


class CreateUpdateDeleteLessonResponse(BaseResponseModel):
    pass
