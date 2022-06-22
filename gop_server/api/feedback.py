from fastapi import APIRouter, Query
from gop_server.api.api_models import CreateFeedbackInput, QuickHelpInput, CreateFeedbackResponse, GetFeedbackResponse

router = APIRouter()

quick_help_str = "Need help"


@router.post("/quick-help", response_model=CreateFeedbackResponse)
def create_feedback(in_: QuickHelpInput):
    """
    Shortcut to add a feedback with "Need help" as message content
    """
    args = CreateFeedbackInput(username=in_.username, token=in_.token, content=quick_help_str,
                               sentence_id=in_.sentence_id)
    return create_feedback(args)


@router.post("/feedback", response_model=CreateFeedbackResponse)
def create_feedback(in_: CreateFeedbackInput):
    from gop_server.db import Feedback, Sentence, User
    from gop_server.error import ErrorCode
    from gop_server.api.auth import verify_token_for_user

    if not verify_token_for_user(token=in_.token, username=in_.username):
        return CreateFeedbackResponse(status=ErrorCode.INVALID_INPUT, message='Invalid login token')

    sentence = Sentence.objects(sentence_id=in_.sentence_id).first()
    if sentence is None:
        return CreateFeedbackResponse(status=ErrorCode.INVALID_INPUT, message='Invalid sentence id')

    author = User.objects(username=in_.username).first()
    if sentence is None:
        return CreateFeedbackResponse(status=ErrorCode.INVALID_INPUT, message='Invalid username')

    Feedback(sentence=sentence, content=in_.content, author=author).save()
    return CreateFeedbackResponse(status=ErrorCode.SUCCESS)


@router.get("/feedback", response_model=GetFeedbackResponse)
def get_feedback(sentence_id: int = Query(...), username: str = Query(...), token: str = Query(...)):
    from gop_server.db import Feedback, User, Sentence, Q
    from gop_server.error import ErrorCode
    from gop_server.api.auth import verify_token_for_user
    ret = []

    user = User.objects(username=username).first()

    # check if user exists
    if user is None:
        return GetFeedbackResponse(
            status=ErrorCode.PERMISSION_DENIED,
            message='Only teacher account can access feedback', feedbacks=[]
        )
    # check if token is correct
    if not verify_token_for_user(username, token):
        return GetFeedbackResponse(
            status=ErrorCode.INVALID_INPUT, message='Invalid username or login token',
            feedbacks=[]
        )

    sentence = Sentence.objects(sentence_id=sentence_id).first()
    if sentence is None:
        return GetFeedbackResponse(
            status=ErrorCode.INVALID_INPUT,
            message='Invalid sentence id', feedbacks=[]
        )

    for f in Feedback.objects(sentence=sentence):
        author: User = User.objects(id=f.author.id).first()
        assert author is not None

        # return this feedback if:
        # - current user is a teacher
        # - or current user is the author
        # - or the author of this feedback is a teacher
        if user.role == 'teacher' or user.id == author.id or author.role == 'teacher':
            ret.append(
                dict(content=f.content, sentence_id=sentence_id, username=author.username)
            )
    return GetFeedbackResponse(status=ErrorCode.SUCCESS, feedbacks=ret)
