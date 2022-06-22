from fastapi import Query, APIRouter
from gop_server.error import ErrorCode
from gop_server.api.api_models import (
    GetPinyinResponse, GetLessonsResponse, CreateLessonInput, UpdateLessonInput,
    CreateUpdateDeleteLessonResponse, DeleteLessonInput
)
from gop_server.api.auth import verify_teacher_account
from gop_server.db import Lesson, Sentence

router = APIRouter()


# FIXME: after deletion and creation, new lesson_id/sentence_id might not be unique

def _check_teacher_account(username, token, response_class):
    if verify_teacher_account(username, token):
        return response_class(status=ErrorCode.SUCCESS, message='')
    else:
        return response_class(status=ErrorCode.PERMISSION_DENIED, message="You don't have permission to do this")


@router.get("/pinyin", response_model=GetPinyinResponse)
def get_pinyin(transcript: str = Query(..., max_length=50, min_length=1)):
    try:
        from gop_server.transcript import clean_transcript
        from gop_server.pinyin import transcript_to_pinyin
        transcript, segments = clean_transcript(transcript)
        pinyin = transcript_to_pinyin(segments, style='human')
    except NameError:
        return GetPinyinResponse(status=ErrorCode.SERVER_ERROR, transcript='', pinyin=[])
    return GetPinyinResponse(status=ErrorCode.SUCCESS, transcript=transcript, pinyin=pinyin)


@router.get("/lessons", response_model=GetLessonsResponse)
def get_lessons():
    from gop_server.db import Lesson, Sentence
    from gop_server.api.api_models import Lesson as LessonModel, Sentence as SentenceModel
    ret: [LessonModel] = []
    for lesson in Lesson.objects().order_by('lesson_name'):  # type: Lesson
        sentences: [SentenceModel] = []

        for s in Sentence.objects(lesson=lesson):  # type: Sentence
            sentences.append(SentenceModel(id=s.sentence_id, transcript=s.transcript, explanation=s.explanation))
        ret.append(LessonModel(id=lesson.lesson_id, lesson_name=lesson.lesson_name, sentences=sentences))

    return GetLessonsResponse(status=ErrorCode.SUCCESS, lessons=ret)


@router.post("/lessons", response_model=CreateUpdateDeleteLessonResponse)
def create_lesson(in_: CreateLessonInput):
    if not verify_teacher_account(in_.username, in_.token):
        return CreateUpdateDeleteLessonResponse(
            status=ErrorCode.PERMISSION_DENIED,
            message="You don't have permission to do this"
        )

    from gop_server.db import Lesson, Sentence
    lesson = Lesson(lesson_id=Lesson.objects().count() + 1, lesson_name=in_.lesson_name).save()
    for s in in_.sentences:
        Sentence(sentence_id=Sentence.objects().count() + 1, transcript=s.transcript, lesson=lesson).save()
    return CreateUpdateDeleteLessonResponse()


@router.put("/lessons", response_model=CreateUpdateDeleteLessonResponse)
def update_lesson(in_: UpdateLessonInput):
    ret = CreateUpdateDeleteLessonResponse()

    if not verify_teacher_account(in_.username, in_.token):
        ret.status = ErrorCode.PERMISSION_DENIED
        ret.message = "You don't have permission to do this"
        return ret

    lesson: Lesson = Lesson.objects(lesson_id=in_.lesson_id).first()
    if lesson is None:
        ret.status = ErrorCode.INVALID_INPUT
        ret.message = f'Cannot find lesson with id={in_.lesson_id}'
        return ret

    # update lesson name
    lesson.lesson_name = in_.lesson_name
    lesson.save()

    # delete old sentences
    for p in Sentence.objects(lesson=lesson):
        p.delete()

    # update sentences
    for s in in_.sentences:
        Sentence(sentence_id=Sentence.objects().count() + 1, transcript=s.transcript, lesson=lesson).save()
    return ret


@router.delete("/lessons", response_model=CreateUpdateDeleteLessonResponse)
def delete_lesson(in_: DeleteLessonInput):
    ret = CreateUpdateDeleteLessonResponse()

    if not verify_teacher_account(in_.username, in_.token):
        ret.status = ErrorCode.PERMISSION_DENIED
        ret.message = "You don't have permission to do this"
        return ret

    lesson: Lesson = Lesson.objects(lesson_id=in_.lesson_id).first()
    if lesson is None:
        ret.status = ErrorCode.INVALID_INPUT
        ret.message = f'Cannot find lesson with id={in_.lesson_id}'
        return ret

    lesson.delete()
    return ret
