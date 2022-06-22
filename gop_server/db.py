import logging
from gop_server.db_model_config import (username_max_length, real_name_max_length)
import mongoengine
from mongoengine import *

logger = logging.getLogger('gop-server')


class Token(EmbeddedDocument):
    token = StringField()


class User(Document):
    username = StringField(max_length=username_max_length, unique=True)
    real_name = StringField(max_length=real_name_max_length, null=True)
    password = StringField()  # store encrypted string
    role = StringField(default='student', choices=['student', 'teacher'])
    token = EmbeddedDocumentField(Token)


class Lesson(Document):
    lesson_id = IntField(required=True, unique=True)
    lesson_name = StringField()


class Sentence(Document):
    sentence_id = IntField(required=True, unique=True)
    transcript = StringField(required=True)
    pinyin = StringField(null=True)
    explanation = StringField(default='')
    lesson = ReferenceField('Lesson', reverse_delete_rule='CASCADE')


class Feedback(Document):
    content = StringField()
    author = ReferenceField('User', reverse_delete_rule='CASCADE')
    sentence = ReferenceField('Sentence', reverse_delete_rule='CASCADE')


class Phone(EmbeddedDocument):
    expected_phone_id = IntField()
    phone_score = FloatField()
    phone_correctness = BooleanField()
    tone_correctness = BooleanField()
    tone_score = FloatField()


class Utterance(Document):
    user = ReferenceField('User', reverse_delete_rule='CASCADE')
    filename = StringField()
    sentence = ReferenceField('Sentence', reverse_delete_rule='CASCADE')
    phones = EmbeddedDocumentListField(Phone)


class UserListenCount(Document):
    user = ReferenceField('User', reverse_delete_rule='CASCADE')
    sentence = ReferenceField('Sentence', reverse_delete_rule='CASCADE')
    count = IntField()


def init_test_account():
    """
    Create a test user if not exist
    """
    from gop_server.api.auth import encode_password
    import uuid

    if User.objects(username='test').first() is None:
        User(
            username='test', password=encode_password('test-test-test'), token=Token(token=str(uuid.uuid4())),
            real_name='test', role='teacher'
        ).save()


# TODO: load db url from file
def init_db(db_url=""):  # TODO: your database URL
    mongoengine.disconnect_all()
    mongoengine.connect(host=db_url)


# TODO: use sentence_id to select sentences
def get_user_listen_count(username: str, transcript: str):
    user = User.objects(username=username).first()
    sentence = Sentence.objects(transcript=transcript).first()
    listen_count = UserListenCount.objects(user=user, sentence=sentence).first()

    return listen_count, user, sentence


# TODO: use sentence_id to select sentences
def increment_user_listen_count(username: str, transcript: str):
    q, user, sentence = get_user_listen_count(username, transcript)

    assert user is not None
    assert sentence is not None

    if q is None:
        new = UserListenCount(
            user=user,
            sentence=sentence,
            count=1,
        )
        new.save()
    else:
        q.update(inc__count=1)
