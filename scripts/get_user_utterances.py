import os
import json
from gop_server.db import init_db, User, Utterance
from gop_server import zh_config


def search(username: str):
    user = User.objects(username=username).first()
    assert user is not None

    q = Utterance.objects(user=user)
    for e in q:
        print(f'{e.sentence.lesson.lesson_name}: {e.sentence.transcript}')


if __name__ == '__main__':
    init_db()
    search('hung')
