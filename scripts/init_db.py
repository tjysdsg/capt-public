import os
import json
from gop_server.db import init_db, init_test_account
from gop_server import zh_config


def read_phone_table(path: str):
    ret = []
    with open(path) as f:
        for line in f:
            tokens = line.strip('\n').split()
            phone = tokens[0]
            ret.append(phone)

    return ret


def init_textbook_lessons():
    from gop_server.db import Lesson, Sentence

    lesson_dir = os.path.join(zh_config.root_path, 'word_list')
    for entry in os.scandir(lesson_dir):
        if entry.is_file() and entry.name.endswith(".json"):
            data = json.load(open(entry.path))

            lesson_id = data['id']
            lesson_name = data['name']

            lesson: Lesson = Lesson.objects(lesson_id=lesson_id).first()
            if lesson is None:
                lesson = Lesson(lesson_id=lesson_id, lesson_name=lesson_name).save()
            else:
                lesson.update(lesson_name=lesson_name)

            sentences = data['sentences']

            for sent in sentences:
                sent_id = sent['id']
                text = sent['text']
                pinyin = sent['pinyin']
                explanation = sent['explanation']

                # check if pinyin is in phone table
                for p in pinyin.split():
                    assert p in all_phones, f'{entry.name}: Invalid phone {p}'

                sentence: Sentence = Sentence.objects(sentence_id=sent_id).first()
                if sentence is None:
                    Sentence(
                        sentence_id=sent_id, transcript=text, pinyin=pinyin, explanation=explanation, lesson=lesson
                    ).save()
                else:
                    sentence.update(transcript=text, pinyin=pinyin, explanation=explanation, lesson=lesson)


if __name__ == '__main__':
    phone_table_path = 'gop_chinese/phones.txt'
    all_phones = read_phone_table(phone_table_path)

    init_db()
    init_test_account()
    init_textbook_lessons()
