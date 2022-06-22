import os
import json
from gop_server import zh_config


def read_phone_table(path: str):
    ret = []
    with open(path) as f:
        for line in f:
            tokens = line.strip('\n').split()
            phone = tokens[0]
            ret.append(phone)

    return ret


phone_table_path = os.path.join(zh_config.root_path, 'phones.txt')
all_phones = read_phone_table(phone_table_path)


def test_word_lists():
    """
    1. Check if the phones in pinyins are in the phone table
    2. Check if standard pronunciation files exist
    """
    all_lesson_ids = set()
    all_sent_ids = set()

    lesson_dir = os.path.join(zh_config.root_path, 'word_list')
    for entry in os.scandir(lesson_dir):
        if entry.is_file() and entry.name.endswith(".json"):
            data = json.load(open(entry.path))

            sentences = data['sentences']

            # check if sentence id is unique
            lesson_id = data['id']
            assert lesson_id not in all_lesson_ids
            all_lesson_ids.add(lesson_id)

            for sent in sentences:
                # check if sentence id is unique
                sent_id = sent['id']
                assert sent_id not in all_sent_ids
                all_sent_ids.add(sent_id)

                text = sent['text']
                pinyin = sent['pinyin']

                # check if pinyin is in phone table
                for p in pinyin.split():
                    assert p in all_phones, f'{entry.name}: Invalid phone {p} in sentence #{sent_id}'

                # check std pronun file
                assert os.path.isfile(os.path.join(lesson_dir, f'{text}.wav')), f'Cannot find {text}.wav'
