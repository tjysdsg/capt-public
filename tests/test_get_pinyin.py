from fastapi.testclient import TestClient
from gop_server.error import ErrorCode
from gop_server.app import create_app
from urllib.parse import quote_plus
from tests.utils import get_test_db_url

app = create_app(get_test_db_url())
client = TestClient(app)

test_cases = [
    ('你好！', ['n', 'í', 'h', 'ǎo']),
    ('请问，你贵姓？', ['q', 'ǐng', 'w', 'èn', 'n', 'ǐ', 'g', 'uì', 'x', 'ìng']),
    ('我姓李。你呢？', ['w', 'ǒ', 'x', 'ìng', 'l', 'ǐ', 'n', 'ǐ', 'n', 'e']),
    ('我姓王。李小姐，你叫什么名字？',
     ['w', 'ǒ', 'x', 'ìng', 'w', 'áng', 'l', 'ǐ', 'x', 'iáo', 'j', 'iě', 'n', 'ǐ', 'j', 'iào', 'sh', 'én', 'm', 'e',
      'm', 'íng', 'z', 'ì']),  # TODO: 'zi'
    ('我叫李友。王先生，你叫什么名字？',
     ['w', 'ǒ', 'j', 'iào', 'l', 'í', 'y', 'ǒu', 'w', 'áng', 'x', 'iān', 'sh', 'ēng', 'n', 'ǐ', 'j', 'iào', 'sh', 'én',
      'm', 'e', 'm', 'íng', 'z', 'ì']),  # TODO: 'zi'
    ('我叫王朋。', ['w', 'ǒ', 'j', 'iào', 'w', 'áng', 'p', 'éng']),
]


def test_get_pinyin():
    for trans, py in test_cases:
        response = client.get(f'/pinyin?transcript={quote_plus(trans)}')
        res = response.json()
        assert response.status_code == 200
        assert res['status'] == ErrorCode.SUCCESS
        assert res['pinyin'] == py

    # missing query parameter
    response = client.get('/pinyin')
    assert response.status_code != 200

    # too short
    response = client.get('/pinyin?transcript=')
    assert response.status_code != 200

    # too long
    response = client.get('/pinyin?transcript=dskfjoashdfojcxkjvljdsiajfoiasjofdjodsjkxjvkjcxjvlkcxjlvkjcxlkjcvjcxkjv'
                          'fjdlksjfksdjlkvcxlkvjkcxjvkjdsioewijfkjvlkxjclkvjxclkj')
    assert response.status_code != 200
