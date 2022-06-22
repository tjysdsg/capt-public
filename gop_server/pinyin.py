import os
import pickle
import re
import itertools
from pypinyin import lazy_pinyin, load_phrases_dict, Style
from typing import List, Tuple, Set, Union
from ltp import LTP
import numpy as np


class _Pinyin:
    INITIALS = ['b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 'sh', 't', 'w', 'x',
                'y',
                'z', 'zh']
    NUMBER_HANS = {'零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十'}
    TONES = {'0', '1', '2', '3', '4'}
    PINYIN_PATTERN = re.compile(r'[a-z]+[0-4]')
    EMPTY_PINYIN = ' '
    CONSONANT_TO_TONE = {
        "a": 0,
        "ā": 1,
        "á": 2,
        "ǎ": 3,
        "à": 4,
        "ai": 0,
        "āi": 1,
        "ái": 2,
        "ǎi": 3,
        "ài": 4,
        "an": 0,
        "ān": 1,
        "án": 2,
        "ǎn": 3,
        "àn": 4,
        "ang": 0,
        "āng": 1,
        "áng": 2,
        "ǎng": 3,
        "àng": 4,
        "ao": 0,
        "āo": 1,
        "áo": 2,
        "ǎo": 3,
        "ào": 4,
        "e": 0,
        "ē": 1,
        "é": 2,
        "ě": 3,
        "è": 4,
        "ei": 0,
        "ēi": 1,
        "éi": 2,
        "ěi": 3,
        "èi": 4,
        "en": 0,
        "ēn": 1,
        "én": 2,
        "ěn": 3,
        "èn": 4,
        "eng": 0,
        "ēng": 1,
        "éng": 2,
        "ěng": 3,
        "èng": 4,
        "er": 0,
        "ēr": 1,
        "ér": 2,
        "ěr": 3,
        "èr": 4,
        "i": 0,
        "ī": 1,
        "í": 2,
        "ǐ": 3,
        "ì": 4,
        "ia": 0,
        "iā": 1,
        "iá": 2,
        "iǎ": 3,
        "ià": 4,
        "ian": 0,
        "iān": 1,
        "ián": 2,
        "iǎn": 3,
        "iàn": 4,
        "iang": 0,
        "iāng": 1,
        "iáng": 2,
        "iǎng": 3,
        "iàng": 4,
        "iao": 0,
        "iāo": 1,
        "iáo": 2,
        "iǎo": 3,
        "iào": 4,
        "ie": 0,
        "iē": 1,
        "ié": 2,
        "iě": 3,
        "iè": 4,
        "in": 0,
        "īn": 1,
        "ín": 2,
        "ǐn": 3,
        "ìn": 4,
        "ing": 0,
        "īng": 1,
        "íng": 2,
        "ǐng": 3,
        "ìng": 4,
        "iong": 0,
        "iōng": 1,
        "ióng": 2,
        "iǒng": 3,
        "iòng": 4,
        "iu": 0,
        "iū": 1,
        "iú": 2,
        "iǔ": 3,
        "iù": 4,
        "o": 0,
        "ō": 1,
        "ó": 2,
        "ǒ": 3,
        "ò": 4,
        "ong": 0,
        "ōng": 1,
        "óng": 2,
        "ǒng": 3,
        "òng": 4,
        "ou": 0,
        "ōu": 1,
        "óu": 2,
        "ǒu": 3,
        "òu": 4,
        "u": 0,
        "ū": 1,
        "ú": 2,
        "ǔ": 3,
        "ù": 4,
        "ua": 0,
        "uā": 1,
        "uá": 2,
        "uǎ": 3,
        "uà": 4,
        "uai": 0,
        "uāi": 1,
        "uái": 2,
        "uǎi": 3,
        "uài": 4,
        "uan": 0,
        "uān": 1,
        "uán": 2,
        "uǎn": 3,
        "uàn": 4,
        "uang": 0,
        "uāng": 1,
        "uáng": 2,
        "uǎng": 3,
        "uàng": 4,
        "ui": 0,
        "uī": 1,
        "uí": 2,
        "uǐ": 3,
        "uì": 4,
        "un": 0,
        "ūn": 1,
        "ún": 2,
        "ǔn": 3,
        "ùn": 4,
        "uo": 0,
        "uō": 1,
        "uó": 2,
        "uǒ": 3,
        "uò": 4,
        "ü": 0,
        "ǖ": 1,
        "ǘ": 2,
        "ǚ": 3,
        "ǜ": 4,
        "üan": 0,
        "ǖan": 1,
        "ǘan": 2,
        "ǚan": 3,
        "ǜan": 4,
        "ue": 0,
        "uē": 1,
        "ué": 2,
        "uě": 3,
        "uè": 4,
        "ün": 0,
        "ǖn": 1,
        "ǘn": 2,
        "ǚn": 3,
        "ǜn": 4,
    }
    CONSONANT_TO_BASE = {
        "a": "a",
        "ā": "a",
        "á": "a",
        "ǎ": "a",
        "à": "a",
        "ai": "ai",
        "āi": "ai",
        "ái": "ai",
        "ǎi": "ai",
        "ài": "ai",
        "an": "an",
        "ān": "an",
        "án": "an",
        "ǎn": "an",
        "àn": "an",
        "ang": "ang",
        "āng": "ang",
        "áng": "ang",
        "ǎng": "ang",
        "àng": "ang",
        "ao": "ao",
        "āo": "ao",
        "áo": "ao",
        "ǎo": "ao",
        "ào": "ao",
        "e": "e",
        "ē": "e",
        "é": "e",
        "ě": "e",
        "è": "e",
        "ei": "ei",
        "ēi": "ei",
        "éi": "ei",
        "ěi": "ei",
        "èi": "ei",
        "en": "en",
        "ēn": "en",
        "én": "en",
        "ěn": "en",
        "èn": "en",
        "eng": "eng",
        "ēng": "eng",
        "éng": "eng",
        "ěng": "eng",
        "èng": "eng",
        "er": "er",
        "ēr": "er",
        "ér": "er",
        "ěr": "er",
        "èr": "er",
        "i": "i",
        "ī": "i",
        "í": "i",
        "ǐ": "i",
        "ì": "i",
        "ia": "ia",
        "iā": "ia",
        "iá": "ia",
        "iǎ": "ia",
        "ià": "ia",
        "ian": "ian",
        "iān": "ian",
        "ián": "ian",
        "iǎn": "ian",
        "iàn": "ian",
        "iang": "iang",
        "iāng": "iang",
        "iáng": "iang",
        "iǎng": "iang",
        "iàng": "iang",
        "iao": "iao",
        "iāo": "iao",
        "iáo": "iao",
        "iǎo": "iao",
        "iào": "iao",
        "ie": "ie",
        "iē": "ie",
        "ié": "ie",
        "iě": "ie",
        "iè": "ie",
        "in": "in",
        "īn": "in",
        "ín": "in",
        "ǐn": "in",
        "ìn": "in",
        "ing": "ing",
        "īng": "ing",
        "íng": "ing",
        "ǐng": "ing",
        "ìng": "ing",
        "iong": "iong",
        "iōng": "iong",
        "ióng": "iong",
        "iǒng": "iong",
        "iòng": "iong",
        "iu": "iu",
        "iū": "iu",
        "iú": "iu",
        "iǔ": "iu",
        "iù": "iu",
        "o": "o",
        "ō": "o",
        "ó": "o",
        "ǒ": "o",
        "ò": "o",
        "ong": "ong",
        "ōng": "ong",
        "óng": "ong",
        "ǒng": "ong",
        "òng": "ong",
        "ou": "ou",
        "ōu": "ou",
        "óu": "ou",
        "ǒu": "ou",
        "òu": "ou",
        "u": "u",
        "ū": "u",
        "ú": "u",
        "ǔ": "u",
        "ù": "u",
        "ua": "ua",
        "uā": "ua",
        "uá": "ua",
        "uǎ": "ua",
        "uà": "ua",
        "uai": "uai",
        "uāi": "uai",
        "uái": "uai",
        "uǎi": "uai",
        "uài": "uai",
        "uan": "uan",
        "uān": "uan",
        "uán": "uan",
        "uǎn": "uan",
        "uàn": "uan",
        "uang": "uang",
        "uāng": "uang",
        "uáng": "uang",
        "uǎng": "uang",
        "uàng": "uang",
        "ui": "ui",
        "uī": "ui",
        "uí": "ui",
        "uǐ": "ui",
        "uì": "ui",
        "un": "un",
        "ūn": "un",
        "ún": "un",
        "ǔn": "un",
        "ùn": "un",
        "uo": "uo",
        "uō": "uo",
        "uó": "uo",
        "uǒ": "uo",
        "uò": "uo",
        "ü": "v",
        "ǖ": "v",
        "ǘ": "v",
        "ǚ": "v",
        "ǜ": "v",
        "üan": "van",
        "ǖan": "van",
        "ǘan": "van",
        "ǚan": "van",
        "ǜan": "van",
        "ue": "ue",
        "uē": "ue",
        "ué": "ue",
        "uě": "ue",
        "uè": "ue",
        "ün": "vn",
        "ǖn": "vn",
        "ǘn": "vn",
        "ǚn": "vn",
        "ǜn": "vn",
    }
    BLADE_ALVEOLARS = ['z', 'c', 's']
    RETROFLEXES = ['zh', 'ch', 'sh']
    ALVEOLAR_NASAL = ['ian', 'uan', 'van', 'an', 'en', 'in', 'un', 'vn']
    VELAR_NASAL = ['iang', 'uang', 'ang', 'eng', 'ing', 'ong', 'iong']

    initialized = False


ltp = LTP(path='base')

# initialize when imported
if not _Pinyin.initialized:
    from gop_server import zh_config

    # import json
    # pinyin_dict_path = os.path.join(zh_config.root_path, 'pinyin-dict.json')
    # pinyin_dict = json.load(open(pinyin_dict_path, 'r'))
    # pickle.dump(pinyin_dict, open(os.path.join(zh_config.root_path, 'pinyin-dict.pkl'), 'wb'))
    pinyin_dict_path = os.path.join(zh_config.root_path, 'pinyin-dict.pkl')
    pinyin_dict: dict = pickle.load(open(pinyin_dict_path, 'rb'))
    load_phrases_dict(pinyin_dict)
    _Pinyin.initialized = True


def to_kaldi_style_pinyin(pinyin: str) -> List[str]:
    if pinyin == _Pinyin.EMPTY_PINYIN:
        return []
    if pinyin[:-1] == 'r':  # 儿化
        pinyin = 'er' + '_' + pinyin[-1]
    else:
        pinyin = pinyin[:-1] + '_' + pinyin[-1]
    ret = []
    len_ = len(pinyin)
    assert len_ > 0
    final_start = 0
    if len_ >= 2 and pinyin[:2] in _Pinyin.INITIALS:
        ret.append(pinyin[:2])
        final_start = 2
    elif pinyin[0] in _Pinyin.INITIALS:
        ret.append(pinyin[0])
        final_start = 1

    ret.append(pinyin[final_start:])
    return ret


def to_human_pinyin(pinyin: str) -> List[str]:
    from gop_server import zh_config
    ret = to_kaldi_style_pinyin(pinyin)
    ret = [zh_config.phones2annotation[e] for e in ret]
    return ret


def transcript_to_pinyin(segments: List[str], style='kaldi') -> List[str]:
    pinyin = sandhi(segments)
    ret = []
    if style == 'kaldi':
        for p in pinyin:
            ret += to_kaldi_style_pinyin(p)
    elif style == 'human':
        for p in pinyin:
            ret += to_human_pinyin(p)
    else:
        ret = pinyin
    return ret


# TODO: 子,头,们,巴,么 轻声


def get_parsetree(segs: List[str]) -> List[int]:
    seg, hidden = ltp.seg(segs)
    _sdp = ltp.sdp(hidden, mode='tree')
    sdp = []
    for s in _sdp:
        sdp += s
    ret = np.asarray([x[0] for x in sdp])
    ret -= 1  # convert to starting from 0
    return ret.tolist()


def find_center_nodes(neighbour_arcs: List[Tuple[int, int]]) -> List[int]:
    neighbour_arcs.sort(key=lambda x: x[1])
    res = []
    last_parent = -100
    for _, parent in neighbour_arcs:
        if last_parent == parent:
            res.append(parent)
        last_parent = parent
    return res


# axis = -1 for left, +1 for right
def aggregate_along(axis: int, center: int, bounds: Set[Tuple[int, int]]) -> List[int]:
    res = []
    idx = center
    while (idx, idx + axis) in bounds:
        idx += axis
        res.append(idx)
    return res


def merge_groups(pinyins: List[List[str]], group_ids: List[int]) -> List[List[str]]:
    res = []
    last_id = group_ids[0]
    buffer = []
    for py, gid in zip(pinyins, group_ids):
        if gid != last_id:
            res.append(buffer)
            buffer = py
        else:
            buffer += py
        last_id = gid

    if buffer:
        res.append(buffer)
    return res


def first_order_merge(segs: List[str], relations: List[int]) -> List[int]:
    # use only neighbour dependencies
    neighbour_arcs = [(child, parent - 1) for child, parent in enumerate(relations)]
    neighbour_arcs = [(a, b) for a, b in neighbour_arcs if abs(a - b) == 1]

    center_nodes = find_center_nodes(neighbour_arcs)
    bounds = set(neighbour_arcs + [(b, a) for a, b in neighbour_arcs])

    group_ids = [-1 for _ in range(len(segs))]

    for center in center_nodes:
        left_ids = aggregate_along(-1, center, bounds)
        right_ids = aggregate_along(1, center, bounds)

        left_mark = min(left_ids)
        for i in left_ids:
            group_ids[i] = left_mark
        right_mark = min(right_ids)
        for i in right_ids:
            group_ids[i] = right_mark

        left_weight = sum([len(segs[i]) for i in left_ids])
        right_weight = sum([len(segs[i]) for i in right_ids])
        center_weight = len(segs[center])

        if abs(left_weight + center_weight - right_weight) < abs(right_weight + center_weight - left_weight):
            # best_suited_to place center on the left
            group_ids[center] = group_ids[center - 1]
        else:
            group_ids[center] = group_ids[center + 1]

    group_ids[0] = 0
    aggregated_length = 0
    for idx in range(1, len(group_ids)):
        if group_ids[idx] < 0:
            if (idx, idx - 1) in bounds and aggregated_length < 4:
                group_ids[idx] = group_ids[idx - 1]
                aggregated_length += len(segs[idx])
            else:
                group_ids[idx] = idx
                aggregated_length = 0

    return group_ids


# def tail_qingsheng(word: Tuple[str, List[str]]):
#     hans, pinyins = word
#
#     if hans[-1] == '个':
#         pinyins[-1] = 'ge0'
#
#     return hans, pinyins


def fix_tone0(pys: List[str]) -> List[str]:
    res = ['{}0'.format(py) if py[-1] not in _Pinyin.TONES and py[-1].isalpha() else py for py in pys]
    return res


def is_tone(py: str, tone: int):
    if py is not None and _Pinyin.PINYIN_PATTERN.match(py) is not None:
        assert py[-1] in _Pinyin.TONES, 'panic : pinyin : {} has no tone!'.format(py)
        return int(py[-1]) == tone
    else:
        return False


def force_tone(py: str, tone: int):
    assert str(tone) in _Pinyin.TONES, 'panic : cannot force tone {} on pinyin {}'.format(tone, py)
    return '{}{}'.format(py[:-1], tone)


def sandhi_shangsheng(pinyins: List[str]) -> List[str]:
    met_shangsheng = False

    for i, py in enumerate(reversed(pinyins)):
        if met_shangsheng:
            if is_tone(py, 3):
                pinyins[-(i + 1)] = force_tone(py, 2)
            else:
                met_shangsheng = False
        if is_tone(py, 3):
            met_shangsheng = True
    return pinyins


def safe_index(xs: Union[List, str], idx: int):
    if idx < 0 or idx >= len(xs):
        return None
    else:
        return xs[idx]


def sandhi_yi(word: Tuple[str, List[str]]) -> Tuple[str, List[str]]:
    hans, pinyins = word

    yi_indices = [i for i, han in enumerate(hans) if han == '一']
    for yi_idx in yi_indices:
        # 表序数的一是一声
        if safe_index(hans, yi_idx - 1) == '第':
            pinyins[yi_idx] = 'yi1'
        # 在数量中的一是一声
        elif safe_index(hans, yi_idx + 1) in _Pinyin.NUMBER_HANS:
            pinyins[yi_idx] = 'yi1'
        # 四声前的一是二声
        elif is_tone(safe_index(pinyins, yi_idx + 1), 4):
            pinyins[yi_idx] = 'yi2'
        # 非四声前的一是四声
        elif safe_index(pinyins, yi_idx + 1) is not None:
            pinyins[yi_idx] = 'yi4'
        # 最普通的默认情况读本音 一
        else:
            pinyins[yi_idx] = 'yi1'
    return hans, pinyins


def sandhi_bu(word: Tuple[str, List[str]], nex: Tuple[str, List[str]]) -> Tuple[str, List[str]]:
    if nex != (None, None):
        hans, pinyins = word[0] + nex[0], word[1] + nex[1]
        # 由于 "不" 相关的分词不可靠，总是把不和后续的一个词一起分析
    else:
        hans, pinyins = word

    bu_indices = [i for i, han in enumerate(hans) if han == '不']
    for bu_idx in bu_indices:
        if is_tone(safe_index(pinyins, bu_idx + 1), 4):
            pinyins[bu_idx] = 'bu2'

    return hans[:len(word[0])], pinyins[:len(word[1])]


def word_sandhi(context: List[Tuple[str, List[str]]]) -> List[str]:
    prev, current, nex = context
    hans, pinyins = current
    if '一' in hans:
        current = sandhi_yi(current)
    if '不' in hans:
        current = sandhi_bu(current, nex)

    return current[1]


def disambig_u_v(pinyin: str) -> str:
    if pinyin[0] in ['j', 'q', 'x', 'y']:
        if 'uan' == pinyin[1:4]:
            return pinyin.replace('uan', 'van')
        elif 'un' == pinyin[1:3]:
            return pinyin.replace('un', 'vn')
        elif 'u' == pinyin[1:2]:
            return pinyin.replace('u', 'v')
    return pinyin


def sandhi(segments: List[str]):
    # noinspection PyTypeChecker
    pinyins = [
        fix_tone0(
            lazy_pinyin(word, errors=lambda x: [_Pinyin.EMPTY_PINYIN for _ in x], style=Style.TONE3)
        ) for word in segments
    ]

    parse_tree = get_parsetree(segments)
    group_ids = first_order_merge(segments, parse_tree)

    extended_pinyin: List[List[str]] = [None] + pinyins + [None]
    extended_segments: List[str] = [None] + segments + [None]
    extended = list(zip(extended_segments, extended_pinyin))

    contexts = [extended[idx: idx + 3] for idx in range(len(pinyins))]
    processed_pinyin = [word_sandhi(context) for context in contexts]

    regrouped = merge_groups(processed_pinyin, group_ids)

    ret = list(itertools.chain(*[sandhi_shangsheng(pinyins) for pinyins in regrouped]))

    # distinguish uan and van even for pinyin like 'yuan'
    n = len(ret)
    for i in range(n):
        ret[i] = disambig_u_v(ret[i])

    return ret


if __name__ == '__main__':
    print(transcript_to_pinyin(['我姓王。', '李小姐，', '你叫什么名字？']))
    print(transcript_to_pinyin(['我姓王。', '李小姐，', '你叫什么名字？'], style='human'))
    print(transcript_to_pinyin(['不好意思', '你是一头猪', '笑死我了'], style='human'))
    print(transcript_to_pinyin(['韵律', '宽面', '卷曲', '寻死', '均衡', '滚蛋'], style='human'))
    print(transcript_to_pinyin(['你', '也', '去吧'], style='human'))
