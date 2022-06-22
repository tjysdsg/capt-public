import jieba
import os
from typing import List, Tuple
import string
import re


class _Tokenizer:
    initialized = False

    # https://github.com/speechio/chinese_text_normalization/blob/master/TN/cn_tn.py
    CHINESE_DIGIS = '零一二三四五六七八九'
    BIG_CHINESE_DIGIS_SIMPLIFIED = '零壹贰叁肆伍陆柒捌玖'
    BIG_CHINESE_DIGIS_TRADITIONAL = '零壹貳參肆伍陸柒捌玖'
    SMALLER_BIG_CHINESE_UNITS_SIMPLIFIED = '十百千万'
    SMALLER_BIG_CHINESE_UNITS_TRADITIONAL = '拾佰仟萬'
    LARGER_CHINESE_NUMERING_UNITS_SIMPLIFIED = '亿兆京垓秭穰沟涧正载'
    LARGER_CHINESE_NUMERING_UNITS_TRADITIONAL = '億兆京垓秭穰溝澗正載'
    SMALLER_CHINESE_NUMERING_UNITS_SIMPLIFIED = '十百千万'
    SMALLER_CHINESE_NUMERING_UNITS_TRADITIONAL = '拾佰仟萬'

    ZERO_ALT = '〇'
    ONE_ALT = '幺'
    TWO_ALTS = ['两', '兩']

    POSITIVE = ['正', '正']
    NEGATIVE = ['负', '負']
    POINT = ['点', '點']
    # PLUS = ['加', '加']
    # SIL = ['杠', '槓']

    # 中文数字系统类型
    NUMBERING_TYPES = ['low', 'mid', 'high']

    CURRENCY_NAMES = '(人民币|美元|日元|英镑|欧元|马克|法郎|加拿大元|澳元|港币|先令|芬兰马克|爱尔兰镑|' \
                     '里拉|荷兰盾|埃斯库多|比塞塔|印尼盾|林吉特|新西兰元|比索|卢布|新加坡元|韩元|泰铢)'
    CURRENCY_UNITS = '((亿|千万|百万|万|千|百)|(亿|千万|百万|万|千|百|)元|(亿|千万|百万|万|千|百|)块|角|毛|分)'
    COM_QUANTIFIERS = '(匹|张|座|回|场|尾|条|个|首|阙|阵|网|炮|顶|丘|棵|只|支|袭|辆|挑|担|颗|壳|窠|曲|墙|群|腔|' \
                      '砣|座|客|贯|扎|捆|刀|令|打|手|罗|坡|山|岭|江|溪|钟|队|单|双|对|出|口|头|脚|板|跳|枝|件|贴|' \
                      '针|线|管|名|位|身|堂|课|本|页|家|户|层|丝|毫|厘|分|钱|两|斤|担|铢|石|钧|锱|忽|(千|毫|微)克|' \
                      '毫|厘|分|寸|尺|丈|里|寻|常|铺|程|(千|分|厘|毫|微)米|撮|勺|合|升|斗|石|盘|碗|碟|叠|桶|笼|盆|' \
                      '盒|杯|钟|斛|锅|簋|篮|盘|桶|罐|瓶|壶|卮|盏|箩|箱|煲|啖|袋|钵|年|月|日|季|刻|时|周|天|秒|分|旬|' \
                      '纪|岁|世|更|夜|春|夏|秋|冬|代|伏|辈|丸|泡|粒|颗|幢|堆|条|根|支|道|面|片|张|颗|块)'

    # punctuation information are based on Zhon project (https://github.com/tsroten/zhon.git)
    CHINESE_PUNC_STOP = '！？｡。'
    CHINESE_PUNC_NON_STOP = '＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞' \
                            '〟〰〾〿–—‘’‛“”„‟…‧﹏'
    CHINESE_PUNC_OTHER = '·〈〉-'
    CHINESE_PUNC_LIST = CHINESE_PUNC_STOP + CHINESE_PUNC_NON_STOP + CHINESE_PUNC_OTHER


# init when imported
if not _Tokenizer.initialized:
    from gop_server import zh_config

    jieba.enable_paddle()
    dict_path = os.path.join(zh_config.root_path, 'dict.txt')
    jieba.set_dictionary(dict_path)
    jieba.initialize()


def clean_transcript(trans: str, remove_punct=True) -> Tuple[str, List[str]]:
    """
    Clean Chinese transcript
    :param trans: Transcript
    :param remove_punct: Remove all punctuation if true
    :return: (trans, segments)
        - Cleaned transcript
        - Tokenized segments, list of str, punctuation is preserved whatever the value of `remove_punct`
    """
    trans = _normalize_transcript(trans)
    segments = _tokenize_chinese(trans)
    trans = ' '.join(segments)
    if remove_punct:
        trans = _remove_punctuation(trans)
    return trans, segments


def _tokenize_chinese(s: str) -> List[str]:
    seg_list = jieba.cut(s, HMM=False)
    return list(seg_list)


def _remove_punctuation(s: str) -> str:
    old_chars = _Tokenizer.CHINESE_PUNC_LIST + string.punctuation  # includes all CN and EN punctuations
    new_chars = ' ' * len(old_chars)
    del_chars = ''
    return s.translate(str.maketrans(old_chars, new_chars, del_chars))


# ---------------- text normalization ----------------#


class BaseSymbol:
    def __init__(self):
        pass

    def normalized(self, *args, **kwargs) -> str:
        raise NotImplementedError()


class ChineseChar(object):
    """
    中文字符
    每个字符对应简体和繁体,
    e.g. 简体 = '负', 繁体 = '負'
    转换时可转换为简体或繁体
    """

    def __init__(self, simplified, traditional):
        self.simplified = simplified
        self.traditional = traditional
        # self.__repr__ = self.__str__

    def __str__(self):
        return self.simplified or self.traditional or None

    def __repr__(self):
        return self.__str__()


class ChineseNumberUnit(ChineseChar):
    """
    中文数字/数位字符
    每个字符除繁简体外还有一个额外的大写字符
    e.g. '陆' 和 '陸'
    """

    def __init__(self, power, simplified, traditional, big_s, big_t):
        super().__init__(simplified, traditional)
        self.power = power
        self.big_s = big_s
        self.big_t = big_t

    def __str__(self):
        return '10^{}'.format(self.power)

    @classmethod
    def create(cls, index, value, numbering_type=_Tokenizer.NUMBERING_TYPES[1], small_unit=False):
        if small_unit:
            return ChineseNumberUnit(power=index + 1, simplified=value[0], traditional=value[1], big_s=value[1],
                                     big_t=value[1])
        elif numbering_type == _Tokenizer.NUMBERING_TYPES[0]:
            return ChineseNumberUnit(power=index + 8, simplified=value[0], traditional=value[1], big_s=value[0],
                                     big_t=value[1])
        elif numbering_type == _Tokenizer.NUMBERING_TYPES[1]:
            return ChineseNumberUnit(power=(index + 2) * 4, simplified=value[0], traditional=value[1], big_s=value[0],
                                     big_t=value[1])
        elif numbering_type == _Tokenizer.NUMBERING_TYPES[2]:
            return ChineseNumberUnit(power=pow(2, index + 3), simplified=value[0], traditional=value[1], big_s=value[0],
                                     big_t=value[1])
        raise ValueError(f'Counting type should be in {_Tokenizer.NUMBERING_TYPES} ({numbering_type} provided).')


class ChineseNumberDigit(ChineseChar):
    """
    中文数字字符
    """

    def __init__(self, value, simplified, traditional, big_s, big_t, alt_s=None, alt_t=None):
        super().__init__(simplified, traditional)
        self.value = value
        self.big_s = big_s
        self.big_t = big_t
        self.alt_s = alt_s
        self.alt_t = alt_t

    def __str__(self):
        return str(self.value)

    @classmethod
    def create(cls, i, v):
        return ChineseNumberDigit(i, v[0], v[1], v[2], v[3])


class ChineseMath(ChineseChar):
    """
    中文数位字符
    """

    def __init__(self, simplified, traditional, symbol, expression=None):
        super().__init__(simplified, traditional)
        self.symbol = symbol
        self.expression = expression
        self.big_s = simplified
        self.big_t = traditional


CC, CNU, CND, CM = ChineseChar, ChineseNumberUnit, ChineseNumberDigit, ChineseMath


class NumberSystem:
    """
    中文数字系统
    """

    def __init__(self):
        self.units = None
        self.digits = None
        self.math = None


class MathSymbol:
    """
    用于中文数字系统的数学符号 (繁/简体), e.g.
    positive = ['正', '正']
    negative = ['负', '負']
    point = ['点', '點']
    """

    def __init__(self, positive, negative, point):
        self.positive = positive
        self.negative = negative
        self.point = point

    def __iter__(self):
        for v in self.__dict__.values():
            yield v


def create_system(numbering_type) -> NumberSystem:
    """
    根据数字系统类型返回创建相应的数字系统
    NUMBERING_TYPES = ['low', 'mid', 'high']: 中文数字系统类型
        low:  '兆' = '亿' * '十' = $10^{9}$,  '京' = '兆' * '十', etc.
        mid:  '兆' = '亿' * '万' = $10^{12}$, '京' = '兆' * '万', etc.
        high: '兆' = '亿' * '亿' = $10^{16}$, '京' = '兆' * '兆', etc.
    返回对应的数字系统
    """

    # chinese number units of '亿' and larger
    all_larger_units = zip(_Tokenizer.LARGER_CHINESE_NUMERING_UNITS_SIMPLIFIED,
                           _Tokenizer.LARGER_CHINESE_NUMERING_UNITS_TRADITIONAL)
    larger_units = [CNU.create(i, v, numbering_type, False) for i, v in enumerate(all_larger_units)]
    # chinese number units of '十, 百, 千, 万'
    all_smaller_units = zip(_Tokenizer.SMALLER_CHINESE_NUMERING_UNITS_SIMPLIFIED,
                            _Tokenizer.SMALLER_CHINESE_NUMERING_UNITS_TRADITIONAL)
    smaller_units = [CNU.create(i, v, small_unit=True) for i, v in enumerate(all_smaller_units)]
    # digits
    chinese_digits = zip(_Tokenizer.CHINESE_DIGIS, _Tokenizer.CHINESE_DIGIS, _Tokenizer.BIG_CHINESE_DIGIS_SIMPLIFIED,
                         _Tokenizer.BIG_CHINESE_DIGIS_TRADITIONAL)
    digits = [CND.create(i, v) for i, v in enumerate(chinese_digits)]
    digits[0].alt_s, digits[0].alt_t = _Tokenizer.ZERO_ALT, _Tokenizer.ZERO_ALT
    digits[1].alt_s, digits[1].alt_t = _Tokenizer.ONE_ALT, _Tokenizer.ONE_ALT
    digits[2].alt_s, digits[2].alt_t = _Tokenizer.TWO_ALTS[0], _Tokenizer.TWO_ALTS[1]

    # symbols
    positive_cn = CM(_Tokenizer.POSITIVE[0], _Tokenizer.POSITIVE[1], '+', lambda x: x)
    negative_cn = CM(_Tokenizer.NEGATIVE[0], _Tokenizer.NEGATIVE[1], '-', lambda x: -x)
    point_cn = CM(_Tokenizer.POINT[0], _Tokenizer.POINT[1], '.', lambda x, y: float(str(x) + '.' + str(y)))
    # sil_cn = CM(SIL[0], SIL[1], '-', lambda x, y: float(str(x) + '-' + str(y)))
    system = NumberSystem()
    system.units = smaller_units + larger_units
    system.digits = digits
    system.math = MathSymbol(positive_cn, negative_cn, point_cn)
    return system


def num2chn(number_string: str, numbering_type=_Tokenizer.NUMBERING_TYPES[1], big=False, traditional=False,
            alt_zero=False, alt_one=False, alt_two=True, use_zeros=True, use_units=True) -> str:
    system = create_system(numbering_type)

    def get_value(value_string: str):
        striped_string = value_string.lstrip('0')

        # return nothing if all zeros
        if not striped_string:
            return []
        # one digit number (not including the leading 0)
        elif len(striped_string) == 1:
            if use_zeros and len(value_string) != len(striped_string):
                return [system.digits[0], system.digits[int(striped_string)]]
            return [system.digits[int(striped_string)]]
        else:  # multiple digits
            result_unit = next(u for u in reversed(system.units) if u.power < len(striped_string))
            result_string = value_string[:-result_unit.power]
            return get_value(result_string) + [result_unit] + get_value(striped_string[-result_unit.power:])

    int_dec = number_string.split('.')
    if len(int_dec) == 1:
        int_string = int_dec[0]
        dec_string = ""
    elif len(int_dec) == 2:
        int_string = int_dec[0]
        dec_string = int_dec[1]
    else:
        # assuming decimal separator is '.'
        raise ValueError("Invalid input num string with more than one dot: {}".format(number_string))

    if use_units and len(int_string) > 1:
        result_symbols = get_value(int_string)
    else:
        result_symbols = [system.digits[int(c)] for c in int_string]
    dec_symbols = [system.digits[int(c)] for c in dec_string]
    if dec_string:
        result_symbols += [system.math.point] + dec_symbols

    if alt_two:
        liang = CND(2, system.digits[2].alt_s, system.digits[2].alt_t, system.digits[2].big_s, system.digits[2].big_t)
        for i, v in enumerate(result_symbols):
            if isinstance(v, CND) and v.value == 2:
                next_symbol = result_symbols[i + 1] if i < len(result_symbols) - 1 else None
                previous_symbol = result_symbols[i - 1] if i > 0 else None
                if isinstance(next_symbol, CNU) and isinstance(previous_symbol, (CNU, type(None))):
                    if next_symbol.power != 1 and ((previous_symbol is None) or (previous_symbol.power != 1)):
                        result_symbols[i] = liang

    # if big is True, '两' will not be used and `alt_two` has no impact on output
    if big:
        attr_name = 'big_'
        if traditional:
            attr_name += 't'
        else:
            attr_name += 's'
    else:
        if traditional:
            attr_name = 'traditional'
        else:
            attr_name = 'simplified'

    result = ''.join([getattr(s, attr_name) for s in result_symbols])

    if not use_zeros:
        result = result.strip(getattr(system.digits[0], attr_name))

    if alt_zero:
        result = result.replace(getattr(system.digits[0], attr_name), system.digits[0].alt_s)

    if alt_one:
        result = result.replace(getattr(system.digits[1], attr_name), system.digits[1].alt_s)

    for i, p in enumerate(_Tokenizer.POINT):
        if result.startswith(p):
            return _Tokenizer.CHINESE_DIGIS[0] + result

    # ^10, 11, .., 19
    if len(result) >= 2 and \
            result[1] in [_Tokenizer.SMALLER_CHINESE_NUMERING_UNITS_SIMPLIFIED[0],
                          _Tokenizer.SMALLER_CHINESE_NUMERING_UNITS_TRADITIONAL[0]] \
            and result[0] in [_Tokenizer.CHINESE_DIGIS[1], _Tokenizer.BIG_CHINESE_DIGIS_SIMPLIFIED[1],
                              _Tokenizer.BIG_CHINESE_DIGIS_TRADITIONAL[1]]:
        result = result[1:]

    return result


class Cardinal(BaseSymbol):
    def __init__(self, cardinal=None, chntext=None):
        super().__init__()
        self.cardinal = cardinal
        self.chntext = chntext

    def normalized(self) -> str:
        return num2chn(self.cardinal)


class Digit:
    def __init__(self, digit=None, chntext=None):
        self.digit = digit
        self.chntext = chntext

    def digit2chntext(self):
        return num2chn(self.digit, alt_two=False, use_units=False)


class Telephone(BaseSymbol):
    def __init__(self, telephone=None, raw_chntext=None, chntext=None):
        super().__init__()
        self.telephone = telephone
        self.raw_chntext = raw_chntext
        self.chntext = chntext

    def normalized(self, fixed=False) -> str:
        if fixed:
            sil_parts = self.telephone.split('-')
            self.raw_chntext = '<SIL>'.join([num2chn(part, alt_two=False, use_units=False) for part in sil_parts])
            self.chntext = self.raw_chntext.replace('<SIL>', '')
        else:
            sp_parts = self.telephone.strip('+').split()
            self.raw_chntext = '<SP>'.join([num2chn(part, alt_two=False, use_units=False) for part in sp_parts])
            self.chntext = self.raw_chntext.replace('<SP>', '')
        return self.chntext


class Fraction(BaseSymbol):
    def __init__(self, fraction=None, chntext=None):
        super().__init__()
        self.fraction = fraction
        self.chntext = chntext

    def normalized(self) -> str:
        numerator, denominator = self.fraction.split('/')
        return num2chn(denominator) + '分之' + num2chn(numerator)


class Date(BaseSymbol):
    def __init__(self, date=None, chntext=None):
        super().__init__()
        self.date = date
        self.chntext = chntext

    def normalized(self) -> str:
        date = self.date
        try:
            year, other = date.strip().split('年', 1)
            year = Digit(digit=year).digit2chntext() + '年'
        except ValueError:
            other = date
            year = ''
        if other:
            try:
                month, day = other.strip().split('月', 1)
                month = Cardinal(cardinal=month).normalized() + '月'
            except ValueError:
                day = date
                month = ''
            if day:
                day = Cardinal(cardinal=day[:-1]).normalized() + day[-1]
        else:
            month = ''
            day = ''
        chntext = year + month + day
        self.chntext = chntext
        return self.chntext


class Money(BaseSymbol):
    def __init__(self, money=None, chntext=None):
        super().__init__()
        self.money = money
        self.chntext = chntext

    def normalized(self) -> str:
        money = self.money
        pattern = re.compile(r'(\d+(\.\d+)?)')
        matchers = pattern.findall(money)
        if matchers:
            for matcher in matchers:
                money = money.replace(matcher[0], Cardinal(cardinal=matcher[0]).normalized())
        self.chntext = money
        return self.chntext


class Percentage(BaseSymbol):
    def __init__(self, percentage=None, chntext=None):
        super().__init__()
        self.percentage = percentage
        self.chntext = chntext

    def normalized(self) -> str:
        return '百分之' + num2chn(self.percentage.strip().strip('%'))


class NSWNormalizer:
    def __init__(self, raw_text):
        self.raw_text = '^' + raw_text + '$'
        self.norm_text = ''

    def _particular(self):
        text = self.norm_text
        pattern = re.compile(r"(([a-zA-Z]+)二([a-zA-Z]+))")
        matchers = pattern.findall(text)
        if matchers:
            for matcher in matchers:
                text = text.replace(matcher[0], matcher[1] + '2' + matcher[2], 1)
        self.norm_text = text
        return self.norm_text

    def normalize(self):
        text = self.raw_text

        # 规范化日期
        pattern = re.compile(r"\D+((([089]\d|(19|20)\d{2})年)?(\d{1,2}月(\d{1,2}[日号])?)?)")
        matchers = pattern.findall(text)
        if matchers:
            for matcher in matchers:
                text = text.replace(matcher[0], Date(date=matcher[0]).normalized(), 1)

        # 规范化金钱
        pattern = re.compile(r"\D+((\d+(\.\d+)?)[多余几]?" +
                             _Tokenizer.CURRENCY_UNITS + r"(\d" + _Tokenizer.CURRENCY_UNITS + r"?)?)")
        matchers = pattern.findall(text)
        if matchers:
            # print('money')
            for matcher in matchers:
                text = text.replace(matcher[0], Money(money=matcher[0]).normalized(), 1)

        # 规范化固话/手机号码
        # 手机
        # http://www.jihaoba.com/news/show/13680
        # 移动：139、138、137、136、135、134、159、158、157、150、151、152、188、187、182、183、184、178、198
        # 联通：130、131、132、156、155、186、185、176
        # 电信：133、153、189、180、181、177
        pattern = re.compile(r"\D((\+?86 ?)?1([38]\d|5[0-35-9]|7[678]|9[89])\d{8})\D")
        matchers = pattern.findall(text)
        if matchers:
            for matcher in matchers:
                text = text.replace(matcher[0], Telephone(telephone=matcher[0]).normalized(), 1)

        # 固话
        pattern = re.compile(r"\D((0(10|2[1-3]|[3-9]\d{2})-?)?[1-9]\d{6,7})\D")
        matchers = pattern.findall(text)
        if matchers:
            for matcher in matchers:
                text = text.replace(matcher[0], Telephone(telephone=matcher[0]).normalized(fixed=True), 1)

        # 规范化分数
        pattern = re.compile(r"(\d+/\d+)")
        matchers = pattern.findall(text)
        if matchers:
            for matcher in matchers:
                text = text.replace(matcher, Fraction(fraction=matcher).normalized(), 1)

        # 规范化百分数
        text = text.replace('％', '%')
        pattern = re.compile(r"(\d+(\.\d+)?%)")
        matchers = pattern.findall(text)
        if matchers:
            for matcher in matchers:
                text = text.replace(matcher[0], Percentage(percentage=matcher[0]).normalized(), 1)

        # 规范化纯数+量词
        pattern = re.compile(r"(\d+(\.\d+)?)[多余几]?" + _Tokenizer.COM_QUANTIFIERS)
        matchers = pattern.findall(text)
        if matchers:
            for matcher in matchers:
                text = text.replace(matcher[0], Cardinal(cardinal=matcher[0]).normalized(), 1)

        # 规范化数字编号
        pattern = re.compile(r"(\d{4,32})")
        matchers = pattern.findall(text)
        if matchers:
            for matcher in matchers:
                text = text.replace(matcher, Digit(digit=matcher).digit2chntext(), 1)

        # 规范化纯数
        pattern = re.compile(r"(\d+(\.\d+)?)")
        matchers = pattern.findall(text)
        if matchers:
            for matcher in matchers:
                text = text.replace(matcher[0], Cardinal(cardinal=matcher[0]).normalized(), 1)

        self.norm_text = text
        self._particular()
        return self.norm_text.lstrip('^').rstrip('$')


def nsw_test_case(raw_text):
    print('I:' + raw_text)
    print('O:' + clean_transcript(raw_text)[0])
    print('')


def nsw_test():
    nsw_test_case('固话：0595-23865596或23880880。')
    nsw_test_case('固话：0595-23865596或23880880。')
    nsw_test_case('手机：+86 19859213959或15659451527。')
    nsw_test_case('分数：32477/76391。')
    nsw_test_case('百分数：80.03%。')
    nsw_test_case('编号：31520181154418。')
    nsw_test_case('纯数：2983.07克或12345.60米。')
    nsw_test_case('日期：1999年2月20日或09年3月15号。')
    nsw_test_case('金钱：12块5，34.5元，20.1万')
    nsw_test_case('特殊：O2O或B2C。')
    nsw_test_case('3456万吨')
    nsw_test_case('2938个')
    nsw_test_case('938')
    nsw_test_case('今天吃了115个小笼包231个馒头')
    nsw_test_case('有62％的概率')


def _normalize_transcript(text: str) -> str:
    text = text.upper()
    # Non-Standard Word normalization
    text = NSWNormalizer(text).normalize()
    return text


if __name__ == '__main__':
    nsw_test()
