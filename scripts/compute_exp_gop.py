import os
import numpy as np
from typing import List
import librosa
from gop_server.preprocess import float2pcm
from argparse import ArgumentParser


def compute_from_zh_wav(path: str, transcript: str = None, pinyins: List[str] = None):
    assert transcript is not None or pinyins is not None

    from gop_server import lang_configs

    # load data
    data, sr = librosa.load(path, sr=16000, mono=True)
    data = float2pcm(data)

    # main routine
    lang_config = lang_configs.get('zh')
    res = lang_config.handler(data, transcript, pinyins)

    gop = res['gop'][:, 0]  # not including tone gop
    gop = np.clip(np.exp(gop), 0, 1).tolist()  # clip to [0, 1]

    tone_corr = res['corr'][:, 1].tolist()
    tone_score = [1 if tc else 0 for tc in tone_corr]

    return gop, tone_score


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--text2phone', type=str)
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--out-path', type=str)
    return parser.parse_args()


def main():
    args = get_args()

    text2phones = {}
    with open(args.text2phone) as inf:
        for line in inf:
            tokens = line.strip('\n').split()
            text2phones[tokens[0]] = tokens[1:]
    print(text2phones)

    utt2path = {}
    utt2phones = {}
    utt2spk = {}
    utt2text = {}
    for d in os.scandir(args.data_dir):  # type: os.DirEntry
        if not d.is_dir():
            continue

        spk = d.name
        for f in os.scandir(d.path):  # type: os.DirEntry
            if not f.is_file():
                continue

            utt = f.name
            tokens = utt.split('-')
            text = tokens[-1].split('.')[0]

            utt2path[utt] = f.path
            utt2spk[utt] = spk
            utt2phones[utt] = text2phones[text]
            utt2text[utt] = text

    of = open(args.out_path, 'w')
    # call gop
    utts = list(utt2path.keys())
    for utt in utts:
        path = utt2path[utt]
        phones = utt2phones[utt]
        text = utt2text[utt]

        print(f'=============== Computing GOP of {path} ===============')
        try:
            gop, tone_score = compute_from_zh_wav(path, transcript=text, pinyins=phones)
        except Exception:
            print(f'=============== {path} FAILED ===============')
            of.write(f'{utt} FAILED\n')
            continue

        gop_str = ' '.join([str(e) for e in gop])
        tone_str = ' '.join([str(e) for e in tone_score])
        of.write(f'{utt},{gop_str},{tone_str}\n')


if __name__ == '__main__':
    main()
