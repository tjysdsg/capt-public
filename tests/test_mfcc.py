import os
from os.path import dirname, abspath
from gop_server.mfcc import make_mfcc
import librosa
import kaldi_io
import numpy as np

file_dir = dirname(abspath(__file__))


# set dither to 0 to make tests deterministic

def compute_mfcc_from_file(path: str):
    from gop_server.preprocess import float2pcm
    wav = librosa.load(path, sr=16000, mono=True)[0]
    wav = float2pcm(wav)
    wav = wav.astype(np.float)
    features = make_mfcc(wav, False, dither=0, compress=True)
    return features


def test_mfcc_zh():
    wav_path = os.path.join(file_dir, 'audio', 'sha-bi.wav')
    test_scp = os.path.join(file_dir, 'test_mfcc_zh.scp')
    expected = [m for _, m in kaldi_io.read_mat_scp(test_scp)][0]
    features = compute_mfcc_from_file(wav_path)
    assert np.max(np.abs(features - expected)) < 0.02


def test_mfcc_eng():
    wav_path = os.path.join(file_dir, 'audio', 'fuck-you.wav')
    test_scp = os.path.join(file_dir, 'test_mfcc_eng.scp')
    expected = [m for _, m in kaldi_io.read_mat_scp(test_scp)][0]
    features = compute_mfcc_from_file(wav_path)
    assert np.max(np.abs(features - expected)) < 0.02
