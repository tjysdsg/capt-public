import os
from os.path import dirname, abspath
import librosa
from gop_server.preprocess import remove_long_silence, float2pcm
import wave

file_dir = dirname(abspath(__file__))
output_path = os.path.join(file_dir, 'tmp.wav')


def test_vad():
    wav_path = os.path.join(file_dir, 'audio', 'Jiaxin-加州冬天不冷-de1e9a6a-cd9d-4be8-a077-ac18990c4749.wav')
    wav_data, sr = librosa.load(wav_path, sr=16000, mono=True)
    assert sr == 16000
    ret = remove_long_silence(wav_data)
    wf = wave.open(output_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(float2pcm(ret).tobytes())
