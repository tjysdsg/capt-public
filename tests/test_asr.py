import os
import wave


def test_asr():
    from gop_server import project_root_dir
    from gop_server.api.asr import asr
    wav_path = os.path.join(project_root_dir, 'tests', 'audio', '她在学校_denoise.wav')
    wav: wave.Wave_read = wave.open(wav_path, 'rb')
    wav_bytes = wav.readframes(wav.getnframes())
    print(asr(wav_bytes))
