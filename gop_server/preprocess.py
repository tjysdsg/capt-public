import webrtcvad
import numpy as np
import collections
from typing import List


class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, data: np.ndarray, timestamp: float, duration: float):
        self.data = data
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration: int, audio: np.ndarray, sample_rate: int = 16000):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration / 1000.0))
    offset = 0
    timestamp = 0.0
    duration = n / sample_rate
    length = audio.shape[0]
    while offset + n < length:
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def pcm2float(sig: np.ndarray, dtype='float32') -> np.ndarray:
    """
    Convert PCM signal to floating point with a range from -1 to 1.

    Use dtype='float32' for single precision.

    :param sig: array_like
        Input array, must have integral type.
    :param dtype: data type, optional
        Desired (floating point) data type.
    :returns numpy.ndarray
        Normalized floating point data.
    """
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max


def float2pcm(sig: np.ndarray, dtype='int16') -> np.ndarray:
    """
    Convert floating point signal with a range from -1 to 1 to PCM.

    Any signal values outside the interval [-1.0, 1.0) are clipped.
    No dithering is used.

    Note that there are different possibilities for scaling floating
    point numbers to PCM numbers, this function implements just one of
    them. For an overview of alternatives see
    http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html

    :param sig: array_like
        Input array, must have floating point type.
    :param dtype : data type, optional
        Desired (integer) data type.
    :returns numpy.ndarray
        Integer data, scaled and clipped to the range of the given *dtype*.
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


def vad_collector(frame_duration: int, window_size: int, vad: webrtcvad.Vad, frames: List[Frame],
                  trigger_thres: float, untrigger_thres: float, sample_rate=16000) -> np.ndarray:
    """
    Filters out non-voiced audio frames.

    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.

    Uses a padded, sliding window algorithm over the audio frames.
    When more than `trigger_tres` of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until `untrigger_tres` of the frames in
    the window are unvoiced to untrigger.

    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.

    :param frame_duration: The frame duration in milliseconds.
    :param window_size: The size of the sliding window in milliseconds.
    :param vad: An instance of `webrtcvad.Vad`.
    :param frames: a source of audio frames (sequence or generator).
    :param untrigger_thres: Threshold for a window to become unvoiced
    :param trigger_thres: Threshold for a window to become voiced
    :param sample_rate: The audio sample rate, in Hz.

    :return: Resulted PCM audio data, np.ndarray
    """
    n_frames_per_window = int(window_size / frame_duration)
    # use a deque for our sliding window/ring buffer.
    window = collections.deque(maxlen=n_frames_per_window)
    triggered = False

    voiced_frames: List[np.ndarray] = []
    for frame in frames:
        is_speech = vad.is_speech(frame.data.tobytes(), sample_rate)
        window.append((frame.data, is_speech))

        if not triggered:
            # find how many voiced frames there are, and the index of the first voiced frame
            voiced = []
            voice_start_i = -1
            for i, w in enumerate(window):
                f = w[0]
                speech = w[1]
                if speech:
                    voiced.append(f)
                    if voice_start_i == -1:
                        voice_start_i = i
            if len(voiced) > trigger_thres * window.maxlen:
                triggered = True

                # starting from the first voiced frame, append the frames to voiced_frames
                for i in range(voice_start_i, len(window)):
                    voiced_frames.append(window[i][0].data)
                window.clear()
        else:
            n_unvoiced = len([f for f, speech in window if not speech])
            if n_unvoiced > untrigger_thres * window.maxlen:
                triggered = False
                window.clear()
            else:
                voiced_frames.append(frame.data)

    ret = np.asarray(voiced_frames).ravel()
    return ret


def remove_long_silence(wav: np.ndarray, frame_duration=10, window_size=50, trigger_thres=0.7, untrigger_thres=0.32,
                        vad_aggressiveness=2) -> np.ndarray:
    vad = webrtcvad.Vad(vad_aggressiveness)
    frames = frame_generator(frame_duration, wav)
    frames = list(frames)
    ret = vad_collector(frame_duration, window_size, vad, frames, trigger_thres=trigger_thres,
                        untrigger_thres=untrigger_thres)
    return ret
