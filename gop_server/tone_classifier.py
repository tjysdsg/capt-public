import numpy as np
import io
import librosa
import librosa.display
import matplotlib.pyplot as plt
from typing import List

model_configs = dict(width=225, height=225, channels=1, activation='relu')

consonant2tone = {
    3: 0,
    4: 1,
    5: 2,
    6: 3,
    7: 4,

    8: 0,
    9: 1,
    10: 2,
    11: 3,
    12: 4,

    13: 0,
    14: 1,
    15: 2,
    16: 3,
    17: 4,

    18: 0,
    19: 1,
    20: 2,
    21: 3,
    22: 4,

    23: 0,
    24: 1,
    25: 2,
    26: 3,
    27: 4,

    32: 3,
    33: 4,
    34: 0,
    35: 1,
    36: 2,

    37: 0,
    38: 1,
    39: 2,
    40: 3,
    41: 4,

    42: 0,
    43: 1,
    44: 2,
    45: 3,
    46: 4,

    47: 0,
    48: 1,
    49: 2,
    50: 3,
    51: 4,

    52: 0,
    53: 1,
    54: 2,
    55: 3,
    56: 4,

    60: 0,
    61: 1,
    62: 2,
    63: 3,
    64: 4,

    65: 0,
    66: 1,
    67: 2,
    68: 3,
    69: 4,

    70: 0,
    71: 1,
    72: 2,
    73: 3,
    74: 4,

    75: 0,
    76: 1,
    77: 2,
    78: 3,
    79: 4,

    80: 0,
    81: 1,
    82: 2,
    83: 3,
    84: 4,

    85: 0,
    86: 1,
    87: 2,
    88: 3,
    89: 4,

    90: 0,
    91: 1,
    92: 2,
    93: 3,
    94: 4,

    95: 0,
    96: 1,
    97: 2,
    98: 3,
    99: 4,

    100: 0,
    101: 1,
    102: 2,
    103: 3,
    104: 4,

    105: 0,
    106: 1,
    107: 2,
    108: 3,
    109: 4,

    115: 0,
    116: 1,
    117: 2,
    118: 3,
    119: 4,

    120: 4,
    121: 0,
    122: 1,
    123: 2,
    124: 3,

    125: 3,
    126: 4,
    127: 0,
    128: 1,
    129: 2,

    137: 0,
    138: 1,
    139: 2,
    140: 3,
    141: 4,

    142: 0,
    143: 1,
    144: 2,
    145: 3,
    146: 4,

    147: 0,
    148: 1,
    149: 2,
    150: 3,
    151: 4,

    152: 0,
    153: 1,
    154: 2,
    155: 3,
    156: 4,

    157: 0,
    158: 1,
    159: 2,
    160: 3,
    161: 4,

    162: 0,
    163: 1,
    164: 2,
    165: 3,
    166: 4,

    167: 0,
    168: 1,
    169: 2,
    170: 3,
    171: 4,

    172: 0,
    173: 1,
    174: 2,
    175: 3,
    176: 4,

    177: 0,
    178: 1,
    179: 2,
    180: 3,
    181: 4,

    182: 0,
    183: 1,
    184: 2,
    185: 3,
    186: 4,

    187: 0,
    188: 1,
    189: 2,
    190: 3,
    191: 4,

    192: 0,
    193: 1,
    194: 2,
    195: 3,
    196: 4,
}


def chop(y: np.ndarray, start: float, end: float, max_dur_len=0.8):
    from gop_server import logger
    dur = end - start

    extra_sec = (max_dur_len - dur) / 2
    s, dur_len, extra_len, L = librosa.time_to_samples([start, dur, extra_sec, max_dur_len], sr=16000)
    ret = np.zeros(L, dtype='float32')
    fade_len = librosa.time_to_samples(0.025, sr=16000)

    if dur < max_dur_len:
        fade_len = min(extra_len, fade_len)
        s = max(0, s - fade_len)
        e = s + dur_len + fade_len
        y = y[s:e]
    else:
        logger.warning(f"Audio length {dur} larger than {max_dur_len}, results might be undesirable")

    mask = np.zeros_like(y)
    mask[:fade_len] = np.linspace(0, 1, num=fade_len)
    mask[-fade_len:] = np.linspace(1, 0, num=fade_len)
    mask[fade_len:-fade_len] = 1
    y *= mask

    if dur < max_dur_len:
        offset = (L - dur_len) // 2 - fade_len
        ret[offset:offset + y.size] = y
        return ret
    else:
        return y


def melspectrogram_feature(y: np.ndarray, start: float, end: float, sr=16000, fmin=50, fmax=350):
    ret = io.BytesIO()

    y = chop(y, start, end)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=2048, hop_length=16, fmin=fmin, fmax=fmax)
    plt.figure(figsize=(2.25, 2.25))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, fmin=fmin, fmax=fmax, cmap='gray')

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(ret, format='jpg')
    plt.close('all')

    ret.seek(0)
    return ret


def load_image_from_bytes(buf):
    import tensorflow as tf
    image = tf.image.decode_jpeg(buf, channels=1)

    # convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # images are already 225x225
    # image = tf.image.resize_images(image, [225, 225])

    return image


def create_model(width=225, height=225, channels=1, activation='relu'):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Flatten, Dense
    model = Sequential()
    model.add(
        Conv2D(filters=64, kernel_size=(5, 5), strides=(3, 3), padding='same', input_shape=(width, height, channels)))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(MaxPool2D(pool_size=(3, 3), strides=(3, 3), padding='same'))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation(activation))

    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation(activation))

    model.add(Dense(4))
    model.add(Activation('softmax'))
    return model


def load_model_from_bytes(model_bytes: bytes):
    import pickle
    weights = pickle.loads(model_bytes)
    model = create_model()
    model.set_weights(weights)
    return model


def infer(model, audio: np.array, starts: List[float], ends: List[float], sr=16000, fmin=50, fmax=350):
    """
    Perform tone inference on a slice of audio
    :param model: Keras model
    :param audio: Audio as a numpy array with float32 as dtype (librosa format)
    :param starts: List of start time in seconds
    :param ends: List of end time in seconds
    :param sr: Sample rate
    :param fmax: Max frequency
    :param fmin: Min frequency
    :return: np.ndarray (2D), An array of the probability of each tones
    """
    n = len(starts)
    if n > 1:  # triphone
        ends[:-1] = ends[1:]
        starts[1:] = starts[:-1]

    bufs = [melspectrogram_feature(audio, starts[i], ends[i], sr, fmin, fmax) for i in range(n)]
    x = np.asarray([load_image_from_bytes(b.read()) for b in bufs], dtype='float32')
    y_pred = model.predict(x)
    return y_pred


def get_non_toned_mask(phone_ids) -> np.ndarray:
    from gop_server import zh_config
    ret = ~np.isin(phone_ids, zh_config.toned_phones)
    return ret


def get_light_tone_mask(phone_ids) -> np.ndarray:
    tones = np.vectorize(lambda x: consonant2tone.get(x, -1))(phone_ids)
    return tones == 0


def get_tone_correctness(phone_ids, tones) -> np.ndarray:
    expected: np.ndarray = np.vectorize(lambda x: consonant2tone.get(x, -1))(phone_ids)
    ret = expected == tones
    # TODO: how to test the correctness of light tones
    ret[expected == -1] = True
    ret[expected == 0] = True
    return ret


def tc_likelihood(wav: np.ndarray, frame_alignment: np.ndarray, nonsil_mask: np.ndarray,
                  frame_shift_sec=0.010) -> np.ndarray:
    """
    Get log-likelihood of the current observation being one of the 4 tones

    :param wav: Audio as a numpy array with float32 as dtype (librosa format)
    :param frame_alignment: List of phone ids of each frame
    :param nonsil_mask: Non-silence phone mask
    :param frame_shift_sec: Time between two frames in seconds
    :return: np.ndarray (2D), An array of the log-likelihood of each tones
    """

    from gop_server import zh_config
    from gop_server.utils import frame_alignment_to_segment_indices as fa2si

    model = load_model_from_bytes(zh_config.tc_bytes)
    seg_idx = fa2si(frame_alignment)
    n = seg_idx.size
    # MFCC frame shift is 10ms (https://github.com/kaldi-asr/kaldi/blob/master/src/feat/feature-window.h#L55)
    seg_time = seg_idx * frame_shift_sec
    starts = []
    ends = []
    for i in range(n - 1):
        if nonsil_mask[i]:
            starts.append(seg_time[i])
            ends.append(seg_time[i + 1])
    ret = infer(model, wav, starts, ends)
    ret = np.log(ret)
    return ret


def get_got(phone_ids: np.ndarray, phone_feats: np.ndarray) -> np.ndarray:
    from gop_server import zh_config

    # get phone ids of the phones that have other tones
    same_based = np.zeros((phone_ids.size, 4), dtype=np.int)
    for i, pid in enumerate(phone_ids):
        if pid in zh_config.toned_phones:
            ids = [pi for pi in zh_config.phone2same_base[pid] if consonant2tone.get(pi, 0) != 0]
            same_based[i] = np.asarray(ids, dtype=np.int) - 1

    n_phones = phone_feats.shape[1] // 2
    tone_feats: np.ndarray = phone_feats[:, n_phones:]

    # get GOT of all 4 tones
    ret = np.zeros((tone_feats.shape[0], 4))
    for i in range(ret.shape[0]):
        if phone_ids[i] in zh_config.toned_phones:
            ret[i] = tone_feats[i, same_based[i]]
            # print(np.argmin(ret[i]) + 1)
    return ret


def ensemble_tones(GOT: np.ndarray, tone_ll: np.ndarray) -> np.ndarray:
    """
    Ensemble GOT and output of the tone classifier
    :param GOT: Goodness of pronunciation
    :param tone_ll: Tone likelihood
    :return: The observed tones (0 N/A, 1 一声, 2 二声, ...)
    """
    # mean = (GOT + tone_ll) / 2 # FIXME: GOT is not usable
    mean = tone_ll
    ret = np.argmax(mean, axis=1) + 1
    return ret
