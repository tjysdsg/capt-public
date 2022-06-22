import cython
from kaldi.feat.mfcc import Mfcc
from kaldi.matrix.compressed import CompressedMatrix
from kaldi.feat.mfcc import MfccOptions
from kaldi.feat.mel import MelBanksOptions
from kaldi.feat.window import FrameExtractionOptions
from kaldi.feat.pitch import PitchExtractionOptions, ProcessPitchOptions, compute_and_process_kaldi_pitch
from kaldi.matrix import Matrix
import numpy as np
cimport numpy as np

ctypedef np.float_t float_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[float_t, ndim=2] make_mfcc(np.ndarray[float_t, ndim=1] wav, bint apply_cmvn, bint compress = True,
                                            float dither = 1.0, int high_freq=7600):
    """
    Compute MFCC

    This is the equivalent of kaldi's `steps/make_mfcc.sh` script.
    :param wav: wav matrix
    :param apply_cmvn: whether to apply CMVN stats to the result
    :param compress: whether to compress the resulting matrix (using kaldi's `CompressedMatrix`)
    :param dither: How much dither to apply. Same as `--dither` option in `make_mfcc.sh`
    :param high_freq: high cutoff frequency
    :return: numpy array containing MFCC features
    """
    # default mfcc configs
    cdef float sample_freq = 16000.0
    # https://github.com/kaldi-asr/kaldi/blob/master/src/feat/mel-computations.h#L43
    cdef mel_opts = MelBanksOptions(num_bins=40)
    mel_opts.low_freq = 20
    mel_opts.high_freq = high_freq

    # https://github.com/kaldi-asr/kaldi/blob/master/src/feat/feature-window.h#L35
    cdef frame_opts = FrameExtractionOptions()
    frame_opts.samp_freq = sample_freq
    frame_opts.dither = dither

    cdef mfcc_opts = MfccOptions()
    mfcc_opts.use_energy = False
    mfcc_opts.num_ceps = 40
    mfcc_opts.mel_opts = mel_opts
    mfcc_opts.frame_opts = frame_opts
    cdef mfcc_computer = Mfcc(mfcc_opts)
    cdef wavmat = Matrix(wav.reshape(1, -1))
    cdef features = mfcc_computer.compute_features(wavmat.row(0), sample_freq=sample_freq, vtln_warp=1.0)

    # apply cmvn
    if apply_cmvn:
        from kaldi.transform.cmvn import Cmvn
        cmvn = Cmvn(features.shape[1])
        cmvn.accumulate(features)
        cmvn.apply(features)

    # compress
    if compress:
        compressed = CompressedMatrix.new(features)
        compressed.copy_to_mat(features)
    return features.numpy().astype(np.float)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[float_t, ndim=2] make_mfcc_pitch(np.ndarray[float_t, ndim=1] wav,
                                                  bint apply_cmvn,
                                                  bint compress = True,
                                                  float dither = 1.0,
                                                  int high_freq=7600):
    cdef np.ndarray[float_t, ndim=2] mfcc = make_mfcc(wav, apply_cmvn=apply_cmvn, compress=compress, dither=dither,
                                                      high_freq=high_freq)
    cdef np.ndarray[float_t, ndim=2] pitch = make_pitch(wav)
    return np.hstack([mfcc, pitch])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[float_t, ndim=2] make_pitch(np.ndarray[float_t, ndim=1] wav):
    cdef wavmat = Matrix(wav.reshape(1, -1))

    # https://github.com/kaldi-asr/kaldi/blob/master/src/feat/pitch-functions.h#L42
    cdef pitch_opts = PitchExtractionOptions()
    pitch_opts.samp_freq = 16000
    cdef process_opts = ProcessPitchOptions()

    # https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/compute-kaldi-pitch-feats.cc#L97
    # plus
    # https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/process-kaldi-pitch-feats.cc#L80
    ret = compute_and_process_kaldi_pitch(pitch_opts, process_opts, wavmat.row(0))
    return ret.numpy().astype(np.float)
