import cython
from kaldi.nnet3 import (Nnet, NnetSimpleComputationOptions, set_batchnorm_test_mode,
                         set_dropout_test_mode, collapse_model,
                         CollapseModelConfig, CachingOptimizingCompiler,
                         DecodableNnetSimple)
from kaldi.matrix import Matrix, Vector
import numpy as np
cimport numpy as np

ctypedef np.float_t float_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[float_t, ndim=2] nnet3_compute(nnet: Nnet, np.ndarray[float_t, ndim=2] _features,
                                               float_t[:, :] _online_ivectors = None,
                                               int online_ivector_period = 0):
    cdef features = Matrix(_features)
    cdef opts = NnetSimpleComputationOptions()
    opts.frames_per_chunk = 50
    opts.extra_left_context = 0
    opts.extra_right_context = 0
    opts.extra_left_context_initial = -1
    opts.extra_right_context_final = -1
    opts.acoustic_scale = 1.0

    set_batchnorm_test_mode(True, nnet)
    set_dropout_test_mode(True, nnet)
    collapse_model(CollapseModelConfig(), nnet)
    cdef compiler = CachingOptimizingCompiler(nnet, opts.compiler_config)
    if _online_ivectors is not None:
        online_ivectors = Matrix(_online_ivectors)
        nnet_computer = DecodableNnetSimple(opts, nnet, Vector(), features, compiler, None, online_ivectors,
                                        online_ivector_period)
    else:
        nnet_computer = DecodableNnetSimple(opts, nnet, Vector(), features, compiler, None)
    cdef int n_frames = nnet_computer.num_frames()
    cdef ret = Matrix(n_frames, nnet_computer.output_dim())
    cdef int i
    for i in range(n_frames):
        row = ret.row(i)
        nnet_computer.get_output_for_frame(i, row)
    return ret.numpy().astype(np.float)
