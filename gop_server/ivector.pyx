from kaldi.online2 import (OnlineIvectorExtractionInfo, OnlineIvectorExtractorAdaptationState, OnlineIvectorFeature)
from kaldi.feat.online import (OnlineSpliceOptions, OnlineMatrixFeature)
from kaldi.matrix import Matrix, DoubleMatrix
from kaldi.matrix.compressed import CompressedMatrix, CompressionMethod
from kaldi.ivector import IvectorExtractor
from kaldi.gmm import DiagGmm
import numpy as np
cimport numpy as np

ctypedef np.float_t float_t

# steps/online/nnet2/extract_ivectors_online.sh
cpdef np.ndarray[float_t, ndim=2] extract_ivectors_online(np.ndarray[float_t, ndim=2] _features,
                                                          ubm: DiagGmm,
                                                          lda: Matrix,
                                                          ivector_extractor: IvectorExtractor,
                                                          float_t[:, :] _global_cmvn_stats,
                                                          int ivector_period = 10,
                                                          int splice_opts_left_context = 3,
                                                          int splice_opts_right_context = 3,
                                                          ):
    cdef global_cmvn_stats = DoubleMatrix(_global_cmvn_stats)

    # config ivector extraction
    cdef splice_opts = OnlineSpliceOptions()
    splice_opts.left_context = splice_opts_left_context
    splice_opts.right_context = splice_opts_right_context

    cdef ivector_info = OnlineIvectorExtractionInfo()
    ivector_info.diag_ubm = ubm
    ivector_info.lda_mat = lda
    ivector_info.extractor = ivector_extractor
    ivector_info.global_cmvn_stats = global_cmvn_stats
    ivector_info.splice_opts = splice_opts
    ivector_info.ivector_period = ivector_period
    ivector_info.max_count = 0
    ivector_info.min_post = 0.025
    ivector_info.num_gselect = 5
    ivector_info.posterior_scale = 0.1
    ivector_info.use_most_recent_ivector = False
    ivector_info.max_remembered_frames = 1000
    ivector_info.check()

    cdef adaption_state = OnlineIvectorExtractorAdaptationState.from_info(ivector_info)
    cdef features = Matrix(_features)
    cdef matrix_feature = OnlineMatrixFeature(features)
    cdef ivector_feature = OnlineIvectorFeature(ivector_info, matrix_feature)
    ivector_feature.set_adaptation_state(adaption_state)

    # n = 1 if repeat else ivector_info.ivector_period
    cdef int n = ivector_info.ivector_period
    cdef int n_ivectors = <int> ((features.shape[0] + n - 1) / n)
    cdef ivectors = Matrix(n_ivectors, ivector_feature.dim())
    cdef int i
    cdef int t
    for i in range(n_ivectors):
        t = <int> (i * n)
        tmp = ivectors.row(i)
        ivector_feature.get_frame(t, tmp)

    # compress
    cdef compressed = CompressedMatrix.new(ivectors, CompressionMethod.AUTO)
    compressed.copy_to_mat(ivectors)
    return ivectors.numpy().astype(np.float)
