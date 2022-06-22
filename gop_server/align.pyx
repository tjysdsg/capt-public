import cython
from kaldi.decoder import TrainingGraphCompilerOptions
from kaldi.matrix import Matrix
from kaldi.nnet3 import NnetSimpleComputationOptions
from kaldi.alignment import NnetAligner
from kaldi.hmm import split_to_phones
from gop_server import ModelPipline, LangConfig
import numpy as np
cimport numpy as np

ctypedef np.float_t float_t
ctypedef np.long_t long_t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dict nnet3_align(pipeline: ModelPipline,
                       np.ndarray[float_t, ndim=2] _features,
                       str transcript,
                       float transition_scale=1.0,
                       float self_loop_scale=0.1,
                       float acoustic_scale=0.1,
                       int frames_per_chunk=50,
                       int extra_left_context=0,
                       int extra_right_context=0,
                       int extra_left_context_initial=-1,
                       int extra_right_context_final=-1,
                       float beam=200.0,
                       float_t[:, :] online_ivectors = None,
                       int online_ivector_period=0,
                       word_boundary_info=None):
    cdef features = Matrix(_features)

    cdef gopts = TrainingGraphCompilerOptions()
    gopts.transition_scale = 0
    gopts.self_loop_scale = 0

    cdef decodable_opts = NnetSimpleComputationOptions()
    decodable_opts.acoustic_scale = acoustic_scale
    decodable_opts.frames_per_chunk = frames_per_chunk
    decodable_opts.extra_left_context = extra_left_context
    decodable_opts.extra_right_context = extra_right_context
    decodable_opts.extra_left_context_initial = extra_left_context_initial
    decodable_opts.extra_right_context_final = extra_right_context_final

    # create aligner if not cached
    if pipeline.nnet3_aligner is None:
        aligner = NnetAligner(transition_model=pipeline.trans_model, acoustic_model=pipeline.am_nnet, tree=pipeline.tree,
                              lexicon=pipeline.lexicon, symbols=pipeline.symbols, disambig_symbols=pipeline.disambig,
                              graph_compiler_opts=gopts, beam=beam, transition_scale=transition_scale,
                              self_loop_scale=self_loop_scale, decodable_opts=decodable_opts,
                              online_ivector_period=online_ivector_period)
        pipeline.nnet3_aligner = aligner
    else:
        aligner = pipeline.nnet3_aligner

    if online_ivectors is not None:
        result = aligner.align((features, Matrix(online_ivectors)), transcript)
    else:
        result = aligner.align(features, transcript)

    cdef list _word_alignment = None
    if word_boundary_info is not None:
        best_path = result['best_path']
        _word_alignment = aligner.to_word_alignment(best_path, word_boundary_info)

    # NnetAligner only does alignment per phone while we want alignment per frame
    success, split_ali = split_to_phones(aligner.transition_model, result['alignment'])
    # cdef list _observed_alignment = []
    if not success:
        raise RuntimeError("Alignment phone split failed")
    cdef list frame_alignment = []
    for entry in split_ali:
        phone = aligner.transition_model.transition_id_to_phone(entry[0])
        duration = len(entry)
        ####
        frame_alignment += [phone for _ in range(duration)]
        ####

    cdef list word_alignment = []
    cdef str ph
    cdef int start
    cdef int end
    cdef list wa
    if word_boundary_info is not None:
        from gop_server import empty_phones
        for wa in _word_alignment:
            ph = wa[0]
            if ph in empty_phones:
                continue
            start = wa[1]  # inclusive
            end = start + wa[2]  # exclusive
            word_alignment.append(list(set(frame_alignment[start:end])))
        return dict(
            frame_alignment=np.asarray(frame_alignment),
            word_alignment=np.asarray(word_alignment),
        )
    else:
        return dict(
            frame_alignment=np.asarray(frame_alignment),
        )


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef long_t[:] postprocess_alignment(config: LangConfig, long_t[:] expected_phones, long_t[:] frame_alignment):
    cdef int j = 0
    cdef int pi = 0
    cdef int n = frame_alignment.size
    cdef int prev_ph
    cdef int i

    # get expected frame sequence
    while j < n and frame_alignment[j] in config.empty_phones_id:  # skip leading empty phones from alignment
        j += 1

    prev_ph = frame_alignment[j]
    for i in range(j, n):
        if frame_alignment[i] in config.empty_phones_id:  # skip empty phones
            continue
        if prev_ph != frame_alignment[i]:
            prev_ph = frame_alignment[i]
            pi += 1
        frame_alignment[i] = expected_phones[pi] if pi < len(expected_phones) else 1

    return frame_alignment
