import cython
from kaldi.hmm import TransitionModel
from gop_server import LangConfig
import logging
cimport numpy as np
import numpy as np

ctypedef np.float_t float_t
ctypedef np.long_t long_t

logger = logging.getLogger('gop-server')


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[long_t, ndim=1] clean_phones(config: LangConfig, long_t[:] phones):
    """
    Clean phone ids, remove silence

    :param config: Language configuration
    :param phones: List of phone indices in an utterance
    :return: Indices of the phones occurred in the utterance
    """
    cdef int e
    phones = np.asarray([e for e in phones if e not in config.empty_phones_id])
    return np.asarray(phones)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list to_annotation(config: LangConfig, long_t[:] annotations):
    cdef int e
    cdef str a
    cdef list _annotations = [config.phone_id2name[e] for e in annotations]
    return [config.phones2annotation[a] for a in _annotations]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[float_t, ndim=1] plpp(int n_phones, float_t[:] prob_row, list pdf2phones):
    cdef np.ndarray[float_t, ndim=1] ret = np.zeros(n_phones)
    cdef int i
    cdef int ph
    cdef int idx
    cdef set dest_idxs
    cdef int n
    for i in range(prob_row.size):
        n = len(pdf2phones[i])
        dest_idxs = set(np.asarray(pdf2phones[i]) - 1)
        for idx in dest_idxs:
            ret[idx] += prob_row[i]

    return np.log(ret)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef list get_pdf_to_phones_map(trans_model: TransitionModel):
    cdef list ret = [[] for _ in range(trans_model.num_pdfs())]
    cdef int n_trans_id = trans_model.num_transition_ids()
    cdef int trans_id = 0
    cdef int pdf_id = 0
    cdef int phone
    cdef int i
    for i in range(n_trans_id):
        trans_id = i + 1  # trans-id is one-based, https://kaldi-asr.org/doc/hmm.html#transition_model_identifiers
        pdf_id = trans_model.transition_id_to_pdf(trans_id)
        phone = trans_model.transition_id_to_phone(trans_id)
        # hmm_state = trans_model.transition_id_to_hmm_state(trans_id)
        ret[pdf_id].append(phone)
    return ret


@cython.boundscheck(False)
@cython.wraparound(False)
cdef list base_tone_scores(dict phone2same_base, dict phone2same_tone, int phone_id, int duration, pfeat: np.ndarray):
    same_toned = np.asarray(phone2same_tone[phone_id + 1], dtype=np.int)
    same_based = np.asarray(phone2same_base[phone_id + 1], dtype=np.int)
    gop_base = np.log(np.sum(np.exp(pfeat[same_toned - 1]))) - pfeat[phone_id]
    gop_tone = np.log(np.sum(np.exp(pfeat[same_based - 1]))) - pfeat[phone_id]
    return [np.exp(gop_base) / duration, np.exp(gop_tone) / duration]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple compute_gop(config: LangConfig, long_t[:] frame_alignment, float_t[:, :] prob_matrix):
    """
    Compute goodness-of-pronunciation (GOP) using a pretrained model

    :param config: Language configuration
    :param frame_alignment: Frame-level phone alignment matrix
    :param prob_matrix: Output of nnet3_compute
    :returns (phones, phone_feats, gop)
     - phones: List of phone indices
     - phone_feats: Phone level features, each row contains the base and tone log-likelihood of a phone
     - gop: [(gop_base, gop_tone), ...], the GOP scores of every phone in the utterance
    """
    cdef trans_model = config.gop_pipeline.trans_model
    cdef int n_phones = trans_model.num_phones()
    prob_matrix = np.exp(prob_matrix)

    # get mapping from pdf-id to phone-id
    cdef list pdf2phones = get_pdf_to_phones_map(trans_model)

    # check frame size, make sure it's smaller than the output size of nnet3
    cdef int n_frames = frame_alignment.size
    if n_frames != prob_matrix.shape[0]:
        logger.warning('The frame numbers of alignment and prob are not equal')
        if n_frames > prob_matrix.shape[0]:
            n_frames = prob_matrix.shape[0]

    # cur_phone_id and next_phone_id start from 0
    cdef int cur_phone_id = frame_alignment[0] - 1
    cdef int next_phone_id = 0
    cdef int duration = 0
    cdef int phone_id = 0
    cdef int i
    cdef list scores = []
    cdef list phone_feats = []
    cdef list phones = []
    cdef list silence_i = []
    cdef float gop_base
    cdef float gop_tone

    # masks
    cdef np.ndarray[long_t, ndim=1] empty_phone_mask = np.asarray(config.empty_phones_id) - 1
    empty_phone_mask = empty_phone_mask[empty_phone_mask >= 0]
    cdef np.ndarray[np.npy_bool, ndim=1] toned_phone_mask = np.zeros(n_phones, dtype=np.bool)
    toned_phone_mask[np.asarray(config.toned_phones) - 1] = True
    cdef np.ndarray[np.npy_bool, ndim=1] non_toned_phone_mask = np.ones(n_phones, dtype=np.bool)
    non_toned_phone_mask[toned_phone_mask] = False
    cdef np.ndarray[np.npy_bool, ndim=1] current_mask

    cdef np.ndarray[float_t, ndim=1] pfeat = np.zeros(n_phones, dtype=np.float)
    for i in range(n_frames):
        duration += 1
        pfeat += plpp(n_phones, prob_matrix[i, :], pdf2phones)

        if cur_phone_id + 1 in config.toned_phones:
            current_mask = non_toned_phone_mask
        else:
            current_mask = toned_phone_mask

        next_phone_id = frame_alignment[i + 1] - 1 if i < n_frames - 1 else -1
        if next_phone_id != cur_phone_id:
            pfeat /= duration

            # prevent silence to be the best match if silence is not the expected phone
            if cur_phone_id + 1 not in config.empty_phones_id:
                pfeat[empty_phone_mask] = -np.inf

            # GOP, GOT
            features = np.asarray([
                base_tone_scores(config.phone2same_base, config.phone2same_tone, pi, duration, pfeat)
                if pi + 1 in config.toned_phones else
                [pfeat[cur_phone_id] - np.max(pfeat), 0]
                for pi in range(n_phones)
            ])
            scores.append(features[cur_phone_id])

            # phone indices
            phones.append(cur_phone_id + 1)

            # remember position of silence, remove them later
            if cur_phone_id + 1 in config.empty_phones_id:
                silence_i.append(phone_id)
            phone_id += 1

            phone_feats.append(np.ravel(features.T))

            # reset
            pfeat = np.zeros(n_phones, dtype=np.float)
            duration = 0
            cur_phone_id = next_phone_id

    # remove silence
    cdef mask = np.ones(len(phones), dtype=bool)
    mask[silence_i] = False
    cdef phones_ = np.asarray(phones)[mask, ]
    cdef phone_feats_ = np.asarray(phone_feats)[mask, ]
    cdef scores_ = np.asarray(scores)[mask, ]
    return phones_, phone_feats_, scores_, mask
