import logging
import numpy as np
from typing import List, Optional

logger = logging.getLogger('gop-server')


def get_base_correctness(phone_ids, scores) -> np.ndarray:
    import csv
    import os
    from gop_server import zh_config

    # read threshold file at gop_chinese/gop-threshold.csv
    threshold_file = os.path.join(zh_config.root_path, 'gop-threshold.csv')
    thres = list(csv.reader(open(threshold_file)))
    thresholds: np.ndarray = np.abs(np.asarray(thres, dtype=np.float))

    # get thresholds of current sentence
    thresholds = thresholds[phone_ids, 1]
    logger.info(f'Phone IDs: {phone_ids}')
    logger.info(f'Thresholds: {thresholds}')
    return np.abs(scores) <= thresholds


def _gop(wav: np.ndarray, transcript: str, pinyin: List[str]) -> dict:
    from gop_server import zh_config
    zh_config.init_align_pipeline()
    zh_config.init_gop_pipeline()

    pinyin_i = np.asarray([zh_config.phone_name2id[p] for p in pinyin])
    """(phone ID + 1) of the expected phones"""

    # VAD
    # after testing, VAD helps (especially long sentence)
    from gop_server.preprocess import remove_long_silence
    wav = remove_long_silence(wav)

    # MFCC
    from gop_server.mfcc import make_mfcc, make_mfcc_pitch
    features = make_mfcc(wav.astype(np.float), apply_cmvn=False, compress=False, high_freq=7600)
    # MFCC hires
    mfcc_hires = make_mfcc_pitch(wav.astype(np.float), apply_cmvn=False, compress=False, high_freq=7800)

    # ivectors from gop pipeline
    from gop_server.ivector import extract_ivectors_online
    ivectors = extract_ivectors_online(mfcc_hires[:, :40], zh_config.gop_pipeline.diag_ubm,
                                       zh_config.gop_pipeline.lda_mat,
                                       zh_config.gop_pipeline.ivector_extractor,
                                       zh_config.gop_pipeline.ivector_cmvn_stats)

    # get posteriors from gop pipeline
    from gop_server.nnet3 import nnet3_compute
    prob_matrix = nnet3_compute(zh_config.gop_pipeline.nnet, mfcc_hires, ivectors,
                                zh_config.gop_pipeline.ivector_period)

    # align
    from gop_server.align import nnet3_align, postprocess_alignment
    aligned = nnet3_align(zh_config.align_pipeline, features, transcript)

    frame_alignment = aligned['frame_alignment']
    # convert phone id in aligning model to one in scoring model
    frame_alignment = np.vectorize(zh_config.pid_align_to_pid_score.__getitem__)(frame_alignment)

    frame_alignment = postprocess_alignment(zh_config, pinyin_i, frame_alignment)

    # compute gop
    from gop_server.compute_gop import compute_gop, clean_phones, to_annotation
    phones, phone_feats, gop, nonsil_mask = compute_gop(zh_config, frame_alignment, prob_matrix)

    # tone classification
    from gop_server.tone_classifier import (tc_likelihood, ensemble_tones, get_got, get_non_toned_mask,
                                            get_light_tone_mask)
    from gop_server.preprocess import pcm2float
    wav_f = pcm2float(wav)
    GOT = get_got(pinyin_i, phone_feats)
    tone_likelihood = tc_likelihood(wav_f, np.asarray(frame_alignment), nonsil_mask)
    tones = ensemble_tones(GOT, tone_likelihood)
    # don't use result if is a light tone, or don't have a tone
    tones[get_non_toned_mask(pinyin_i)] = 0
    tones[get_light_tone_mask(pinyin_i)] = 0

    # clean phones and convert to annotations
    phones = clean_phones(zh_config, phones)
    phones_a = to_annotation(zh_config, phones)

    # convert scores to correctness
    from gop_server.tone_classifier import get_tone_correctness
    tcorr = get_tone_correctness(pinyin_i, tones)
    bcorr = get_base_correctness(pinyin_i, gop[:, 0])
    corr = np.asarray([bcorr, tcorr]).T

    # result
    ret = dict(annotations_a=phones_a, annotations=phones, gop=gop, tones=tones, corr=corr, phone_feats=phone_feats)

    from gop_server.utils import print_evaluation
    print_evaluation(phones_a, gop, tones, corr)

    return ret


def zh_gop_main(wav: np.ndarray, transcript: str, pinyin: Optional[List[str]] = None) -> dict:
    """
    Compute GOP scores for Chinese (Mandarin)
    :param wav: Audio in PCM16 format, dtype int16
    :param transcript: Transcript of an utterance, in simplified Chinese. Tokenization will be applied using `jieba`.
        https://github.com/fxsjy/jieba/
    :param pinyin: Expected pinyin, overwrite transcript if specified
    :return: Dictionary
        | annotations_a (list)      | Pinyin of expected phones                                |
        | annotations (numpy array) | Phone indices of expected phones                         |
        | gop (numpy array)         | GOP scores                                               |
        | tones (numpy array)       | observed tones (0 N/A, 1: 1st, 2: 2nd, 3: 3rd, 4: 4th)   |
        | corr (numpy array)        | correctness the phone and the tone of phones             |
        | phone_feats (numpy array) | raw phone-level features                                 |
    """
    # lazy loading
    from gop_server.transcript import clean_transcript
    from gop_server.pinyin import transcript_to_pinyin

    # clean transcript text
    transcript, segments = clean_transcript(transcript)
    logger.info(f'Tokenized transcript: {transcript}')

    # get expected phones indices from transcript
    if pinyin is None:
        logger.warning(f'Standard pinyin not found, generating it')
        pinyin = transcript_to_pinyin(segments)

    pinyin = [p for p in pinyin if p != '']
    logger.info(f'Expected pinyin: {pinyin}')

    return _gop(wav, transcript, pinyin)
