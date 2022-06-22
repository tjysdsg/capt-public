import numpy as np
import logging

logger = logging.getLogger('gop-server')


def get_error_str(e: Exception) -> str:
    import sys
    import traceback
    _, _, tb = sys.exc_info()
    ret = f'{str(e)}\n\n' + ''.join(traceback.format_tb(tb))
    return ret


def print_evaluation(phones, gop, tones, corr):
    logger.info('============= RESULTS ============')
    for i in range(gop.shape[0]):
        tone_names = ['无', '一声', '二声', '三声', '四声']
        ph = phones[i]
        t = tone_names[tones[i]]
        bcorr = corr[i, 0]
        tcorr = corr[i, 1]
        logger.info(
            f'{ph} | {gop[i, 0]}'
            f' {"✓" if bcorr else "✖"}, {gop[i, 1]} {"✓" if tcorr else "✖"}'
            f' {t}'
        )
    logger.info('==================================')


def edit_distance(a, b, mx=-1):
    from array import array

    def result(d):
        return d if mx < 0 else False if d > mx else True

    if a == b:
        return result(0)
    la, lb = len(a), len(b)
    if 0 <= mx < abs(la - lb):
        return result(mx + 1)
    if la == 0:
        return result(lb)
    if lb == 0:
        return result(la)
    if lb > la:
        a, b, la, lb = b, a, lb, la

    cost = array('i', range(lb + 1))
    for i in range(1, la + 1):
        cost[0] = i
        ls = i - 1
        mn = ls
        for j in range(1, lb + 1):
            ls, act = cost[j], ls + int(a[i - 1] != b[j - 1])
            cost[j] = min(ls + 1, cost[j - 1] + 1, act)
            if ls < mn:
                mn = ls
        if 0 <= mx < mn:
            return result(mx + 1)
    if 0 <= mx < cost[lb]:
        return result(mx + 1)
    return result(cost[lb])


def frame_alignment_to_segment_indices(fa: np.ndarray) -> np.ndarray:
    diff = fa[1:] - fa[:-1]
    idx = np.flatnonzero(diff)
    idx += 1
    idx = np.insert(idx, 0, 0)
    idx = np.append(idx, fa.size)
    return idx
