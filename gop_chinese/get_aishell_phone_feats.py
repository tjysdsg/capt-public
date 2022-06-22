from os.path import abspath, dirname
from typing import List
import logging
import os
import sys
import numpy as np
from multiprocessing import Process

data_dir = '/NASdata/AudioData/AISHELL-ASR-SSB/SPEECHDATA'

file_dir = dirname(abspath(__file__))
project_root_dir = dirname(file_dir)
sys.path.insert(0, project_root_dir)
from gop_server.zh_gop import zh_gop_main
from gop_server import zh_config

# configuring logger
logging.root.setLevel(logging.INFO)
log_file = os.path.join(file_dir, 'get_aishell_phone_feats.log')
logger = logging.getLogger()
file_handler = logging.FileHandler(log_file)
logger.addHandler(file_handler)
stdout_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stdout_handler)

# init
transcript_path = os.path.join(zh_config.root_path, 'aishell-annotation', 'trans.txt')
with open(transcript_path, 'r') as f:
    file_trans = [line.split() for line in f]

annotation_path = os.path.join(zh_config.root_path, 'aishell-annotation', 'results.txt')
with open(transcript_path, 'r') as f:
    annotated_files = set([line.split()[0] for line in f])

# prepare output dir
output_dir = os.path.join(file_dir, 'aishell_phone_feats')
os.makedirs(output_dir, exist_ok=True)


def worker(data: List[str]):
    file = data[0]
    trans = data[1]
    logger.info('Computing GOP for {}'.format(file))
    file_path = os.path.join(data_dir, file[:7], file)  # SSBabcd/SSBabcdxxxx.wav
    out_path = os.path.join(output_dir, f'{file}.txt')
    if file not in annotated_files:
        logger.warning('Skipping {} since it\'s not annotated'.format(file))
        return
    try:
        s = zh_gop_main(file_path, trans)['phone_feats']
        np.savetxt(out_path, s)
    except Exception as e:
        logger.error('Compute gop for {} failed'.format(file))
        logger.error(str(e))
        return None
    logger.info('Computed GOP for {}'.format(file))


def main():
    nj = 8
    # remove previously calculated ones
    n = len(file_trans)
    trans = []
    for i in range(n):
        file = file_trans[i][0]
        out_path = os.path.join(output_dir, f'{file}.txt')
        if not os.path.exists(out_path):
            trans.append(file_trans[i])
        else:
            logger.warning(f'Skipping {file} since it\'s already been processed')

    # calculate phone feats concurrently
    n = len(trans)
    for i in range(n):
        data = trans[i: i + nj]
        ps = [Process(target=worker, args=(data[j],)) for j in range(nj) if len(data) > j]
        for p in ps:
            p.start()

        for j, p in enumerate(ps):
            p.join()
            print(f'Batch completed: {j + 1}/{nj}')

        print(f"=========== Progress: {i}/{n} ============")


if __name__ == '__main__':
    main()
