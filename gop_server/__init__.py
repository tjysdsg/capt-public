import cython

# compile cython files
if not cython.compiled:
    import numpy
    import pyximport

    pyximport.install(language_level=3, setup_args={"include_dirs": numpy.get_include()})

import os
import sys
import json
import logging
from kaldi.util.io import Input
from kaldi.ivector import IvectorExtractor
from kaldi.gmm import DiagGmm
from kaldi.matrix import Matrix
from kaldi.lat.align import WordBoundaryInfo, WordBoundaryInfoNewOpts
from os.path import abspath, dirname
from typing import Callable, Dict
from gop_server.read_kaldi import load_nnet3_model, load_lex, load_symbol, load_tree, load_disambig
from gop_server import tone_classifier as tc
from datetime import datetime

project_root_dir = dirname(dirname(abspath(__file__)))

# configuring logger
log_dir = os.path.join(project_root_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)

datefmt = '%Y-%m-%d-%H-%M-%S'
timestr = datetime.now().strftime(datefmt)
log_file = os.path.join(log_dir, f'{timestr}.log')

logger = logging.getLogger('gop-server')
logger.setLevel(logging.INFO)

formatter = logging.Formatter(fmt='[%(levelname)s] %(asctime)s: %(message)s', datefmt=datefmt)

file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)


class ModelPipline:
    def __init__(self, tree_path: str, model_path: str, lang_path: str, lex_path: str, symbol_path: str,
                 disambig_path: str, ivector_path: str = '', lda_path: str = '',
                 ubm_path: str = '', extractor_path: str = '', ivector_period: int = 0, ivector_cmvn_stats=None,
                 splice_opts_left_context=3, splice_opts_right_context=3, word_boundary_info_path: str = ''
                 ):
        self.tree_path = tree_path
        self.model_path = model_path
        self.lang_path = lang_path
        self.lex_path = lex_path
        self.symbol_path = symbol_path
        self.disambig_path = disambig_path
        self.lexicon = load_lex(lex_path)
        self.symbols = load_symbol(symbol_path)
        self.tree = load_tree(tree_path)
        self.disambig = load_disambig(disambig_path)
        self.am_nnet, self.trans_model = load_nnet3_model(self.model_path)
        self.nnet = self.am_nnet.get_nnet()

        # optionals
        self.ivector_path = ivector_path
        self.lda_path = lda_path
        self.ubm_path = ubm_path
        self.extractor_path = extractor_path
        self.ivector_period = ivector_period
        self.ivector_cmvn_stats = ivector_cmvn_stats
        self.splice_opts_left_context = splice_opts_left_context
        self.splice_opts_right_context = splice_opts_right_context
        self.word_boundary_info_path = word_boundary_info_path
        self.diag_ubm = None
        self.lda_mat = None
        self.ivector_extractor = None

        if ubm_path != "":
            ubm_ifs = Input(ubm_path, True)
            self.diag_ubm = DiagGmm()
            self.diag_ubm.read(ubm_ifs.stream(), True)

        if lda_path != "":
            lda_ifs = Input(lda_path, True)
            self.lda_mat = Matrix()
            self.lda_mat.read_(lda_ifs.stream(), True)

        if extractor_path != "":
            extractor_ifs = Input(extractor_path, True)
            self.ivector_extractor = IvectorExtractor()
            self.ivector_extractor.read(extractor_ifs.stream(), True)

        if word_boundary_info_path != "":
            self.word_boundary_info = WordBoundaryInfo.from_file(WordBoundaryInfoNewOpts(), word_boundary_info_path)

        # cache
        self.nnet3_aligner = None


class LangConfig:
    def __init__(self, root_path: str, handler: Callable):
        self.root_path = root_path
        self.handler: Callable = handler

        logger.info('Preloading config and data files')

        # preload some txt and json files
        with open(os.path.join(root_path, "empty_phones.txt")) as f:
            self.empty_phones_id = [int(line) for line in f]

        with open(os.path.join(root_path, "phones.txt")) as f:
            _lines = [line.strip().split() for line in f]
            self.phone_id2name = {int(line[1]): line[0] for line in _lines}
            self.phone_name2id = {line[0]: int(line[1]) for line in _lines}

        with open(os.path.join(root_path, 'phones-to-annotation.json'), 'r') as f:
            self.phones2annotation = json.load(f)

        # get base and tone info about phones
        if os.path.exists(os.path.join(self.root_path, 'phone2same_base.json')):
            with open(os.path.join(self.root_path, 'phone2same_base.json')) as f:
                self.phone2same_base: dict = json.load(f)
                self.phone2same_base = {int(k): v for k, v in self.phone2same_base.items()}
            with open(os.path.join(self.root_path, 'phone2same_tone.json')) as f:
                self.phone2same_tone: dict = json.load(f)
                self.phone2same_tone = {int(k): v for k, v in self.phone2same_tone.items()}

            phones = []
            for p in self.phone2same_tone.values():
                phones += p
            self.toned_phones = list(set(phones))
        else:
            self.phone2same_tone = dict()
            self.phone2same_base = dict()
            self.toned_phones = []

        logger.info('Completed preloading basic data files')


class ZhConfig(LangConfig):
    def __init__(self):
        from gop_server.zh_gop import zh_gop_main

        root_path = os.path.join(project_root_dir, 'gop_chinese')
        super().__init__(
            root_path=root_path,
            handler=zh_gop_main
        )

        pa2ps = json.load(open(os.path.join(root_path, 'pid_align_to_pid_score.json')))
        self.pid_align_to_pid_score = {int(k): v for k, v in pa2ps.items()}

        with open(os.path.join(self.root_path, 'phone2base_tone.json')) as f:
            self.phone2base_tone = json.load(f)
            self.phone2base_tone = {int(k): v for k, v in self.phone2base_tone.items()}

        with open(os.path.join(self.root_path, 'base_tone2phone.json')) as f:
            self.base_tone2phone = json.load(f)

        # load ToneNet
        with open(os.path.join(root_path, 'ToneNet.pickle'), 'rb') as f:
            self.tc_bytes = f.read()

        # non-trivial initializations, do later
        self.align_pipeline: ModelPipline = None
        self.gop_pipeline: ModelPipline = None

    def init_align_pipeline(self) -> None:
        """
        Load the TDNN model for aligning and relevant files. Won't do anything if called multiple times.
        """
        if self.align_pipeline is not None:
            return

        # tdnn for aligning audio
        lang_path = os.path.join(project_root_dir, 'tdnn_align', 'lang')
        self.align_pipeline = ModelPipline(
            model_path=os.path.join(project_root_dir, 'tdnn_align', 'tdnn_sp', 'final.mdl'),
            tree_path=os.path.join(project_root_dir, 'tdnn_align', 'tdnn_sp', 'tree'),
            lang_path=lang_path,
            lex_path=os.path.join(lang_path, 'L.fst'),
            symbol_path=os.path.join(lang_path, 'words.txt'),
            disambig_path=os.path.join(lang_path, 'phones', 'disambig.int'),
        )

    def init_gop_pipeline(self) -> None:
        """
        Load the TDNN model for GOP and relevant files. Won't do anything if called multiple times.
        """
        if self.gop_pipeline is not None:
            return

        import numpy as np
        ivector_cmvn_stats = np.asarray([
            # tdnn_score/extractor/global_cmvn.stats
            4.982863e+10, -5.991834e+09, -6.287509e+09, -1.433221e+09, -1.52946e+10, -9.10766e+09, -1.471071e+10,
            -9.935119e+09, -1.033122e+10, -7.373663e+09, -9.266507e+09, -5.89387e+09, -9.317053e+09, -3.88284e+09,
            -7.774835e+09, -3.009649e+09, -4.793519e+09, -1.259678e+09, -2.257356e+09, -2.038197e+08, -7.896355e+08,
            9.60449e+07, -1.374521e+08, -1.790353e+07, 2.413269e+08, -7.467374e+07, 3.693677e+08, -1.168874e+08,
            3.15988e+08, -2.706289e+08, 1.668748e+08, -4.039211e+08, 3.05525e+08, -1.588691e+08, 2.526846e+08,
            -2.827762e+08, -4.432634e+07, -1.52894e+08, -6.515878e+07, -2.517534e+07, 5.68017e+08, 4.572781e+12,
            2.793279e+11, 2.60608e+11, 2.86564e+11, 6.704984e+11, 3.653587e+11, 6.080814e+11, 3.908465e+11,
            3.899764e+11, 3.158944e+11, 3.64831e+11, 2.968505e+11, 3.25104e+11, 1.9831e+11, 2.251081e+11, 1.063795e+11,
            1.173472e+11, 5.623634e+10, 4.707055e+10, 2.288816e+10, 1.329537e+10, 4.150353e+09, 6.230184e+08,
            1.665151e+08, 2.109218e+09, 4.961134e+09, 8.133012e+09, 1.060787e+10, 1.342184e+10, 1.540732e+10,
            1.656645e+10, 1.724727e+10, 1.811497e+10, 1.750119e+10, 1.339395e+10, 1.155018e+10, 1.143706e+10,
            8.98112e+09, 6.471408e+09, 4.553445e+09, 0,
        ]).reshape((1, -1))
        lang_path = os.path.join(project_root_dir, 'tdnn_score', 'lang')
        ivector_path = os.path.join(project_root_dir, 'tdnn_score', 'extractor')
        self.gop_pipeline = ModelPipline(
            model_path=os.path.join(project_root_dir, 'tdnn_score', 'tdnn_sp', 'final.mdl'),
            tree_path=os.path.join(project_root_dir, 'tdnn_score', 'tdnn_sp', 'tree'),
            lang_path=lang_path,
            lex_path=os.path.join(lang_path, 'L.fst'),
            symbol_path=os.path.join(lang_path, 'words.txt'),
            disambig_path=os.path.join(lang_path, 'phones', 'disambig.int'),
            ivector_path=ivector_path,
            lda_path=os.path.join(ivector_path, 'final.mat'),
            ubm_path=os.path.join(ivector_path, 'final.dubm'),
            extractor_path=os.path.join(ivector_path, 'final.ie'),
            ivector_cmvn_stats=ivector_cmvn_stats,
            # tdnn_score/ivectors_dev/conf/ivector_extractor.conf
            ivector_period=10,
        )


class ServerConfig:
    def __init__(self, api_config_path: str):
        with open(api_config_path) as f:
            self.configs = json.load(f)

        self.asr_api_url = self.configs['asr']
        self.tts_api_url = self.configs['tts']


zh_config = ZhConfig()
lang_configs: Dict[str, LangConfig] = dict(zh=zh_config)

empty_phones = ['<eps>', 'sil', 'SIL', 'SPN', 'spn', '$0']

audio_save_path = os.path.join(project_root_dir, 'data', 'audio')

server_config_dir = os.path.join(project_root_dir, 'configs')

server_config = ServerConfig(os.path.join(server_config_dir, 'external_api.json'))
