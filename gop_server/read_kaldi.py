from kaldi.nnet3 import AmNnetSimple
from kaldi.tree import ContextDependency
from kaldi.hmm import TransitionModel
from kaldi.util.io import Input, xopen
from kaldi.fstext import StdVectorFst, read_fst_kaldi, SymbolTable
from typing import Tuple, List
import os


def _check_file(p: str) -> None:
    if not os.path.exists(p):
        raise NameError("File at '{}' doesn't exist".format(p))


def load_nnet3_model(model_path: str) -> Tuple[AmNnetSimple, TransitionModel]:
    _check_file(model_path)
    ifs = Input(model_path, True)
    trans_model = TransitionModel()
    trans_model.read(ifs.stream(), True)
    nnet = AmNnetSimple()
    nnet.read(ifs.stream(), True)
    return nnet, trans_model


def load_tree(tree_path: str) -> ContextDependency:
    _check_file(tree_path)
    ifs = Input(tree_path, True)
    tree = ContextDependency()
    tree.read(ifs.stream(), True)
    return tree


def load_lex(lex_path: str) -> StdVectorFst:
    _check_file(lex_path)
    return read_fst_kaldi(lex_path)


def load_symbol(symbol_path: str) -> SymbolTable:
    _check_file(symbol_path)
    return SymbolTable.read_text(symbol_path)


def load_disambig(disambig_path: str) -> List[int]:
    _check_file(disambig_path)
    with xopen(disambig_path, "rt") as ki:
        return [int(line.strip()) for line in ki]
