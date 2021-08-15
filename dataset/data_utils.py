import copy

from transformers import PreTrainedTokenizer
from typing import List, Set


def is_bpe(_tokenizer: PreTrainedTokenizer):
    return _tokenizer.__class__.__name__ in [
        "RobertaTokenizer",
        "LongformerTokenizer",
        "BartTokenizer",
        "RobertaTokenizerFast",
        "LongformerTokenizerFast",
        "BartTokenizerFast",
    ]


def get_sep_tokens(_tokenizer: PreTrainedTokenizer):
    return [_tokenizer.sep_token] * (_tokenizer.max_len_single_sentence - _tokenizer.max_len_sentences_pair)


def dfs(src: List[int], vis: Set, state: List[int], ans: List[List[int]]):
    if len(state) == len(src):
        if not all(a == b for a, b in zip(src, state)):
            ans.append(state)

    for x in src:
        if x not in vis:
            new_vis = copy.deepcopy(vis)
            new_vis.add(x)
            new_state = copy.deepcopy(state)
            new_state.append(x)
            dfs(src, new_vis, new_state, ans)


def get_all_permutation(array: List[int]):
    res = []
    dfs(array, set(), list(), res)
    for state in res:
        assert not all(a == b for a, b in zip(state, array))
    return res
