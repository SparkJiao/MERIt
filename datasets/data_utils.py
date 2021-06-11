from transformers import PreTrainedTokenizer


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
