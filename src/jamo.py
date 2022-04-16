from typing import Callable
from .base import VocabGenerator, Tokenizer
from hangul_utils import split_syllables, join_jamos


class JamoVocabGenerator(VocabGenerator):
    def __init__(
            self,
            save_path: str,
            script_spliter: Callable,
            preprocessor: Callable,
            pad_id: int = 0,
            unk_id: int = 1,
            sos_id: int = 2,
            sep_id: int = 3,
            eos_id: int = 4,
            space_id: int = 5,
            blank_id: int = 6,
            language: str = 'kor'
    ):
        super(JamoVocabGenerator, self).__init__(
            save_path,
            script_spliter,
            preprocessor,
            pad_id,
            unk_id,
            sos_id,
            sep_id,
            eos_id,
            space_id,
            blank_id,
            language
        )

    @staticmethod
    def _split(transcript):
        return list(split_syllables(transcript))


class JamoTokenizer(object):
    def __init__(self, vocab_path: str, add_blank: bool = True, add_special_tokens: bool = True):
        super(JamoTokenizer, self).__init__(vocab_path, add_blank, add_special_tokens)

    def
