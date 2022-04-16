import torch
from .base import VocabGenerator
from typing import Callable


class CharVocabGenerator(VocabGenerator):
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
        super(CharVocabGenerator, self).__init__(
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
        return list(transcript)
