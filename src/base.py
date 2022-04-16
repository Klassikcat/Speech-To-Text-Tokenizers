import re
import csv
import tqdm
from typing import Callable, Tuple

import torch


class VocabGenerator(object):
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
        self.pad_id = pad_id
        self.unk_id = unk_id
        self.sos_id = sos_id
        self.sep_id = sep_id
        self.eos_id = eos_id
        self.space_id = space_id
        self.blank_id = blank_id

        self.vocab = {
            '[PAD]': self.pad_id,
            '[UNK]': self.unk_id,
            '[SOS]': self.sos_id,
            '[SEP]': self.sep_id,
            '[EOS]': self.eos_id,
            ' ': self.space_id,
            '[BLANK]': self.blank_id
        }
        self.script_spliter = script_spliter
        self.preprocessor = preprocessor

        self.save_path = save_path
        if language.lower() == 'kor':
            self.preprocess_rule = re.compile('[^ ㄱ-ㅣ가-힣]+')
        elif language.lower() == 'eng':
            self.preprocess_rule = re.compile('[^ A-Za-z]+')
        else:
            raise NotImplementedError('Not a supported language. supported: kor, eng')

    def _update_vocab(self, text: str) -> None:
        if text not in self.vocab.keys():
            self.vocab[text] = len(self.vocab)

    def _load_transcripts(self, script_path):
        raise NotImplementedError

    @staticmethod
    def _split(transcript):
        raise NotImplementedError

    def _save_vocab(self) -> None:
        with open(self.save_path, 'w') as file:
            writer = csv.writer(file)
            for key, value in self.vocab.items():
                writer.writerow([key, value])
            file.close()

    def __call__(self, transcript_path: str) -> None:
        transcripts = self._load_transcripts(transcript_path)
        for idx, script in enumerate(tqdm.notebook.tqdm(transcripts)):
            if idx + 1 > len(transcripts):
                pass
            else:
                script = self.script_spliter(script)
                script = self.preprocessor(script)
                script = self.preprocess_rule.sub('', script)
                script = self._split(script)
                self._update_vocab(script)
        self._save_vocab()


#
#
#   This Tokenizer Module was originally made by OpenSpeech Team.
#   However, there are few modification in the below code.
#
#   original code: https://github.com/openspeech-team/openspeech
#
#


class Tokenizer(object):
    def __init__(self, vocab_path: str, add_blank: bool = True, add_special_tokens: bool = True):
        self.vocab, self.ids = self._load_vocab(vocab_path)
        self.add_special_tokens = add_special_tokens
        self.add_blank = add_blank

        self.vocab_path = vocab_path
        self.sos_id = int(self.vocab['[SOS]'])
        self.eos_id = int(self.vocab['[EOS]'])
        self.sep_id = int(self.vocab['[SEP]'])
        self.pad_id = int(self.vocab['[PAD]'])
        self.unk_id = int(self.vocab['[UNK]'])
        self.space_id = int(self.vocab[' '])
        self.blank_id = int(self.vocab['[BLANK]'])
        self.vocab_size = len(self.vocab)

    def __len__(self):
        len(self.vocab)

    @staticmethod
    def _load_vocab(vocab_path: str) -> Tuple[dict, dict]:
        with open(vocab_path, 'r') as file:
            r = csv.reader(file)
            vocab = dict(r)
            file.close()
        ids_vocab = dict(map(reversed, vocab.items()))
        return vocab, ids_vocab

    def _encode(self, sentence: str) -> list:
        label = list()
        if self.add_blank is True:
            label.append(self.blank_id)
        if self.add_special_tokens is True:
            label.append(self.sos_id)
        for jamo in sentence:
            try:
                label.append(int(self.vocab[jamo]))
            except KeyError:
                label.append(self.unk_id)
        if self.add_special_tokens is True:
            label.append(self.eos_id)
        return label

    def _decode(self, labels: torch.Tensor):
        sentence = str()
        for label in labels:
            if label.item() == self.blank_id:
                continue
            else:
                if label.item() == self.eos_id:
                    break
                elif label.item() == self.blank_id:
                    continue
                sentence += self.ids[str(label.item())]
        return sentence

    def _batch_decode(self, batch: torch.Tensor):
        sentence = str()
        for label in batch:
            if label.item() == self.blank_id:
                pass
            else:
                if label.item() == self.eos_id:
                    break
                elif label.item() == self.blank_id:
                    continue
                sentence += self.ids[str(label.item())]
        return sentence

    def encode(self, sentence):
        raise NotImplementedError

    def decode(self, sentence):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError