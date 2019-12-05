from abc import ABC, abstractmethod
from typing import List, Union, Dict, Callable

import torch
import numpy as np


class _Vocab(object):
    pass


class Vocab(_Vocab):
    padtoken = "@PAD@"
    unktoken = "@UNK@"
    starttoken = "@START@"
    endtoken = "@END@"

    def __init__(self, padid: int = 0, unkid: int = 1, startid: int = 2, endid: int = 3, **kw):
        self.D = {self.padtoken: padid, self.unktoken: unkid}
        self.D[self.starttoken] = startid
        self.D[self.endtoken] = endid
        self.counts = {k: np.infty for k in self.D.keys()}
        self.rare_tokens = set()
        self.rare_ids = set()
        self.RD = {v: k for k, v in self.D.items()}
        self.growing = True

    def nextid(self):
        return max(self.D.values()) + 1

    def stopgrowth(self):
        self.growing = False

    def do_rare(self, min_freq: int = 0, top_k: int = np.infty):
        tokens_with_counts = self.counts.items()
        if min_freq == 0 and top_k > len(tokens_with_counts):
            self.rare_tokens = set()
            self.rare_ids = set()
            return

        tokens_with_counts = sorted(tokens_with_counts, key=lambda x: x[1], reverse=True)
        if top_k < len(tokens_with_counts) and tokens_with_counts[top_k][1] > min_freq:
            i = top_k
        else:
            if top_k < len(tokens_with_counts):
                tokens_with_counts = tokens_with_counts[:top_k]
            # binary search for min_freq position
            i = 0
            divider = 2
            where = +1
            while True:
                i += where * len(tokens_with_counts) // divider
                if (i == len(tokens_with_counts)) or (tokens_with_counts[i][1] <= min_freq - 1 and tokens_with_counts[i - 1][1] >= min_freq):
                    break  # found
                elif tokens_with_counts[i][1] < min_freq:  # go up
                    where = -1
                elif tokens_with_counts[i][1] >= min_freq:  # go down
                    where = +1
                divider *= 2
                divider = min(divider, len(tokens_with_counts))
        nonrare = set([t[0] for t in tokens_with_counts[:i]])
        self.rare_tokens = set(self.D.keys()) - nonrare
        self.rare_ids = set([self[rare_token] for rare_token in self.rare_tokens])

    def add_token(self, token, seen: Union[int, bool] = True):
        if token not in self.D:
            assert (self.growing)
            if self.growing:
                id = self.nextid()
                self.D[token] = id
                self.RD[id] = token
                self.counts[token] = 0
        if seen > 0:
            self.counts[token] += float(seen)

    def __getitem__(self, item: str) -> int:
        if item not in self.D:
            assert (self.unktoken in self.D)
            item = self.unktoken
        id = self.D[item]
        return id

    def __call__(self, item: int) -> str:
        return self.RD[item]

    def number_of_ids(self):
        return max(self.D.values()) + 1

    def reverse(self):
        return {v: k for k, v in self.D.items()}

    def __iter__(self):
        return iter([(k, v) for k, v in self.D.items()])

    def __contains__(self, item: Union[str, int]):
        if isinstance(item, str):
            return item in self.D
        if isinstance(item, int):
            return item in self.RD
        else:
            raise Exception("illegal argument")

    def print(self, x: Union[np.ndarray, torch.Tensor]):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = list(np.vectorize(lambda e: self(e))(x))
        x = [e for e in x if e != self.padtoken]
        return " ".join(list(x))


class FixedVocab(Vocab):
    def __init__(self, padid: int = 0, unkid: int = 1, vocab: Dict = None, **kw):
        super(FixedVocab, self).__init__(padid, unkid, **kw)
        self.D = vocab
        self.growing = False

    def add_token(self, token, seen=True):
        print("Warning: trying to add token to fixed vocab")
        pass

    def do_rare(self, min_freq=0, top_k=np.infty):
        print("Warning: trying to do rare on fixed vocab")
        pass


def try_vocab():
    vocab = Vocab()
    tokens = "a b c d e a b c d a b c a b a a a a b e d g m o i p p x x i i b b ai ai bi bi bb bb abc abg abh abf".split()
    for t in tokens:
        vocab.add_token(t)
    vocab.do_rare(min_freq=2, top_k=15)
    print(vocab.rare_tokens)
    print(vocab.rare_ids)


class VocabBuilder(ABC):
    @abstractmethod
    def inc_build_vocab(self, x: str, seen: bool = True):
        raise NotImplemented()

    @abstractmethod
    def finalize_vocab(self, min_freq: int = 0, top_k: int = np.infty):
        raise NotImplemented()

    @abstractmethod
    def vocabs_finalized(self):
        raise NotImplemented()


class SentenceEncoder(VocabBuilder):
    endtoken = "@END@"

    def __init__(self, tokenizer: Callable[[str], List[str]], vocab: Vocab = None, add_end_token=False, **kw):
        super(SentenceEncoder, self).__init__(**kw)
        self.tokenizer = tokenizer
        self.vocab = vocab if vocab is not None else Vocab()
        self.vocab_final = False
        self.add_end_token = add_end_token

    def inc_build_vocab(self, x: str, seen: bool = True):
        if not self.vocab_final:
            tokens = self.tokenizer(x)
            if self.add_end_token:
                tokens.append(self.endtoken)
            for token in tokens:
                self.vocab.add_token(token, seen=seen)

    def finalize_vocab(self, min_freq: int = 0, top_k: int = np.infty):
        self.vocab_final = True
        self.vocab.stopgrowth()
        self.vocab.do_rare(min_freq=min_freq, top_k=top_k)

    def vocabs_finalized(self):
        return self.vocab_final

    def convert(self, x: str, return_what="tensor"):  # "tensor", "ids", "tokens" or comma-separated combo of all
        rets = [r.strip() for r in return_what.split(",")]
        tokens = self.tokenizer(x)
        if self.add_end_token:
            tokens.append(self.endtoken)
        ids = [self.vocab[token] for token in tokens]
        tensor = torch.tensor(ids, dtype=torch.long)
        ret = {"tokens": tokens, "ids": ids, "tensor": tensor}
        ret = [ret[r] for r in rets]
        return ret


if __name__ == '__main__':
    try_vocab()