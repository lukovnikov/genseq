import random
from typing import Iterable

import lorem
import torch
from torch.utils.data import DataLoader

from genseq.vocab import SentenceEncoder


class StringDataset(object):
    def __init__(self, data:str, testdata:str=None, maxlen:int=30, minlen:int=None, sentence_encoder:SentenceEncoder=None,
                 min_freq=2, **kw):
        super(StringDataset, self).__init__(**kw)
        self.data = data
        self.testdata = testdata
        self.maxlen, self.minlen = maxlen, minlen if minlen is not None else maxlen
        self.sentence_encoder = sentence_encoder

        # build vocabulary
        self.sentence_encoder.inc_build_vocab(data, seen=True)
        if testdata is not None:
            self.sentence_encoder.inc_build_vocab(testdata, seen=False)
        self.sentence_encoder.finalize_vocab(min_freq=min_freq)

        self.build_data(data, testdata)

    def build_data(self, data:str, testdata:str):
        self.data_tensor, self.data_tokens = self.sentence_encoder.convert(data, return_what="tensor,tokens")
        if self.testdata is not None:
            self.testdata_tensor, self.testdata_tokens = self.sentence_encoder.convert(testdata, return_what="tensor,tokens")

    def len_of(self, split:str="train"):
        x = self.data_tensor if split == "train" else self.testdata_tensor
        return len(x) - self.maxlen

    def get_item_from(self, i, split:str="train"):
        x = self.data_tensor if split == "train" else self.testdata_tensor
        l = random.random()
        ret = x[i:i+self.maxlen]
        return ret

    def dataloader(self, split:str="train", batsize:int=5):
        dl = DataLoader(DatasetSplitProxy(self, split), batch_size=batsize, shuffle=split=="train", collate_fn=StringDataset.collate_fn)
        return dl

    @staticmethod
    def collate_fn(data:Iterable):
        maxlen = 0
        data = [x.clone() for x in data]
        for x in data:
            maxlen = max(maxlen, x.size(0))
        ret = []
        for x in data:
            x = torch.cat([x, x.new_zeros(maxlen - x.size(0))], 0)
            ret.append(x)
        ret = torch.stack(ret, 0)
        return ret


class DatasetSplitProxy(object):
    def __init__(self, ds, split, **kw):
        super(DatasetSplitProxy, self).__init__(**kw)
        self.ds, self.split = ds, split

    def __getitem__(self, item):
        return self.ds.get_item_from(item, split=self.split)

    def __len__(self):
        return self.ds.len_of(self.split)


def try_string_dataset():
    x = lorem.paragraph()
    se = SentenceEncoder(tokenizer=lambda x: [xe for xe in x])
    d = StringDataset(x, sentence_encoder=se)
    print(x)
    dl = d.dataloader(batsize=5)
    for batch in dl:
        print(se.vocab.print(batch[0]))
        print(batch)
    print(random.random())


if __name__ == '__main__':
    try_string_dataset()