import lorem

from genseq.data import StringDataset
from genseq.vocab import SentenceEncoder
import nltk


def build_data(p="../../datasets/TODO", maxlen:int=50, batsize=10):
    if p is None:
        emma = nltk.corpus.gutenberg.raw('austen-emma.txt')
        # print(emma)
        print(len(emma))
        splitposition = int(len(emma)*.8)
        x = emma[:splitposition]
        xtest = emma[splitposition:]
    else:
        pass    # TODO
    se = SentenceEncoder(tokenizer=lambda x: [xe for xe in x])
    ds = StringDataset(x, xtest, maxlen=maxlen, sentence_encoder=se)
    return ds, ds.dataloader("train", batsize=batsize), ds.dataloader("test", batsize=batsize)






def try_build_data():
    ds, traindl, testdl = build_data(p=None)
    print(len(traindl))
    print(len(testdl))
    # for x in traindl:
        # print(x.size())
        # for xe in x:
        #     print(ds.sentence_encoder.vocab.print(xe))


if __name__ == '__main__':
    try_build_data()