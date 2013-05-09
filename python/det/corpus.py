# import os
# import numpy as np
# from scipy import optimize
# from termcolor import cprint
# from sklearn import hmm

from nltk.tokenize import word_tokenize, sent_tokenize  # word_tokenize
from collections import defaultdict  # Counter,


# import re
import itertools


def tokenize_sentences(sentences):
    for sentence in sent_tokenize(sentence):
        for token in word_tokenize(sentence):
            yield token


# def tokenize(raw):
    # , flags=re.I
    # assume that raw is one sentence (should be the case with Brown corpus)
    # return (raw)
    # return re.split("[^'a-z0-9A-Z]+", raw)  # .strip()

# def read_wsj(regex='00/wsj_00.+mrg'):
#     # ./00/wsj_0001.pos
#     wsj_root = '/Users/chbrown/Dropbox/ut/nlp/data/penn-treebank3/parsed/mrg/wsj'
#     reader = BracketParseCorpusReader(wsj_root, regex)
#     for sent in wsj.tagged_sents():
#         yield sent


def read_brown(max_lines):
    # corpus is simply a list of sentences
    brown_path = '/Users/chbrown/corpora/browncorpus.txt'
    return map(DefinitenessDocument, itertools.islice(open(brown_path), max_lines))


# print 'Corpus'
# print '\n'.join('%s\n' % doc for doc in corpus)
