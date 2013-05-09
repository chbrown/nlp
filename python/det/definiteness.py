import os
import nltk
# from nltk.tokenize import word_tokenize  # word_tokenize
from termcolor import cprint
from nltk.tag import HiddenMarkovModelTagger
import argparse
from munging import gloss, bifurcate
from collections import Counter, defaultdict
from nltk.corpus import BracketParseCorpusReader
import crf
import json
# from scikits.learn import svm

from tsa.lsi import Dictionary
import svmlight


def definiteness_transform_tokens(tokens):
    for token in tokens:
        if token == 'the':
            yield 'DET', 'DEF'
        elif token == 'a' or token == 'an':
            yield 'DET', 'INDEF'
        else:
            yield token, 'NA'


class Document(object):
    def __init__(self, tokens=None, pos_tags=None):
        self.tokens = tokens or []
        self.pos_tags = pos_tags or []


class DefinitenessDocument(Document):
    # tokens = he told DET committee DET measure would merely provide means
    # labels = NA NA   DEF NA        DEF NA      NA    NA     NA      NA
    def __init__(self, tokens=None, pos_tags=None):
        super(DefinitenessDocument, self).__init__(tokens, pos_tags)

        def_token_tags = definiteness_transform_tokens(self.tokens)
        self.def_tokens, self.def_tags = zip(*def_token_tags)

    @classmethod
    def from_token_postag_pairs(cls, token_postag_pairs):
        tokens, pos_tags = zip(*token_postag_pairs)
        return cls(tokens, pos_tags)

    def token_pos_tag_pairs(self):
        return zip(self.tokens, self.pos_tags)

    def def_token_tag_pairs(self):
        return zip(self.def_tokens, self.def_tags)

    def __str__(self):
        # spaced_tokens, spaced_labels = spaced_tokens + '\n' + spaced_labels
        return gloss(self.token_pos_tag_pairs())

    def __len__(self):
        return len(self.tokens)

    def zipped(self):
        return zip(self.tokens, self.pos_tags, self.def_tokens, self.def_tags)

    __repr__ = __str__


def count_attrs(xs, attr):
    counts = dict()
    for x in xs:
        for value in getattr(x, attr):
            counts[value] = counts.get(value, 0) + 1
    return counts


def write_svm(data, filepath):
    with open(filepath, 'w') as fp:
        for label, feature_values in data:
            fp.write('%d %s\n' % (label, ' '.join('%d:%d' % feature_value for feature_value in feature_values)))


def read_wsj(article_count):
    wsj_root = '/Users/chbrown/Dropbox/ut/nlp/data/penn-treebank3/parsed/mrg/wsj'
    articles = []
    for section in range(25):
        for article_path in os.listdir('%s/%02d' % (wsj_root, section)):
            reader = BracketParseCorpusReader(wsj_root, '%02d/%s' % (section, article_path))

            sentences = []
            for tagged_sent in reader.tagged_sents():
                # token_postag_pairs = sentence
                token_postag_pairs = [
                    (token.lower(), pos_tag)
                    for token, pos_tag in tagged_sent
                    if pos_tag not in ('-LRB-', '-RRB-', '-NONE-')]
                sentence = DefinitenessDocument.from_token_postag_pairs(token_postag_pairs)
                sentences.append(sentence)

            articles.append(sentences)

            if len(articles) >= article_count:
                return articles
    return articles


# each feature_function goes from a sentence to a list of lists of features to add to the total data for that sentence:
def ff_def_token(sentence, seen_nouns):
    return [[def_token] for def_token in sentence.def_tokens]


def ff_next_def_token(sentence, seen_nouns):
    # '''ff_next_def_token'''
    return [[def_token] for def_token in sentence.def_tokens[1:]] + [[]]


def ff_next_noun(sentence, seen_nouns):
    # the token of the next thing tagged noun phrase
    for i in range(len(sentence)):
        next_noun = None
        for j in range(i + 1, len(sentence)):
            # match NN, NNS, NNP, NNPS
            if sentence.pos_tags[j].startswith('NN'):
                next_noun = sentence.def_tokens[j]
                break
        yield [next_noun] if next_noun else []


def ff_next_noun_seen(sentence, seen_nouns):
    # this function has to be the one to update the seen nouns set
    # it's a side effect. sorry.
    for next_noun, token, pos_tag in zip(ff_next_noun(sentence, seen_nouns), sentence.tokens, sentence.pos_tags):
        if len(next_noun) and next_noun[0] in seen_nouns:
            yield ['-SEEN-']
        else:
            yield ['-NEW-']
        if pos_tag.startswith('NN'):
            seen_nouns.add(token)


def ff_def_na(sentence, seen_nouns):
    # for i, (token, pos_tag, def_token, def_tag) in enumerate(sentence.zipped()):
    for def_tag in sentence.def_tags:
        yield ['-DEF-NA-'] if def_tag == 'NA' else []


def preprocess_wsj(article_count, feature_functions):
    '''
    Returns (articles, total_token_count)
    '''
    articles = read_wsj(article_count)
    # articles is now a list of lists of documents, which are actually sentences
    # articles = [
    #     [Doc(1a), Doc(1b)],           # WSJ article 0001
    #     [Doc(2a), Doc(2b), Doc(2c)],  # WSJ article 0002
    # ]
    # all those Doc's are DefinitenessDocument's
    total_tokens = []
    for article in articles:
        seen_nouns = set()
        for sentence in article:
            total_tokens += sentence.tokens
            feature_lists = [list(ff(sentence, seen_nouns)) for ff in feature_functions]
            sentence.data = [sum(features, []) for features in zip(*feature_lists)]
            # for i, token in enumerate(sentence.tokens):
            # print gloss(zip(sentence.tokens, sentence.data))
    return articles, len(total_tokens)


def run_crf(article_count, feature_functions, split=0.9, model_path='crf.model'):
    '''
    Returns:
        total_articles_count (int)
        total_token_count (int)
        training_articles_count (int)
        testing_articles_count (int)
        confusion_matrix (dict)

    '''
    articles, total_token_count = preprocess_wsj(article_count, feature_functions)

    train, test = bifurcate(articles, split, shuffle=True)
    # print 'Training on %d articles, testing on %d.' % (len(train), len(test))

    # take = 0
    # print 'First %d articles:' % take
    # for article in train[:take]:
    #     print '---'
    #     for sentence in article:
    #         print gloss(zip(sentence.def_tokens, sentence.def_tags, sentence.data))
    #         print

    trainer = crf.Trainer()
    for article in train:
        for sentence in article:
            trainer.append_raw(sentence.data, sentence.def_tags)

    trainer.save(model_path)
    tagger = crf.Tagger(model_path)

    # results = defaultdict(list)
    confusion_matrix = defaultdict(int)
    for article in test:
        for sentence in article:
            predicted_tags = tagger.tag_raw(sentence.data)
            for token, gold, predicted in zip(sentence.tokens, sentence.def_tags, predicted_tags):
                key = (gold, predicted)
                # results[key] += [token]
                if gold != predicted:
                    confusion_matrix['%s->%s' % key] += 1

    # print 'Results'
    # for (predicted_label, gold_label), tokens in results.items():
        # color = 'green' if predicted_label == gold_label else 'red'
        # cprint('%5d predicted=%s -> gold=%s' % (len(tokens), predicted_label, gold_label), color)
        # print '  ', Counter(tokens).most_common(20)

    # documents = read_brown(opts.docs)
    # for sentence in read_wsj():
    # print 'Total corpus labels:', count_labels(documents)
    # print 'POS Tags', count_attrs(documents, 'pos_tags')
    # print 'POS Tags', count_attrs(documents, 'pos_tags')

    return dict(
        total_articles_count=len(articles),
        total_token_count=total_token_count,
        training_articles_count=len(train),
        testing_articles_count=len(test),
        confusion_matrix=confusion_matrix
    )


def do_svm(article_count, split):
    # https://bitbucket.org/wcauchois/pysvmlight
    articles = preprocess_wsj(article_count)

    dictionary = Dictionary()
    dictionary.add_one('ZZZZZ')  # so that no features are labeled 0
    data = []
    for article in articles:
        for sentence in article:
            for tag, token_features in zip(sentence.def_tags, sentence.data):
                # only use def / indef tokens
                if tag in ('DEF', 'INDEF'):
                    features = dictionary.add(token_features)
                    features = sorted(list(set(features)))
                    feature_values = zip(features, [1]*len(features))
                    data.append((+1 if tag == 'DEF' else -1, feature_values))

    train, test = bifurcate(data, split)

    print 'Training on %d determiners, testing on %d.' % (len(train), len(test))

    # for corpus, name in [(train, 'train'), (test, 'test')]:
        # write_svm(corpus, 'wsj_svm-%s.data' % name)

    #####################
    # do svm in Python...
    kernel = 'polynomial'
    model = svmlight.learn(train, type='classification', kernel=kernel)

    # svmlight.learn options
    # type: select between 'classification', 'regression', 'ranking' (preference ranking), and 'optimization'.
    # kernel: select between 'linear', 'polynomial', 'rbf', and 'sigmoid'.
    # verbosity: set the verbosity level (default 0).
    # C: trade-off between training error and margin.
    # poly_degree: parameter d in polynomial kernel.
    # rbf_gamma: parameter gamma in rbf kernel.
    # coef_lin
    # coef_const
    # costratio (corresponds to -j option to svm_learn)
    svmlight.write_model(model, 'wsjmodel.svm')

    gold_labels, test_feature_values = zip(*test)
    # print 'gold_labels', gold_labels
    # print 'test_feature_values', test_feature_values

    test_pairs = [(0, feature_values) for feature_values in test_feature_values]
    predicted_labels = svmlight.classify(model, test_pairs)
    # print 'len(predicted_labels), len(gold_labels)', len(predicted_labels), len(gold_labels)
    print 'gold_labels', Counter(gold_labels)

    correct = len([1 for predicted, gold in zip(predicted_labels, gold_labels) if (predicted > 0) == (gold == 1)])
    wrong = len(gold_labels) - correct
    # print 'predicted', 'gold'
    # for predicted, gold in zip(predicted_labels, gold_labels):
        # print '%.8f (%d)' % (predicted, gold)

    print 'Correct: %d, Wrong: %d' % (correct, wrong)



# successes = 0
# errors = defaultdict(list)
def do_hmm(documents, split):
    train, test = bifurcate(documents, split)

    # train does NOT accept generators
    tagger = HiddenMarkovModelTagger.train([doc.token_label_pairs() for doc in train])
    results = defaultdict(list)
    for doc in test:
        predicted = tagger.tag(doc.tokens)
        gold_labels = doc.labels
        # precision =
        # recall =

        token_tag_pairs = nltk.pos_tag(doc.literal)
        print '\n-----------\n' + gloss(token_tag_pairs)

        for (token, predicted_label), gold_label in zip(predicted, gold_labels):
            results[(predicted_label, gold_label)] += [token]
            # if predicted_label == gold_label:
            #     successes += 1
            # else:
            #     errors[(predicted_label, gold_label)] += [token]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect definites.')
    parser.add_argument('--articles', type=int, default=1000, help='Number of articles to read from corpus')
    parser.add_argument('--model_path', default='definiteness.model')
    parser.add_argument('--split', type=float, default=.9, help='Train/test split')
    opts = parser.parse_args()

    # do_crf(opts.articles, opts.model, opts.split)
    # do_svm(opts.articles, opts.split)
    article_counts = [100, 1000, 10000]
    feature_function_selections = [
        # (name, feature_function_list), pairs
        ('ff_def_na', [ff_def_na]),
        ('ff_def_token', [ff_def_na, ff_def_token]),
        ('ff_next_def_token', [ff_def_na, ff_def_token, ff_next_def_token]),
        ('ff_next_noun', [ff_def_na, ff_def_token, ff_next_noun]),
        ('ff_next_noun_with_seen', [ff_def_na, ff_def_token, ff_next_noun, ff_next_noun_seen]),
        ('full', [ff_def_na, ff_def_token, ff_next_def_token, ff_next_noun, ff_next_noun_seen]),
    ]

    for article_count in article_counts:
        for ff_label, feature_functions in feature_function_selections:
            result = run_crf(article_count, feature_functions, split=opts.split, model_path=opts.model_path)
            result['feature_functions'] = ff_label
            print json.dumps(result)



# each thing in errors has the predicted label first (what our tagger thought it should be) and then what the actual label was.
# the tokens it made this error on are recorded.

# hmm_model = hmm.MultinomialHMM(n_components=3)
# hmm_model.fit([X])
# Z2 = model2.predict(X)
