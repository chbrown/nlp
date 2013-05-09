#!/usr/bin/env python

import crfsuite
import sys
import argparse
from termcolor import colored  # cprint, RESET
from munging import gloss


def start_color(name):
    return colored('', name)[:-4]
# [(start_color('white'), start_color('yellow'), '')] +
# [(RESET, RESET, '')])


def group_by_newline(lines):
    """
    this will take in a generator
      (a, b, \n, code, dir, exec, \n, ...)
    and return
      ([a, b], [code, dir, exec], [...])
    """
    cache = []
    for line in lines:
        if line.strip():
            cache.append(line)
        else:
            yield cache
            cache = []
    yield cache


def read_svm_format(lines):
    # reads lines like:
    # Y 15:0.4 16:0.01 19:3.4
    # or
    # X 4 9 23
    # and iterates over sentences (which are separated by a whitespace-only line).
    # it yield pairs like:
    #   (crfsuite.ItemSequence([
    #      crfsuite.Item([crfsuite.Attribute("15"->0.4), crfsuite.Attribute("16"->0.01), crfsuite.Attribute("19"->3.4)]),
    #      crfsuite.Item([crfsuite.Attribute("4"), crfsuite.Attribute("9"), crfsuite.Attribute("23")])
    #    ]), ("X", "Y", ...))
    # crfsuite.Attribute has 2 properties: attr, and value
    for sentence_lines in group_by_newline(lines):
        data = crfsuite.ItemSequence()
        labels = crfsuite.StringList()
        for line in sentence_lines:
            # Split the line with TAB characters.
            # print '>>>', line
            cells = line.strip().split(' ')
            datum = crfsuite.Item()
            for data_field in cells[1:]:
                # don't split if the whole field is a literal colon
                parts = data_field.rsplit(':', 1) if data_field != ':' else data_field
                if len(parts) > 1:
                    # we read the optional weight:
                    datum.append(crfsuite.Attribute(parts[0], float(parts[1])))
                else:
                    # otherwise, weight = 1 by default
                    datum.append(crfsuite.Attribute(parts[0]))

            # Append the item to the item sequence.
            data.append(datum)
            # Append the label to the label sequence.
            labels.append(cells[0])
            # empty line is document boundary
        yield (data, tuple(labels))


class Trainer(crfsuite.Trainer):
    """
    Inherit crfsuite.Trainer to implement message() function, which receives
    progress messages from a training process.
    """
    def message(self, s):
        return
        sys.stdout.write(s)

    def append_raw(self, features_seq, labels):
        # len(labels) = len(data) = length of sentence / sequence
        # labels is a tuple of strings, data is an tuple/list of lists of strings.
        # this just wraps all the data / labels with crfsuite types
        items = crfsuite.ItemSequence()
        for features in features_seq:
            item = crfsuite.Item()
            for feature in features:
                if isinstance(feature, tuple):
                    attribute = crfsuite.Attribute(*feature)
                else:
                    attribute = crfsuite.Attribute(feature)
                item.append(attribute)
            items.append(item)

        # labels = crfsuite.StringList(labels)
        self.append(items, tuple(labels), 0)

    def save(self, model_path):
        # Use L2-regularized SGD and 1st-order dyad features.
        self.select('l2sgd', 'crf1d')

        # This demonstrates how to list parameters and obtain their values.

        # Set the coefficient for L2 regularization to 0.1
        self.set('c2', '0.1')

        # Start training; the training process will invoke trainer.message()
        # to report the progress.
        self.train(model_path, -1)

        # print 'After training: params and their values'
        # for name in trainer.params():
        #     print name, trainer.get(name), trainer.help(name)


class Tagger(crfsuite.Tagger):
    def __init__(self, model_path):
        super(Tagger, self).__init__()
        self.open(model_path)

    def tag(self, data):
        # Obtain the label sequence predicted by the tagger.
        self.set(data)
        return self.viterbi()

    def tag_raw(self, data):
        # data is a list of lists, which may very well be just 1-long
        # data = [['The'], ['man'], ['barked']]
        # The sublists maybe contain tuples (of string->float pairs)
        # data = [['The', ('first', 1)], ['man', 'human', ('first', 0)], ...]
        items = crfsuite.ItemSequence()
        for datum in data:
            item = crfsuite.Item()
            for feature in datum:
                if isinstance(feature, tuple):
                    item.append(crfsuite.Attribute(*feature))
                else:
                    item.append(crfsuite.Attribute(feature))
            items.append(item)

        return self.tag(items)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CRFSuite using Python.')
    parser.add_argument('--train')
    parser.add_argument('--tag')
    parser.add_argument('--model', default='crf.model')
    opts = parser.parse_args()

    # This demonstrates how to obtain the version string of CRFsuite.
    print 'CRFSuite v%s' % crfsuite.version()

    if opts.train:
        # Create a Trainer object.
        trainer = Trainer()

        # Read training instances from STDIN, and set them to trainer.
        with open(opts.train) as lines:
            for data, labels in read_svm_format(lines):
                trainer.append(data, labels, 0)

        trainer.save(opts.model)
    else:
        tagger = Tagger(opts.model)
        with open(opts.tag) as lines:
            for data, gold_labels in read_svm_format(lines):
                predicted_labels = tagger.tag(data)
                tokens = [item[0].attr for item in data]
                print gloss(zip(tokens, predicted_labels, gold_labels))

                # total_probability = tagger.probability(predicted_labels)
                # marginals = [tagger.marginal(label, i) for i, label in enumerate(predicted_labels)]
                # Output the predicted labels with their marginal probabilities.
                # print '%s:%d-%f' % (y, )
