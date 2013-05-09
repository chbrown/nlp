import subprocess
import random
from copy import copy

def infer(s):
    if s.isdigit():
        return int(s)
    elif s.isalpha():
        return s
    return float(s)


def read_csv(path):
    # headers = []
    rows = []
    with open(path) as csv_fp:
        for line in csv_fp:
            rows.append([infer(cell) for cell in line.strip().split(',')])
    return rows


def gloss(alignments, toksep=' ', linesep='\n', groupsep='\n'):
    '''
    Take a list of [('a', 'DET'), ('beluga', 'N')] and return two lines, such that:
    line1 = 'a   beluga'
    line2 = 'DET N     '
    # pairs can be tuples with length > 1
    '''
    max_width = int(subprocess.check_output('tput cols', shell=True))
    toksep_len = len(toksep)

    cache_languages = [[] for _ in alignments[0]]  # [[e1, e2, e3], [f1, f2, f3], [g1, g2, g3]]
    groups = []
    cache_width = 0
    for aligned in alignments:
        aligned = map(str, aligned)
        length = max(map(len, aligned))
        cache_width += length + toksep_len
        if cache_width >= max_width:
            groups.append(linesep.join(toksep.join(tokens) for tokens in cache_languages))
            cache_languages = [[] for _ in cache_languages]
            cache_width = length
        for i, token in enumerate(aligned):
            cache_languages[i].append(token.ljust(length))

    if len(cache_languages) > 0:
        groups.append(linesep.join(toksep.join(tokens) for tokens in cache_languages))

    return groupsep.join(groups)


def bifurcate(xs, ratio, shuffle=False):
    # takens a list like [b, c, a, m, n] and ratio like 0.6 and returns two lists: [b, c, a], [m, n]
    length = len(xs)
    pivot = int(ratio * length)
    if shuffle:
        xs = copy(xs)
        random.shuffle(xs)

    return (xs[:pivot], xs[pivot:])
