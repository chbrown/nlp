# import os
from glob import iglob
import ujson as json

count_types = dict(indefinites=['a', 'an'], definites=['the'])

print '\t'.join(['sri', 'indefinites', 'definites', 'wc', 'gpa', 'sat'])
for filepath in iglob('/Users/chbrown/corpora/admissions-json/*.json'):
    # print filepath
    doc = json.load(open(filepath))
    stats = doc['statistics']
    counts = dict(((key, sum([stats['words'].get(word, 0) for word in words])) for key, words in count_types.items()))

    cells = [
        stats['social']['sri'],
        counts['indefinites'],
        counts['definites'],
        doc['summary']['wordcount'],
        stats['social']['overall_fygpa'],
        stats['social']['sat_equiv']
    ]
    print '\t'.join(map(str, cells))
