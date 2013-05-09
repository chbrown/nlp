import json

headers = ["total_articles_count", "testing_articles_count", "feature_functions", "training_articles_count", "total_token_count", 'DEF->INDEF', 'DEF->NA', 'INDEF->DEF', 'INDEF->NA', 'NA->DEF', 'NA->INDEF']
print '\t'.join(headers)
for line in open('crf-conditions.json'):
    d = json.loads(line)
    d.update(d['confusion_matrix'])
    print '\t'.join(str(d.get(key, 0)) for key in headers)
