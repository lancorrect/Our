from supar import Parser
import json

parser = Parser.load('biaffine-dep-en')

with open('./dataset/Tweets_stanza/test.json') as fout:
    stanza = json.load(fout)
    fout.close()

biaffine = []
for elem in stanza:
    data_pre = parser.predict(elem['token'], verbose=False)
    elem['head'] = data_pre.arcs[0]
    elem['deprel'] = data_pre.rels[0]
    biaffine.append(elem)

print(biaffine[0])

with open('./dataset/Tweets_test.json', 'w') as fin:
    json.dump(biaffine, fin)
    fin.close()
