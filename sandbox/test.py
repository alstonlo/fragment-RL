from src.data.vocab import FragmentVocab

vocab = FragmentVocab.load_from_pkl("../data/vocab.pkl")
vocab.cull(1000)
for i in range(len(vocab)):
    print(vocab[i].smiles, vocab[i].count)
print(len(vocab))
