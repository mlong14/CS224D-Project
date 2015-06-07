import gensim
import pdb


def gensim2dict(w2v):
    w2v_dict = defaultdict(float)
    for word in w2v.vocab:
        w2v_dict[word] = w2v[word]
    return w2v_dict

if __name__=="__main__":
    vocab_file = "../data/asp/vocab.txt"
    wordVec_file = "../data/asp/wordVectors.txt"

    w2v = gensim.models.Word2Vec.load("../../word2vec_model/word2vec_cbow_size300_window10_mincount40_sample1e-3")
    pdb.set_trace()
