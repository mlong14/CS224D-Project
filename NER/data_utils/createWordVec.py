import gensim
import pdb

if __name__=="__main__":
    vocab_file = "../data/asp/vocab.txt"
    wordVec_file = "../data/asp/wordVectors.txt"

    w2v = gensim.models.Word2Vec.load("../../word2vec_model/word2vec_cbow_size300_window10_mincount40_sample1e-3")

    with open(vocab_file,'w') as fv:
        with open(wordVec_file,'w') as fw:
            for word in w2v.vocab:
                fv.write(str(word)+'\n')
                fw.write(str(w2v[word])[1:-1].replace('\n','').strip()+'\r\n')
