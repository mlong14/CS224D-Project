import os
import logging
import re
import gzip
import simplejson as json
import settings
import gensim
import numpy as np
from datetime import datetime
from utils import SentenceStream

logging.basicConfig(level=logging.INFO)

def trainW2V():
	sentences = SentenceStream(os.path.join(settings.DATA_ROOT,settings.JSON_FNAME))
	
        if os.path.isfile(os.path.join(settings.MODEL_ROOT,settings.NGRAM_FNAME)) is False:
		logging.info("collecting n-grams")
		bigram = gensim.models.phrases.Phrases(sentences)
		bigram.save(os.path.join(settings.MODEL_ROOT,settings.NGRAM_FNAME))
	else:
		logging.info("loading n-grams")
		bigram = gensim.models.phrases.Phrases.load(os.path.join(settings.MODEL_ROOT,settings.NGRAM_FNAME))

	logging.info("training w2v")
	st_time = datetime.now()

	# Run word2vec
	model_name = "word2vec_cbow_size300_window10_mincount40_sample1e-3"
	model = gensim.models.word2vec.Word2Vec(sentences=bigram[sentences], sg=0, size=300, alpha=0.025, window=10, workers = 6, min_count=40, sample=1E-3, negative=10)

	logging.info("model done: {0}".format(datetime.now()-st_time))

	model.save(os.path.join(settings.MODEL_ROOT,model_name))
	
	print model.most_similar(positive=['computer'])
	print model.most_similar(positive=['screen','resolution'])
	print model.most_similar(positive=['build','quality'])



def main():
	trainW2V()


if __name__ == '__main__':
    main()
