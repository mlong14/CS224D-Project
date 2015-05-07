import os
import logging
import re
import gzip
import simplejson as json

import settings
from datetime import datetime

from utils import read_file

import gensim

import numpy as np

logging.basicConfig(level=logging.INFO)




def trainW2V():

	sentences = read_file()

	logging.info("collecting n-grams")
	bigram = gensim.models.phrases.Phrases(sentences)
	trigram = gensim.models.phrases.Phrases(bigram[sentences])

	bigram.save(os.path.join(settings.MODEL_ROOT,'bigrams2'))
	trigram.save(os.path.join(settings.MODEL_ROOT,'trigrams2'))

	logging.info("training w2v")
	st_time = datetime.now()

	model = gensim.models.word2vec.Word2Vec(sentences=trigram[bigram[sentences]], size=100, alpha=0.025, window=5, min_count=5, sample=1E-5, negative=10)

	logging.info("model done: {0}".format(datetime.now()-st_time))

	model.save(os.path.join(settings.MODEL_ROOT,'word2vec2'))

	print model.most_similar(positive=['computer'])
	print model.most_similar(positive=['screen','resolution'])
	print model.most_similar(positive=['build','quality'])



def main():
	trainW2V()


if __name__ == '__main__':
    main()