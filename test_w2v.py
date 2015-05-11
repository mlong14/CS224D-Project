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

def testW2V():
	model = gensim.models.word2vec.Word2Vec.load(os.path.join(settings.MODEL_ROOT,'word2vec2'))

	print model.most_similar(positive=['tech','support'])
	print model.most_similar(positive=['macbook'])
	print model.most_similar(positive=['quality'])
def main():
	testW2V()

if __name__ == '__main__':
    main()
