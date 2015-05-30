import os
import logging
import re
import gzip
import simplejson as json

import settings
from datetime import datetime

from utils import read_file

import gensim
import sys
import numpy as np

logging.basicConfig(level=logging.INFO)

def discAspects():
    model = gensim.models.word2vec.Word2Vec.load(os.path.join(settings.MODEL_ROOT,'word2vec_cbow_size300_window10_mincount40_sample1e-3'))

    # while (1):
    #     print "Enter query"
    #     query = raw_input()
    #     try:
    #         print model.most_similar(positive=query.split())
    #     except (KeyError):
    #         logging.info("word not found")
    #         pass
    #print model.most_similar(positive=['build','quality'])
    #print model.most_similar(positive=['screen','quality'])
    #print model.most_similar(positive=['macbook'])
    #print model.most_similar(positive=['quality'])
def main():
    discAspects()

if __name__ == '__main__':
    main()
