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

def discAspects(inputFile):
    model = gensim.models.word2vec.Word2Vec.load(os.path.join(settings.MODEL_ROOT,'word2vec_cbow_size300_window10_mincount40_sample1e-3'))

    # Read input file
    with open(inputFile) as f:
        for line in f:
            try:
                print line.split()
                print model.most_similar(positive=line.split())
            except (KeyError):
                print("word %s not found"%(line.split()))
                pass
def main():
    discAspects(sys.argv[1])

if __name__ == '__main__':
    main()
