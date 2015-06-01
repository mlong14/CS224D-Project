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
from sentimentFunctions import SentimentAnnotator

logging.basicConfig(level=logging.INFO)


## splitDataset.py takes in the raw json reviews, and for each sentence,
#    looks for a known aspect word, finds a window around it, calculates the 
#    averaged sentiment of the window, and using that, classifies the window
#    into a positive, negative, or neutral phrase. It outputs them into three files:
#
#	'cnn_input/positivePhrases.txt'
#	'cnn_input/negativePhrases.txt'
#	'cnn_input/neutralPhrases.txt'
#
#

## todo:
# needs to handle bigram
# increase window size
# 
# in discoverAspects, fix the bigram
#    e.g. user_experience
#
## split dataset into
# train_pos, train_neu, train_neg
# dev_pos, dev_neu, dev_neg
# test_pos, test_neu, test_neg

#sentences = SentenceStream('data/small_dataset')
#positivePhrases = ''
#negativePhrases = ''
#neutralPhrases = ''


 
#aspectDictionary = ["videos", "dvd", "dvds"]
#sentences = SentenceStream(os.path.join(settings.DATA_ROOT,'small_dataset'))



def split(windowWidth = 2):
	# window width = 2 on each side!

	aspectDictionary = []
	with open('aspectDictionary.txt','r') as f:
	    #aspectDictionary.append(f.readline())
	    aspectDictionary = [line.rstrip('\n') for line in f]

	# sets up sentiment annotator
	myAnnotator = SentimentAnnotator()
	
	# sets up sentence stream
	sentences = SentenceStream(os.path.join(settings.DATA_ROOT,settings.JSON_FNAME))
	#sentences = SentenceStream(os.path.join(settings.DATA_ROOT,'small_dataset'))
	
	#bigram = gensim.models.phrases.Phrases(sentences)
	bigram = gensim.models.phrases.Phrases.load(os.path.join(settings.MODEL_ROOT,settings.NGRAM_FNAME))


	logging.info("reading in sentences")
	st_time = datetime.now()
	counter = 0
	for sentence in sentences:
		counter+=1
		if counter%10000==0:
			print ("Bigram. WindowWidth: " + str(windowWidth) + ", numSentences: " + str(counter))
		for idx, word in enumerate(sentence):
			# checks if the word is a known aspect
			if word in aspectDictionary:
				# get list of words in window. Makes sure indices don't go out of bounds
				listOfIdx = [(j+idx-windowWidth) for j in xrange(windowWidth*2+1)]
				listOfIdxCleaned = [j for j in listOfIdx if (j >= 0 and j <len(sentence))]

				# collect the phrase with the aspect words in the center
				phrase = [sentence[j] for j in listOfIdxCleaned]

				# annotates the sentiment. does a simple check to classify into pos, neg, or neutral
				sentiments = [myAnnotator.annotate(sentence[j]) for j in listOfIdxCleaned]
				if sentiments.count('pos') > sentiments.count('neg'):
					file_name = 'cnn_input/positivePhrases_bigram_' + str(windowWidth) + '.txt'
				elif sentiments.count('pos') < sentiments.count('neg'):
					file_name = 'cnn_input/negativePhrases_bigram_' + str(windowWidth) + '.txt'
				else:
					file_name = 'cnn_input/neutralPhrases_bigram_' + str(windowWidth) + '.txt'

				# writes the phrase to the corresponding file
				with open(file_name,'a') as f:
					f.write(' '.join(bigram[phrase] ) + "\n")

				#print idx, word
				#print phrase
				#print sentiments
				#print(myAnnotator.annotate('awesome'))
	print ("Final Counter: " + str(counter))
	# Run word2vec
	#model_name = "word2vec_cbow_size300_window10_mincount40_sample1e-3"
	#model = gensim.models.word2vec.Word2Vec(sentences=bigram[sentences], sg=0, size=300, alpha=0.025, window=10, workers = 6, min_count=40, sample=1E-3, negative=10)

	#logging.info("model done: {0}".format(datetime.now()-st_time))

	#model.save(os.path.join(settings.MODEL_ROOT,model_name))
	
	#print model.most_similar(positive=['computer'])
	#print model.most_similar(positive=['screen','resolution'])
	#print model.most_similar(positive=['build','quality'])


def clearOutputFiles(windowWidth = 0):
	file_name = 'cnn_input/positivePhrases_bigram_' + str(windowWidth) + '.txt'
	with open(file_name,'w') as f:
		f.write('')
	file_name = 'cnn_input/negativePhrases_bigram_' + str(windowWidth) + '.txt'
	with open(file_name,'w') as f:
		f.write('')
	file_name = 'cnn_input/neutralPhrases_bigram_' + str(windowWidth) + '.txt'
	with open(file_name,'w') as f:
		f.write('')

def main():
	clearOutputFiles(windowWidth = 5)
	split(windowWidth = 5)


if __name__ == '__main__':
    main()
