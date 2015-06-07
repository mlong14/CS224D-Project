import os
import simplejson as json
import settings
import gensim
from utils import SentenceStream
import numpy as np

#person = "desmond"
#valence = "negative"
#inputFile = person + "_" + valence + "_bigram_5_window_5_1000"


def generateLabeledExamplesForNER(lineCount, desiredNumber = 20000):
	indices = np.random.choice(lineCount, size=desiredNumber, replace=False)
	with open("data/NER_input/full_phrases_indices",'w+') as fOut:
		for item in indices:
			fOut.write(str(item) + ' ')

	aspectDictionary = []
	with open('aspectDictionary.txt','r') as f:
	    #aspectDictionary.append(f.readline())
	    aspectDictionary = [line.rstrip('\n') for line in f]

	sentences = SentenceStream(os.path.join(settings.DATA_ROOT,settings.JSON_FNAME))
	bigram = gensim.models.phrases.Phrases.load(os.path.join(settings.MODEL_ROOT,settings.NGRAM_FNAME))

	counter = 0
	testCounter = 0
	#with open(inputFile, "r") as fIn:
	for line in sentences:
		if counter in indices:
			if((testCounter%10)<7):
				outputFile = "data/NER_input/train_set"
			elif((testCounter%10)<9):
				outputFile = "data/NER_input/dev_set"
			else:
				outputFile = "data/NER_input/test_set"
				print("Processed: " + str(testCounter))
			testCounter+=1
			with open(outputFile, "a+") as fOut:
				#words = line.split()
				words = line
				outLine = ""
				for word in words:
					if word in aspectDictionary:
						outLine += (str(word) + '\tASP\n')
					else:
						outLine += (str(word) + '\tO\n')
				fOut.write(outLine)
				fOut.write("\n")
		counter+=1



def main():
	#inputFile = "../cnn_input/allPhrases_bigram_top5_window_5.txt"
	#inputFile = "../cnn_input/allPhrases_bigram_top5_window_5.txt"
	#generateLabeledExamplesForNER(2000000, 20000)

	generateLabeledExamplesForNER(10, 1)
	

if __name__ == '__main__':
    main()

