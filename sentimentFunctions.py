import settings
import pickle

class SentimentAnnotator():
	def __init__(self):
		self.bingliu_pos = pickle.load(open(settings.SENTIMENT_PATH + "posWordList.p", "r"))
		self.bingliu_neg = pickle.load(open(settings.SENTIMENT_PATH + "negWordList.p", "r"))
		self.anew = pickle.load(open(settings.SENTIMENT_PATH + "anew.p", "r"))

	def annotateDiscrete(self, word):
		if word in self.bingliu_pos:
			return "pos"
		elif word in self.bingliu_neg:
			return "neg"
		else:
			return "neu"

	def annotateContinuous(self, word):
		if word in self.anew:
			return self.anew[word]-4.5
		else:
			return 0

	def annotate(self, word):
		return self.annotateDiscrete(word)

	def annotateSentence(self, sentence):
		return 0
