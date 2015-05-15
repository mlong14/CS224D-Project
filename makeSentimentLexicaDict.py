import pickle

posWords = [line.rstrip() for line in open('sentiment/opinion-lexicon/bingliu_pos.txt')]
negWords = [line.rstrip() for line in open('sentiment/opinion-lexicon/bingliu_neg.txt')]

pickle.dump(posWords, open("sentiment/posWordList.p", "wb"))
pickle.dump(negWords, open("sentiment/negWordList.p", "wb"))


f = open('sentiment/opinion-lexicon/anew_1999.csv', 'r')
allStrings = f.readline()
anew_dict = {} # dictionary that just links word to mean valence
for line in allStrings.split():
	thisLine = line.split(",")
	anew_dict[thisLine[0]] = thisLine[1]
f.close()

pickle.dump(anew_dict, open("sentiment/anew.p", "wb"))
