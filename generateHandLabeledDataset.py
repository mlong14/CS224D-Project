import numpy as np

def selectExamplesToHandlabel(inputFile, outFile, lineCount, desiredNumber = 1000):
	indices = np.random.choice(lineCount, size=desiredNumber, replace=False)
	with open(outFile + "_indices",'w') as fOut:
		for item in indices:
			fOut.write(str(item) + ' ')
	#with open(outFile,'w') as f:
	#	f.write('')

	aspectDictionary = []
	with open('aspectDictionary.txt','r') as f:
	    #aspectDictionary.append(f.readline())
	    aspectDictionary = [line.rstrip('\n') for line in f]

	counter = 0

	with open(inputFile, "r") as f:
		for line in f:
			if counter in indices:
				sentence = line.rstrip('\n')
				wordList = sentence.split()
				aspectIdx = 0
				currentIdx = 0
				for word in wordList:
					# checks if the word is a known aspect
					if word in aspectDictionary:
						aspectIdx = currentIdx
					elif '_' in word:
						#bigram!
						twograms = word.split('_')
						if twograms[0] in aspectDictionary:
							aspectIdx = currentIdx
						elif twograms[0] in aspectDictionary:
							aspectIdx = currentIdx+1
						currentIdx += 1
					currentIdx += 1
				
				# pad and stuff
				if aspectIdx < 5:
					sentence = '<s> ' + sentence
					aspectIdx += 1
					currentIdx += 1
				while aspectIdx < 5:
					sentence = '000 ' + sentence
					aspectIdx += 1
					currentIdx += 1
				if currentIdx < 11:
					sentence = sentence + ' <\s>'
					currentIdx += 1
				while currentIdx < 11:
					sentence = sentence + ' 000'
					currentIdx += 1
				#print(sentence)
				#print(aspectIdx, currentIdx)

				with open(outFile,'a') as fOut:
					fOut.write(sentence + "\n")
			counter+=1








def main():
	#selectExamplesToHandlabel('cnn_input/toy.txt', 'desmond_output', 10, 3)
	print("Processing Negative Phrases")
	selectExamplesToHandlabel('cnn_input/negativePhrases_bigram_top5_window_5.txt', 'desmond_negative_bigram_5_window_5_1000', 1300744, 1000)
	print("Processing Neutral Phrases")
	selectExamplesToHandlabel('cnn_input/neutralPhrases_bigram_top5_window_5.txt', 'desmond_neutral_bigram_5_window_5_1000', 6499188, 1000)
	print("Processing Positive Phrases")
	selectExamplesToHandlabel('cnn_input/positivePhrases_bigram_top5_window_5.txt', 'desmond_positive_bigram_5_window_5_1000', 4674723, 1000)

	print("Processing Negative Phrases 2")
	selectExamplesToHandlabel('cnn_input/negativePhrases_bigram_top5_window_5.txt', 'matt_negative_bigram_5_window_5_1000', 1300744, 1000)
	print("Processing Neutral Phrases 2")
	selectExamplesToHandlabel('cnn_input/neutralPhrases_bigram_top5_window_5.txt', 'matt_neutral_bigram_5_window_5_1000', 6499188, 1000)
	print("Processing Positive Phrases 2")
	selectExamplesToHandlabel('cnn_input/positivePhrases_bigram_top5_window_5.txt', 'matt_positive_bigram_5_window_5_1000', 4674723, 1000)
	

if __name__ == '__main__':
    main()
