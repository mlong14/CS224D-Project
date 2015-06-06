
def runConverter():
	person = 'matt'
	valenceNumber = int(raw_input('\nWhich word list do you want to process? Enter -1 for negative, 0 for neutral, 1 for positive: '))
	if (valenceNumber > 1) and (valenceNumber < -1):
		print("Error. Number not recognized. \n")
		return
	startingIndex = 0

	if(valenceNumber == 1):
		valence = 'positive'
	elif valenceNumber== -1:
		valence = 'negative'
	else:
		valence = 'neutral'
	print("Preparing " + valence + " labeler for user: " + person + "...\n")

	inputFile = person + '_' + valence + "_bigram_5_window_5_1000"
	outputFile = inputFile + "_labels"

	counter = 0
	print(counter)
	fIn = open(inputFile, "r")
	with open(outputFile, "r") as fNum:
		for label in fNum:
			#if (counter % 2) == 0:
			line = fIn.readline()
			print("Index: " + str(counter) + ",  Label: " + label)
			print("Line: " + line)
			with open(outputFile + "_" + label.rstrip("\n"),'a+') as fOut:
				fOut.write(line + '\n')
			#counter+=1
	fIn.close()


def main():
	runConverter()

if __name__ == '__main__':
    main()