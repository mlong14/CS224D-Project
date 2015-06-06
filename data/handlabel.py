import numpy as np

# file: 'readchar.py'
"""
readChar from stackoverflow http://stackoverflow.com/questions/510357/python-read-a-single-character-from-the-user
"""

import tty, sys, termios

class ReadChar():
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setraw(sys.stdin.fileno())
        return sys.stdin.read(1)
    def __exit__(self, type, value, traceback):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)



def runLabeler():
	person = raw_input('\nEnter your name (matt or desmond): ')
	if ((person != 'matt') and (person != 'desmond')):
		print("Error. User not recognized. \n")
		return
	valenceNumber = int(raw_input('\nWhich word list do you want to process? Enter -1 for negative, 0 for neutral, 1 for positive: '))
	if (valenceNumber > 1) and (valenceNumber < -1):
		print("Error. Number not recognized. \n")
		return
	startingIndex = int(raw_input('\nWhich index do you want to start from? (0 for beginning): '))

	if(valenceNumber == 1):
		valence = 'positive'
	elif valenceNumber== -1:
		valence = 'negative'
	else:
		valence = 'neutral'
	print("Preparing " + valence + " labeler for user: " + person + "...\n")

	inputFile = person + '_' + valence + "_bigram_5_window_5_1000"
	outputFile = inputFile + "_labels"

	print("1: Very negative")
	print("2: Slightly negative")
	print("3: Neutral")
	print("4: Slightly positive")
	print("5: Very positive")
	print("Escape or any other key: Exit")
	print("\n Hit any key to proceed \n")

	with ReadChar() as rc:
		char = rc
	#if ord(char) <= 32:
	#	print("You entered character with ordinal {}." .format(ord(char)))
	#else:
	#	print("You entered character '{}'.".format(char))
	#if char in "^C^D":
	#	sys.exit()

	counter = 0
	print(counter)
	with open(inputFile, "r") as f:
		for line in f:
			if counter >= startingIndex:
				print("Index: " + str(counter))
				print(line)
				with ReadChar() as rc:
					char = rc
				if ord(char) == 27:
					print("Exiting... (you were at index: " + str(counter) + ")")
					return
				if ord(char) <= 53 and ord(char) >= 49: #49 = '1', 53 = '5'
					print("You entered: " + char + "\n")
					with open(outputFile + "_" + str(ord(char) - 48),'a+') as fOut:
						fOut.write(line)
					# if ord(char) == 49:
					# 	with open(outputFile + "_1",'a+') as fOut:
					# 		fOut.write(line)
					# elif ord(char) == 50:
					# 	with open(outputFile + "_2",'a+') as fOut:
					# 		fOut.write(line)
					# elif ord(char) == 51:
					# 	with open(outputFile + "_3",'a+') as fOut:
					# 		fOut.write(line)
					# elif ord(char) == 52:
					# 	with open(outputFile + "_4",'a+') as fOut:
					# 		fOut.write(line)
					# else:
					# 	with open(outputFile + "_5",'a+') as fOut:
					# 		fOut.write(line)
					##with open(outputFile,'a') as fOut:
					##	fOut.write(char + '\n')
				else:
					print("Exiting... (you were at index: " + str(counter) + ")")
					return
			counter+=1


def main():
	runLabeler()

if __name__ == '__main__':
    main()