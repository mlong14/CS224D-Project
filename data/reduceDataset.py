# Sample dataset and reduce
import sys
import numpy as np
import pdb

if __name__=="__main__":

    input_file = sys.argv[1]
    totalNumberOfEg = int(sys.argv[2])
    numEg = 0;

    with open(input_file,'rb') as f:
        for line in f:

            if numEg >= totalNumberOfEg:
                break

            if np.random.randint(2) == 1:
                numEg += 1
                print line.strip()
