from sentimentFunctions import SentimentAnnotator

myAnnotator = SentimentAnnotator()

print(myAnnotator.annotate('awesome'))


import os

#testCommand = 'java -cp stanford-corenlp-full-2013-11-12/"*" -mx5g edu.stanford.nlp.sentiment.SentimentPipeline -file test_review.txt > test_sentiment_output.txt'
testCommand = 'java -cp stanford-corenlp-full-2013-11-12/"*" -mx5g edu.stanford.nlp.sentiment.SentimentPipeline -file test_review.txt'
os.system(testCommand)


## This works but doesn't do sentiment
#from corenlp import StanfordCoreNLP
#corenlp_dir = "stanford-corenlp-full-2013-11-12/"
#corenlp = StanfordCoreNLP(corenlp_dir)  # wait a few minutes...
#corenlp.raw_parse("Parse it")