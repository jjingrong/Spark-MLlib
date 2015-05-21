"""
Test running using spark-submit

Lessons learnt :
	1. Require to launch by spark-submit, instead of calling pySpark shell.

Problems faced : 
	1. Cannot have 'spacebars' in directory. Pathing will fail. ( Possible bug/issue to open)
	2. Installation troublesome.

"""


from pyspark import SparkContext

logFile = "tests.txt"  # Should be some file on your system

sc = SparkContext("local", "Rubbish")

logData = sc.textFile(logFile).cache()

numAs = logData.filter(lambda s: 'a' in s).count()
numBs = logData.filter(lambda s: 'b' in s).count()

words = sc.textFile(logFile).count()
firstWord = sc.textFile(logFile).first()


print "Lines with a: %i, lines with b: %i" % (numAs, numBs)
print "Number of words: ", words
print "First item in this RDD: ", firstWord