"""

Testing with Hypothesis testing

https://spark.apache.org/docs/latest/mllib-statistics.html

"""

# Imports
from pyspark import SparkContext
from pyspark.mllib.linalg import Vectors, Matrices
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.stat import Statistics


sc = SparkContext("local", "Rubbish")

"""
# RDD of Vectors
data = sc.parallelize([Vectors.dense([2, 0, 0, -2]),
                       Vectors.dense([4, 5, 0,  3]),
                       Vectors.dense([6, 7, 0,  8])])
"""

# Sample vector composing of frequency of events
vect = Vectors.dense([4,5,0,3])

# Summary of the test including the p-value, degrees of freedom,
goodnessOfFitTestResult = Statistics.chiSqTest(vect)

sampleData = [40.0, 24.0, 29.0, 56.0, 32.0, 42.0, 31.0, 10.0, 0.0, 30.0, 15.0, 12.0]
matrix = Matrices.dense(3,4, sampleData)
# Conduct Pearson's independence test on the input contingency matrix
independenceTestResult = Statistics.chiSqTest(matrix)


# Test statistic, the method used, and the null hypothesis.
print "SINGLE VECTOR FIT: "
print goodnessOfFitTestResult 
## Summary of the test including the p-value, degrees of freedom.
print "INDEPENDENCE TEST RESULT: "
print independenceTestResult