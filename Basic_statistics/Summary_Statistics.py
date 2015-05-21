"""

Testing with Summary statistics
https://spark.apache.org/docs/latest/mllib-statistics.html

"""

from pyspark.mllib.stat import Statistics
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import SparseVector, Vectors


sc = SparkContext("local", "Rubbish")

mat = sc.parallelize([Vectors.dense([2, 0, 0, -2]),
                       Vectors.dense([4, 5, 0,  3]),
                       Vectors.dense([6, 7, 0,  8])])


# Compute column summary statistics.
summary = Statistics.colStats(mat)
mean =  summary.mean()
variance =  summary.variance()
nonZeros =  summary.numNonzeros()

print "Mean: ", mean
print "Variance: ", variance
print "Non-Zeros: ", nonZeros