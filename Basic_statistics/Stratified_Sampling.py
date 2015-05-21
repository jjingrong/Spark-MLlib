"""

Testing with Stratified Sampling
Sample by key | Sample by key exact ( Not supported by Python)

https://spark.apache.org/docs/latest/mllib-statistics.html

"""

# Imports
from pyspark.mllib.stat import Statistics
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import SparseVector, Vectors

sc = SparkContext("local", "Rubbish")

# Sample fraction
fractions = {"a": 0.2, "b": 0.1, "c" : 0.15}
# RDD
data = sc.parallelize(fractions.keys()).cartesian(sc.parallelize(range(0, 1000)))

# Get approximate sample by key
approxSample = data.sampleByKey(False, fractions)

print "Approximate sample by key: ", approxSample