"""

Testing with Correlation
https://spark.apache.org/docs/latest/mllib-statistics.html

"""

from pyspark.mllib.stat import Statistics
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import SparseVector, Vectors


sc = SparkContext("local", "Rubbish")

seriesX = sc.parallelize([1.0, 2.0, -2.0], 2)
seriesY = sc.parallelize([3.0, 4.0, 5.0], 2)
corrXY =  Statistics.corr(seriesX, seriesY, method="pearson")

# RDD of Vectors
data = sc.parallelize([Vectors.dense([2, 0, 0, -2]),
                       Vectors.dense([4, 5, 0,  3]),
                       Vectors.dense([6, 7, 0,  8])])

print "Correlation between x & y: ", corrXY
print "Correlation matrix: ", data