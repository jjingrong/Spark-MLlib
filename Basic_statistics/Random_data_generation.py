"""

Testing with Random data generation

https://spark.apache.org/docs/latest/mllib-statistics.html

"""

from pyspark.mllib.random import RandomRDDs
from pyspark import SparkContext

sc = SparkContext("local", "Rubbish")

# Generate a random double RDD that contains 1 million i.i.d. values drawn from the
# standard normal distribution `N(0, 1)`, evenly distributed in 10 partitions.
u = RandomRDDs.uniformRDD(sc, 1000000L, 10)
# Apply a transform to get a random double RDD following `N(1, 4)`.
v = u.map(lambda x: 1.0 + 2.0 * x)

print v