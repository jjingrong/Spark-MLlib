"""

Created by: Jing Rong

Note - Python API does not yet support model save/load but will in the future.

Linear least squares is the most common formulation for regression problems.
Linear regression predicts scores on 1 variable from scores on a second variable.
	-> Assumes linear relationship between the 2 variables
	-> Find best fit straight line through the x-y plot.

The following example demonstrate how to load training data, parse it as an RDD of LabeledPoint. 
"""


# Imports
# The example then uses LinearRegressionWithSGD to build a simple linear model to predict label values. 
# We then compute the mean squared error at the end to evaluate goodness of fit.
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark import SparkContext
from numpy import array

sc = SparkContext("local", "SVM")

# Loading and parsing data
def parsePoint(line):
	# Replace comma with spacebar for data
	vals = [float(i) for i in line.replace(',', ' ').split(' ')]
	return LabeledPoint(vals[0], vals[1:])

# Sample data provided by Spark 1.3.1 folder
data = sc.textFile("jingrong/lpsa.data")
parsedData = data.map(parsePoint)

# Building the model 
model = LinearRegressionWithSGD.train(parsedData)

# Evaluate the model based on training data
# Calculates mean-squared error to evaluate the goodness of fit.
valuesAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
MSE = valuesAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / valuesAndPreds.count()

print("Mean Squared Error = " + str(MSE))