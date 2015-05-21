"""

Created by: Jing Rong

Note - Python API does not yet support multiclass classification and model save/load but will in the future.

Logistic regression is widely used to predict a binary response.
For binary classification problems, the algorithm outputs a binary logistic regression model

The following example shows how to load a sample dataset, build Logistic Regression model, 
and make predictions with the resulting model to compute the training error.
"""


# Imports
# The L-BFGS method approximates the objective function locally 
# as a quadratic without evaluating the second partial derivatives of the objective function to construct the Hessian matrix. 
# LogBFGS over mini-batch gradient descent for faster convergence.
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext
from numpy import array

sc = SparkContext("local", "SVM")

# Loading and parsing data
def parsePoint(line):
	vals = [float(i) for i in line.split(' ')]
	return LabeledPoint(vals[0], vals[1:])

# Sample data provided by Spark 1.3.1 folder
data = sc.textFile("jingrong/sample_svm_data.txt")
parsedData = data.map(parsePoint)

# Building the model 
model = LogisticRegressionWithLBFGS.train(parsedData)

# Evaluate the model based on training data
labelAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
trainingError = labelAndPreds.filter(lambda (v,p): v!=p).count() / float(parsedData.count())

print "Training Error: ", str(trainingError)