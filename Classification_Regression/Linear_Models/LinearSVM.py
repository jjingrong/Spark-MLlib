"""

Created by: Jing Rong

Note - Python API does not yet support model save/load but will in the future.

Linear SVM is standard method for large-scale classification tasks.
By default trained with L2 SVM -> Faster & more optimised than L1

The linear SVMs algorithm outputs an SVM model. 
Given a new data point, denoted by x, the model makes predictions based on the value of ( (w^T) * x). 
By the default, if wTx >= 0 then the outcome is positive, and negative otherwise.
The following example shows how to load a sample dataset, 
build Logistic Regression model, and make predictions with the 
resulting model to compute the training error.
"""


# Imports
# SGD = Stochastic Gradient Descent. Convex optimization to optimize objective functions.
from pyspark.mllib.classification import LogisticRegressionWithSGD

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
model = LogisticRegressionWithSGD.train(parsedData)

# Evaluate the model based on training data
labelAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
trainingError = labelAndPreds.filter(lambda (v,p): v!=p).count() / float(parsedData.count())

print "Training Error: ", str(trainingError)