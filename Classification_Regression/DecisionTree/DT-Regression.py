"""

Created by: Jing Rong

Regression - Predict the outcome (number).

Scaling potential - Computation scales approximately linearly in the number of training instances, 
					in the number of features, and in the maxBins parameter. 
					Communication scales approximately linearly in the number of features and in maxBins.
					The implemented algorithm reads both sparse and dense data. 
					It is however, not optimized for sparse input.

The example below demonstrates how to load a LIBSVM data file, 
	parse it as an RDD of LabeledPoint, 
	perform classification using a decision tree 
	with Gini impurity as an impurity measure and a maximum tree depth of 5. 

The mean squared error is then calculated to measure the algorithm accuracy.

"""


# Imports
# Import DecisionTree / DecisionTreeModel
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext

sc = SparkContext("local", "SVM")

# Loading and parsing data into RDD of LabeledPoint
# Sample data provided by Spark 1.3.1 folder

# To run locally
#data = MLUtils.loadLibSVMFile(sc, 'sample_libsvm_data.txt')

# To run on hadoop server
data = MLUtils.loadLibSVMFile(sc, 'jingrong/sample_libsvm_data.txt')

# Splits data - Approximately 70% training , 30% testing
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train the decision tree model
# Empty categoricalFeaturesInfo indicates that all features are continuous.
model = DecisionTree.trainRegressor(trainingData, categoricalFeaturesInfo={}, impurity='variance', maxDepth=5, maxBins=32)

# Evaluate the model on test instances, compute test error
allPredictions = model.predict(testData.map(lambda x: x.features))
predictionsAndLabels = testData.map(lambda pl: pl.label).zip(allPredictions)
testMeanSquaredError = predictionsAndLabels.map(lambda (v, p): (v - p) * (v - p)).sum() / float(testData.count())

# Printing results
print "Tested Mean Squared Error: ", testMeanSquaredError
print 
print "Learned regression tree model: "
print model.toDebugString()

"""
Optional Saving/Loading of model
model.save(sc, "myModelPath")
sameModel = DecisionTreeModel.load(sc, "myModelPath")
"""


