"""

Created by: Jing Rong

* Random forests correct for decision trees' habit of overfitting to their training set.

Random forests train a set of decision trees separately, so the training can be done in parallel. 
The algorithm injects randomness into the training process so that each decision tree is a bit different. 
Combining the predictions from each tree reduces the variance of the predictions, improving the performance on test data.

Training - Randomess is injected into the training process
	-> Subsampling the original dataset on each iteration to get a different training set (a.k.a. bootstrapping).
	-> Considering different random subsets of features to split on at each tree node.
	-> Decision tree training is done the same way as individual decision trees ( in Linear Model)
Prediction - Requires for random forest to aggregate the prediction from its set of decision trees.
	-> Done differently for classification and regression

Note: Tuning (numTrees , maxDepth) parameters can improve performance.
	numTrees -> Training time increases roughly linearly with number of trees
			 -> Increasing trees decreases variance in prediction, improving model's test-time accuracy
	maxDepth -> Increasing depth makes model more expressive and powerful
			 -> Deep trees take longer to train, more prone to overfitting
			 -> Generally more acceptable to use deep trees for random forest compared to single tree.

Regression - Predict the outcome (number).
	-> Works by Averaging.  Each tree predicts a real value(number).
	-> Label is predicted to be the mean of the tree predictions.

The example below demonstrates how to load a LIBSVM data file, 
parse it as an RDD of LabeledPoint and then perform regression using a Random Forest. 
The Mean Squared Error (MSE) is computed at the end to evaluate goodness of fit.
"""


# Imports
# Import DecisionTree / DecisionTreeModel
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext

sc = SparkContext("local", "Ensemble")

# Loading and parsing data into RDD of LabeledPoint
# Sample data provided by Spark 1.3.1 folder

# To run locally
#data = MLUtils.loadLibSVMFile(sc, 'sample_libsvm_data.txt')

# To run on hadoop server
data = MLUtils.loadLibSVMFile(sc, 'jingrong/sample_libsvm_data.txt')

# Splits data - Approximately 70% training , 30% testing
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train the Random Forest model
# Empty categoricalFeaturesInfo indicates that all features are continuous.
# In practice -> use larger numTrees

# Settings featureSubsetStrategy to "auto" lets the algo choose automatically
model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo={},
                                    numTrees=3, featureSubsetStrategy="auto",
                                    impurity='variance', maxDepth=4, maxBins=32)

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


