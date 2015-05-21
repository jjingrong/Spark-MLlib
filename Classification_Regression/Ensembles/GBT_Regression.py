"""

Created by: Jing Rong

Gradient-Boosted Trees (GBTs) are ensembles of decision trees. 
GBTs iteratively train decision trees in order to minimize a loss function. 
Similar to decision trees, GBTs handle categorical features, extend to the multiclass classification setting, 
	do not require feature scaling, and are able to capture non-linearities and feature interactions.
More sensetive to overfitting if data is noisy. Exhibits higher variance as well.

tl;dr -> Gradient boosting combines weak learners into a single strong learner, in an iterative fashion

Note: MLlib supports GBTs for binary classification and for regression, using both continuous and categorical features.
	  MLlib implements GBTs using the existing decision tree implementation.
	  GBTs do not yet support multiclass classification. For multiclass problems, please use decision trees or Random Forests.


Regression - Predict the outcome (number).
	-> Works by Averaging.  Each tree predicts a real value(number).
	-> Label is predicted to be the mean of the tree predictions.

The example below demonstrates how to load a LIBSVM data file, 
parse it as an RDD of LabeledPoint and then perform regression using a Random Forest. 
The Mean Squared Error (MSE) is computed at the end to evaluate goodness of fit.
"""

# Imports
# Import DecisionTree / DecisionTreeModel
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext

sc = SparkContext("local", "Ensemble")

# Loading and parsing data into RDD of LabeledPoint
# Sample data provided by Spark 1.3.1 folder

# To run locally
# data = MLUtils.loadLibSVMFile(sc, 'sample_libsvm_data.txt')

# To run on hadoop server
data = MLUtils.loadLibSVMFile(sc, 'jingrong/sample_libsvm_data.txt')

# Splits data - Approximately 70% training , 30% testing
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train the Gradient Boosted Forest model
# Empty categoricalFeaturesInfo indicates that all features are continuous.
# In practice -> use more iterations

model = GradientBoostedTrees.trainRegressor(trainingData, categoricalFeaturesInfo={}, numIterations=3)

# Evaluate the model on test instances, compute test error
allPredictions = model.predict(testData.map(lambda x: x.features))
predictionsAndLabels = testData.map(lambda pl: pl.label).zip(allPredictions)
testMeanSquaredError = predictionsAndLabels.map(lambda (v, p): (v - p) * (v - p)).sum() / float(testData.count())

# Printing results
print "Tested Mean Squared Error: ", testMeanSquaredError
print 
print "Learned regression Gradient Boosted Tree model: "
print model.toDebugString()

"""
Optional Saving/Loading of model
model.save(sc, "myModelPath")
sameModel = DecisionTreeModel.load(sc, "myModelPath")
"""


