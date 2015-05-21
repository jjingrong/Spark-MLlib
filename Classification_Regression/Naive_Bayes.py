"""

Created by: Jing Rong

Note - Python API does not yet support model save/load but will in the future.

Naive Bayes is a simple multiclass classification algorithm with the assumption of independence between every pair of features. 
It can also be trained very efficiently in a supervised learning setting.

Within a single pass to the training data, it computes the conditional probability distribution of each feature given label, 
	and then applies Bayes' theorem to compute the conditional probability distribution of label given an observation, then use it for prediction.

Additive smoothing can be used by setting the parameter (lambda), which is default to 1.0

Multinomial naive Bayes is typically used for document classification.
Each observation is a document and each feature represents a term whose value is the frequency of the term.

NaiveBayes implements multinomial naive Bayes. 
It takes an RDD of LabeledPoint and an optionally smoothing parameter lambda as input, and output a NaiveBayesModel, which can be used for evaluation and prediction.

"""


# Imports
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext
import os

sc = SparkContext("local", "SVM")
directoryPath = os.path.dirname(os.path.abspath(__file__))

# Loading and parsing data
def parseLine(line):
    tokens = line.split(',')
    label = float(tokens[0])
    features = Vectors.dense([float(i) for i in tokens[1].split(' ')])
    return LabeledPoint(label, features)

# Sample Naive Bayes data provided by Spark 1.3.1 folder
fileName = os.path.join(directoryPath, "sample_naive_bayes_data.txt")
print "FILE NAME: ",(fileName)
data = sc.textFile("jingrong/sample_naive_bayes_data.txt").map(parseLine)

# Splits data - Approximately 60% training , 40% testing
forTraining, forTest = data.randomSplit([0.6, 0.4], seed = 0)

# Train the naive Bayes model
model = NaiveBayes.train(forTraining, 1.0)

# Make prediction
labelsAndPredictions = forTest.map(lambda p : (model.predict(p.features), p.label))
# Calculate accuracy to test
accuracy = 1.0 * labelsAndPredictions.filter(lambda (x, v): x == v).count() / forTest.count()

# Print results 
print "Accuracy: ", accuracy
