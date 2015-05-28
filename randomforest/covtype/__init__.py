import sys
import os

# Path for spark source folder
os.environ['SPARK_HOME']="/path/to/spark"

# Append pyspark  to Python Path
sys.path.append("/path/to/spark/python")

try:
    from pyspark import SparkContext, SparkConf
    from pyspark.mllib.tree import RandomForest, RandomForestModel
    from pyspark.mllib.util import MLUtils
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.linalg import Vectors
    print ("Successfully imported Spark Modules")
except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)

def parseData(line):
    values = [float(s) for s in line.split(",")]
    label = values[-1] - 1
    featuresVector = Vectors.dense(values[0:-1])
    return LabeledPoint(label, featuresVector)

if __name__ == "__main__":
    conf = SparkConf().setAppName("RandomForest_CovType")
    sc = SparkContext(conf=conf)
    print "Loading data..."
    rawData = sc.textFile('../../data/covtype/covtype.data')
    data = rawData.map(parseData)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])
    # Train a RandomForest model.
    model = RandomForest.trainClassifier(trainingData, numClasses=10,
                                         categoricalFeaturesInfo={},
                                         numTrees=5, featureSubsetStrategy="auto",
                                         impurity='gini', maxDepth=5, maxBins=32)
    # Evaluate model on test instances and compute test error
    predictions = model.predict(testData.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())
    print('Test Error = ' + str(testErr))
    print('Learned classification forest model:')
    print(model.toDebugString())
    # Save model
    model.save(sc, "model")