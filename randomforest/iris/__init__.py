import sys
import os

# Path for spark source folder
os.environ['SPARK_HOME']="path/to/spark"

# Append pyspark  to Python Path
sys.path.append("path/to/spark/python")

try:
    from pyspark import SparkContext, SparkConf
    from pyspark.mllib.tree import RandomForest, RandomForestModel
    from pyspark.mllib.util import MLUtils
    print ("Successfully imported Spark Modules")
except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)

if __name__ == "__main__":
    conf = SparkConf().setAppName("RandomForest")
    sc = SparkContext(conf = conf)
    print "Loading data..."
    data = MLUtils.loadLibSVMFile(sc, '../../data/iris/iris.scale')
    (trainingData, testData) = data.randomSplit([0.7, 0.3])
    # Train a RandomForest model.
    model = RandomForest.trainClassifier(trainingData, numClasses=4,
                                         categoricalFeaturesInfo={},
                                         numTrees=10, featureSubsetStrategy="auto",
                                         impurity='gini', maxDepth=4, maxBins=32)

    # Evaluate model on test instances and compute test error
    predictions = model.predict(testData.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())
    print('Test Error = ' + str(testErr))
    print('Learned classification forest model:')
    print(model.toDebugString())
    # Save and load model
    model.save(sc, "model")
    sameModel = RandomForestModel.load(sc, "model")