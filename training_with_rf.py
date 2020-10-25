# -*- coding: utf-8 -*-
"""alfredyang@pharbers.com.

功能描述：job3：left join cpa和prod
  * @author yzy
  * @version 0.0
  * @since 2020/08/12
  * @note  落盘数据：cpa_prod_join

"""

import os
from pyspark.sql import SparkSession
from dataparepare import *
from interfere import *
from feature import *
# from similarity import *
# from oldsimi import *
from pyspark.sql.types import *
from pyspark.sql.functions import desc
from pyspark.sql.functions import rank
from pyspark.sql import Window
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def prepare():
	os.environ["PYSPARK_PYTHON"] = "python3"
	# 读取s3桶中的数据
	spark = SparkSession.builder \
		.master("yarn") \
		.appName("CPA&GYC match refactor") \
		.config("spark.driver.memory", "1g") \
		.config("spark.executor.cores", "1") \
		.config("spark.executor.instances", "2") \
		.config("spark.executor.memory", "2g") \
		.config('spark.sql.codegen.wholeStage', False) \
		.getOrCreate()

	access_key = os.getenv("AWS_ACCESS_KEY_ID")
	secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
	if access_key is not None:
		spark._jsc.hadoopConfiguration().set("fs.s3a.access.key", access_key)
		spark._jsc.hadoopConfiguration().set("fs.s3a.secret.key", secret_key)
		spark._jsc.hadoopConfiguration().set("fs.s3a.impl","org.apache.hadoop.fs.s3a.S3AFileSystem")
		spark._jsc.hadoopConfiguration().set("com.amazonaws.services.s3.enableV4", "true")
		# spark._jsc.hadoopConfiguration().set("fs.s3a.aws.credentials.provider","org.apache.hadoop.fs.s3a.BasicAWSCredentialsProvider")
		spark._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "s3.cn-northwest-1.amazonaws.com.cn")

	return spark



if __name__ == '__main__':
	spark = prepare()

	# 1. load the training data
	df_result = load_training_data(spark)
	data_training = df_result.select("id", "label", "features").orderBy("id")

	# 2. Split the data into train and test
	labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data_training)

	# 3. specify layers for the neural network:
	featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=6).fit(data_training)

	# 4. create the trainer and set its parameters
	# Split the data into training and test sets (30% held out for testing)
	(trainingData, testData) = data_training.randomSplit([0.7, 0.3])

	# Train a DecisionTree model.
	rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

	# Chain indexers and tree in a Pipeline
	labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=labelIndexer.labels)
	pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

	# Train model.  This also runs the indexers.
	model = pipeline.fit(trainingData)

	# Make predictions.
	predictions = model.transform(testData)

	# Select example rows to display.
	predictions.select("prediction", "indexedLabel", "features").show(5)

	# Select (prediction, true label) and compute test error
	evaluator = MulticlassClassificationEvaluator(
	    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
	accuracy = evaluator.evaluate(predictions)
	print("Test Error = %g " % (1.0 - accuracy))

	treeModel = model.stages[2]
	# summary only
	print(treeModel)

	model.write().save("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/rf")