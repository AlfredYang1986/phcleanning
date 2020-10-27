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
from pyspark.sql.types import *
from pyspark.sql.functions import desc
from pyspark.sql.functions import rank
from pyspark.sql import Window
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
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

	# 1.5. cutting for the more accurate training
	df_result = df_result.withColumn("SIMILARITY", \
					df_result.EFFTIVENESS_MOLE_NAME + \
					df_result.EFFTIVENESS_PRODUCT_NAME + \
					df_result.EFFTIVENESS_DOSAGE + \
					df_result.EFFTIVENESS_SPEC + \
					df_result.EFFTIVENESS_PACK_QTY + \
					df_result.EFFTIVENESS_MANUFACTURER)

	windowSpec  = Window.partitionBy("id").orderBy(desc("SIMILARITY"))

	df_result = df_result.withColumn("RANK", rank().over(windowSpec))
	df_result = df_result.where(df_result.RANK <= 5)
	df_result.persist()

	data_training = df_result.select("id", "label", "features").orderBy("id")

	# 2. Split the data into train and test
	labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data_training)

	# 3. specify layers for the neural network:
	featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=2).fit(data_training)

	# 4. create the trainer and set its parameters
	# Split the data into training and test sets (30% held out for testing)
	(trainingData, testData) = data_training.randomSplit([0.7, 0.3])

	# Train a DecisionTree model.
	dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

	# Chain indexers and tree in a Pipeline
	pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

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

	model.write().overwrite().save("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/dt")