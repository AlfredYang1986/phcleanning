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

	# 0. load the cleanning data
	df_cleanning = load_training_data(spark).select("id").distinct()
	# Split the data into training and test sets (30% held out for testing)
	(df_training, df_test) = df_cleanning.randomSplit([0.7, 0.3])

	# 1. load the training data
	# 准备训练集合
	df_result = load_training_data(spark)
	df_result = df_result.select("id", "label", "features")
	labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(df_result)
	featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=6).fit(df_result)
	df_training.show(10)
	# 1.1 构建训练集合
	df_training = df_training.join(df_result, how="left", on="id")
	df_training.show()

	# 1.2 构建测试集合
	df_test = df_test.join(df_result, how="left", on="id")
	df_test.show()

	# Train a DecisionTree model.
	dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

	# Chain indexers and tree in a Pipeline
	pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

	# Train model.  This also runs the indexers.
	model = pipeline.fit(df_training)

	# Make predictions.
	df_predictions = model.transform(df_test)

	# Select example rows to display.
	df_predictions.show(10)
	df_predictions.select("prediction", "indexedLabel", "features").show(5)

	# Select (prediction, true label) and compute test error
	evaluator = MulticlassClassificationEvaluator(
	    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
	accuracy = evaluator.evaluate(df_predictions)
	print("Test Error = %g " % (1.0 - accuracy))

	treeModel = model.stages[2]
	# summary only
	print(treeModel)

	# model.write().overwrite().save("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/dt")
	model.write().overwrite().save("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/pfizer_model/0.0.4/model")