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
from pyspark.ml.classification import MultilayerPerceptronClassificationModel
from pyspark.ml.classification import DecisionTreeClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import PipelineModel


def prepare():
	os.environ["PYSPARK_PYTHON"] = "python3"
	# 读取s3桶中的数据
	spark = SparkSession.builder \
		.master("yarn") \
		.appName("CPA&GYC match refactor") \
		.config("spark.driver.memory", "1g") \
		.config("spark.executor.cores", "2") \
		.config("spark.executor.instances", "4") \
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

	# 1. load the data
	df_result = load_training_data(spark)
	df_validate = df_result #.select("id", "label", "features").orderBy("id")

	# 2. load model
	model = PipelineModel.load("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/dt")

	# 3. compute accuracy on the test set
	predictions = model.transform(df_validate)
	predictions.where((predictions.label == 1.0) & (predictions.prediction == 0.0)).show(truncate=False)
	# predictions.where((predictions.label == 1.0) & (predictions.prediction == 0.0)).select("id", "label", "probability", "prediction").show(truncate=False)
	# predictions.where((predictions.label == 1.0) & (predictions.prediction == 1.0)).select("id", "label", "probability", "prediction").show(truncate=False)
	evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
	accuracy = evaluator.evaluate(predictions)
	print("Test Error = %g " % (1.0 - accuracy))
	print("Test set accuracy = " + str(accuracy))

	# 4. Test with Pharbers defined methods
	result = predictions
	result.printSchema()
	result = result.withColumn("JACCARD_DISTANCE_MOLE_NAME", result.JACCARD_DISTANCE[0]) \
				.withColumn("JACCARD_DISTANCE_DOSAGE", result.JACCARD_DISTANCE[1]) \
				.drop("JACCARD_DISTANCE", "features", "indexedFeatures").drop("rawPrediction", "probability")
	result.orderBy("id").repartition(1).write.mode("overwrite").csv("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/result")
	df_ph = result.where((result.prediction == 1.0) | (result.label == 1.0))
	ph_total = result.groupBy("id").agg({"prediction": "first", "label": "first"}).count()
	print(ph_total)
	ph_positive_hit = result.where((result.prediction == result.label) & (result.label == 1.0)).count()
	print(ph_positive_hit)
	# ph_negetive_hit = result.where(result.prediction != result.label).count()
	print("Pharbers Test set accuracy = " + str(ph_positive_hit / ph_total))