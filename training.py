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
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def prepare():
	os.environ["PYSPARK_PYTHON"] = "python3"
	# 读取s3桶中的数据
	spark = SparkSession.builder \
		.master("yarn") \
		.appName("CPA&GYC match refactor") \
		.config("spark.driver.memory", "1g") \
		.config("spark.executor.cores", "2") \
		.config("spark.executor.instance", "4") \
		.config("spark.executor.memory", "2g") \
		.config('spark.sql.codegen.wholeStage', False) \
		.config("spark.sql.autoBroadcastJoinThreshold", 1048576000) \
    .config("spark.sql.files.maxRecordsPerFile", 33554432) \
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


@udf(returnType=IntegerType())
def check_similarity(packid_check, packid_standard):
	if (packid_check == "") & (packid_standard == ""):
		return 1
	elif len(packid_check) == 0:
		return 0
	elif len(packid_standard) == 0:
		return 0
	else:
		try:
			if int(packid_check) == int(packid_standard):
				return 1
			else:
				return 0
		except ValueError:
			return 0


if __name__ == '__main__':
	spark = prepare()
  
	df_standard = load_standard_prod(spark)
	df_cleanning = load_cleanning_prod(spark)
	# df_cleanning = df_cleanning.limit(100)
	df_interfere = load_interfere_mapping(spark)

	# 1. human interfere
	df_cleanning = human_interfere(spark, df_cleanning, df_interfere)
	df_cleanning.persist()

	# 2. cross join
	df_result = feature_cal(spark, df_cleanning, df_standard)
	df_result.persist()
	df_result.write.mode("overwrite").parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/fulljoin")

	df_result = load_training_data(spark)
	df_result = feature_cal(df_result)

	vec_udf = udf(lambda vs: Vectors.dense(vs), VectorUDT())
	similarity_udf = udf(lambda vs: 0.1 * vs[2] + 0.1 * vs[3] + 0.5 * vs[4] + 0.1 * vs[0] + 0.1 * vs[1] + 0.1 * vs[5], DoubleType())
	df_result = df_result.withColumn("similarity", similarity_udf(df_result.featureCol))
	# df_result = df_result.where(df_result.similarity > 0.7)
	df_result = df_result.withColumn("features", vec_udf(df_result.featureCol)) \
					.withColumn("label", check_similarity(df_result.PACK_ID_CHECK, df_result.PACK_ID_STANDARD))
	df_result.persist()

	df_training = df_result.select("id", "label", "features").orderBy("id")
	# Split the data into train and test
	splits = df_training.randomSplit([0.6, 0.4], 1234)
	train = splits[0]
	test = splits[1]

	# specify layers for the neural network:
	# input layer of size 6 (features)
	# and output of size 2 (boolean)
	layers = [6, 3, 2]

	# create the trainer and set its parameters
	trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)

	# train the model
	model = trainer.fit(train)

	# compute accuracy on the test set
	result = model.transform(test)
	predictionAndLabels = result.select("id", "prediction", "label").orderBy("id")
	predictionAndLabels.show()
	evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
	print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))

	# save the model
	model.write().save("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/model2")
