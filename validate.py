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
from pdu_feature import *


def prepare():
	os.environ["PYSPARK_PYTHON"] = "python3"
	# 读取s3桶中的数据
	spark = SparkSession.builder \
		.master("yarn") \
		.appName("CPA&GYC match refactor") \
		.config("spark.driver.memory", "2g") \
		.config("spark.executor.cores", "4") \
		.config("spark.executor.instances", "4") \
		.config("spark.executor.memory", "2g") \
		.config('spark.sql.codegen.wholeStage', False) \
		.config("spark.sql.execution.arrow.enabled", "true") \
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
	df = spark.read.parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/dosage1_prediction0")
	# df = spark.read.parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/data2")
	# df = df.withColumn("JACCARD_DISTANCE_MOLE_NAME", df.JACCARD_DISTANCE[0]) \
	# 		.withColumn("JACCARD_DISTANCE_DOSAGE", df.JACCARD_DISTANCE[1]) \
	# 		.drop("JACCARD_DISTANCE", "features")
	print(df.count())
	df_r = df.select("id","label", "features", "rawPrediction", "probability", "prediction").where((df.label == 1.0) & (df.prediction == 1.0))
	# df_r.show(1000, truncate=False)
	print(df_r.count())
	df_f = df.select("id","label", "features", "rawPrediction", "probability", "prediction").where((df.label == 1.0) & (df.prediction == 0.0))
	# df_f.show(1000, truncate=False)
	print(df_f.count())

	# df = df.orderBy("id").drop("features").repartition(1).write.mode("overwrite").csv("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/validate")
