# -*- coding: utf-8 -*-
"""alfredyang@pharbers.com.

功能描述：one hot encoding for manifacture name

"""

import os
from pyspark.sql import SparkSession
from dataparepare import *
from interfere import *
from pdu_feature import *
from pyspark.sql.types import *
from pyspark.sql.functions import array, array_contains
from pyspark.sql.functions import broadcast
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import explode
from pyspark.sql.functions import pandas_udf, PandasUDFType


def prepare():
	os.environ["PYSPARK_PYTHON"] = "python3"
	# 读取s3桶中的数据
	spark = SparkSession.builder \
		.master("yarn") \
		.appName("CPA&GYC match refactor") \
		.config("spark.driver.memory", "2g") \
		.config("spark.executor.cores", "2") \
		.config("spark.executor.instances", "2") \
		.config("spark.executor.memory", "2g") \
		.config('spark.sql.codegen.wholeStage', False) \
		.config("spark.sql.execution.arrow.enabled", "true") \
		.config("spark.sql.crossJoin.enabled", "true") \
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

	# 1. 利用standard中的中文列通过中文分词
	df_cleanning = load_training_data(spark)
	# 增加两列MANUFACTURER_NAME_CLEANNING_WORDS MANUFACTURER_NAME_STANDARD_WORDS - array(string)
	df_cleanning = phcleanning_mnf_seg(df_cleanning, "MANUFACTURER_NAME_STANDARD", "MANUFACTURER_NAME_STANDARD_WORDS")
	df_cleanning = phcleanning_mnf_seg(df_cleanning, "MANUFACTURER_NAME", "MANUFACTURER_NAME_CLEANNING_WORDS")
	df_cleanning.where((df_cleanning.label == 1.0) & (df_cleanning.EFFTIVENESS_MANUFACTURER < 0.9)) \
		.select("MANUFACTURER_NAME", "MANUFACTURER_NAME_CLEANNING_WORDS", "MANUFACTURER_NAME_STANDARD", "MANUFACTURER_NAME_STANDARD_WORDS", "EFFTIVENESS_MANUFACTURER").show(10)

	# 2. WORD 编码化
	# 2.1 读WORD 编码
	schema = StructType([
			StructField("WORD", StringType()), \
			StructField("ENCODE", IntegerType()), \
		])
	df_encode = spark.read.csv("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/district/words", schema=schema)
	
	df_cleanning = words_to_reverse_index(df_cleanning, df_encode, "MANUFACTURER_NAME_STANDARD_WORDS", "MANUFACTURER_NAME_STANDARD_WORDS")
	df_cleanning = words_to_reverse_index(df_cleanning, df_encode, "MANUFACTURER_NAME_CLEANNING_WORDS", "MANUFACTURER_NAME_CLEANNING_WORDS")
	df_cleanning.where((df_cleanning.label == 1.0) & (df_cleanning.EFFTIVENESS_MANUFACTURER < 0.9)) \
		.select("MANUFACTURER_NAME", "MANUFACTURER_NAME_CLEANNING_WORDS", "MANUFACTURER_NAME_STANDARD", "MANUFACTURER_NAME_STANDARD_WORDS", "EFFTIVENESS_MANUFACTURER").show(10)

	# df_cleanning.repartition(10).write.mode("overwrite").parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/district/words-index")
