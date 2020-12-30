# -*- coding: utf-8 -*-
"""alfredyang@pharbers.com.

功能描述：one hot encoding for manifacture name

"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.functions import regexp_replace
from pyspark.sql.functions import when
from pyspark.sql.functions import broadcast
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import isnull
from pyspark.sql import Window
from pyspark.sql.functions import rank
from pyspark.sql.functions import desc
from pyspark.sql.functions import lit
import pandas as pd


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
		.config("spark.sql.execution.arrow.pyspark.enabled", "true") \
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

	df_mnf_recheck = spark.read.parquet("s3a://ph-max-auto/2020-08-11/data_matching/airflow_runs/1609240316/prediction_report_division/mnf_check_path/105bde")
	print(df_mnf_recheck.count())

	windowSpec = Window.orderBy(desc("COSINE_SIMILARITY"))
	df_mnf_recheck = df_mnf_recheck.withColumn("RANK", rank().over(windowSpec))
	df_mnf_recheck = df_mnf_recheck.where((df_mnf_recheck.RANK < 500) & (df_mnf_recheck.RANK >= 400))

	# df_high_scroe_word = spark.read.parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/high_word_seg")
	# df_high_scroe_word.printSchema()

	# df_mnf_recheck.select("MANUFACTURER_NAME", "MANUFACTURER_NAME_CLEANNING_WORDS_SEG", "MANUFACTURER_NAME_STANDARD", "MANUFACTURER_NAME_STANDARD_WORDS_SEG", "COSINE_SIMILARITY").show(truncate=False)
	# df_mnf_recheck.select("MANUFACTURER_NAME", "MANUFACTURER_NAME_CLEANNING_WORDS_SEG", "MANUFACTURER_NAME_STANDARD", "MANUFACTURER_NAME_STANDARD_WORDS_SEG", "COSINE_SIMILARITY").show(100, truncate=False)

	df_mnf_recheck.printSchema()