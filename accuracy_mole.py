# -*- coding: utf-8 -*-
"""alfredyang@pharbers.com.

This is test pyspark job for zzyin
"""

import os
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import re
# import numpy as np
from pyspark.sql.functions import *
from pyspark.sql import Window
from pyspark.sql.functions import col, pandas_udf, PandasUDFType
from pyspark.sql.types import *
import pandas as pd


def execute():
	os.environ["PYSPARK_PYTHON"] = "python3"
	
	spark = SparkSession.builder \
		.master("yarn") \
		.appName("data from s3") \
		.config("spark.driver.memory", "1g") \
		.config("spark.executor.cores", "1") \
		.config("spark.executor.instance", "1") \
		.config("spark.executor.memory", "1g") \
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

	
	df_result = spark.read.parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/validate/error_match/error_match_par")
	# print(df_result.groupBy("MOLE_NAME").agg({"label": "first"}).count())
	# print(df_result.groupBy("id").agg({"label": "first"}).count())
	
	
	schema = StructType([
	    StructField("id", IntegerType(), True),
	    StructField("MOLE_NAME", StringType(), True),
	    StructField("label", DoubleType(), True),
	    StructField("RANK", IntegerType(), True),
	    StructField("per", IntegerType(), True)
	])
	
	@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
	def cal_percentage_pandas_udf(df):
	
		df = df[(df["label"] == 1.0)]
		total_count = df.count()
		df_right =  df[(df["RANK"] <= 5)]
		right_count = df_right.count()
		per = right_count / total_count
		df["per"] = per
		return df
	
	# df_result = df_result.filter(df_result.MOLE_NAME == "沙丁胺醇")
	# df_result.show()
	
	
	df_result = df_result.select("id", "MOLE_NAME", "label", "RANK").groupBy("MOLE_NAME").apply(cal_percentage_pandas_udf)
	df_result.show()
	print(df_result.count())
	

	
execute()
		
