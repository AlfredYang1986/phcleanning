# -*- coding: utf-8 -*-
"""alfredyang@pharbers.com.

功能描述： encoding for high score words

"""

import os
from pyspark.sql import SparkSession
from dataparepare import *
from interfere import *
from pdu_feature import *
from pyspark.sql.types import *
from pyspark.sql.functions import when
from pyspark.sql.functions import explode
from pyspark.sql.functions import min
from pyspark.sql.functions import first
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd
import numpy
from math import sqrt


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
	df = spark.read.parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/mnf_seg").select("HIGH_SCORE_WORDS", "SYNONYMS1", "SYNONYMS2", "SYNONYMS3").distinct()
	
	df.show()
	print(df.count())
	df = df.repartition(1).withColumn("ID", monotonically_increasing_id())
	df.repartition(1).write.mode("overwrite").parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/fenci")
	df = spark.read.parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/fenci")
	df.show()
	first_encoding_df = df.where(df.HIGH_SCORE_WORDS != "nan")
	first_encoding_df = first_encoding_df.select("HIGH_SCORE_WORDS", "ID").withColumnRenamed("HIGH_SCORE_WORDS", "HIGH_WORD")
	
	dfdf = df.where(df.SYNONYMS1 == "新致")
	dfdf.show()
	
	second_encoding_df = df.where(df.SYNONYMS1 != "nan")
	second_encoding_df = second_encoding_df.select("SYNONYMS1", "ID").withColumnRenamed("SYNONYMS1", "HIGH_WORD")
	
	third_encoding_df = df.where(df.SYNONYMS2 != "nan")
	third_encoding_df = third_encoding_df.select("SYNONYMS2", "ID").withColumnRenamed("SYNONYMS2", "HIGH_WORD")
	
	dfdf = second_encoding_df.where(second_encoding_df.HIGH_WORD == "新致")
	dfdf.show()
	dfdf = third_encoding_df.where(third_encoding_df.HIGH_WORD == "中峘本草")
	dfdf.show()
	
	fourth_encoding_df = df.where(df.SYNONYMS3 != "nan")
	fourth_encoding_df = fourth_encoding_df.select("SYNONYMS3", "ID").withColumnRenamed("SYNONYMS3", "HIGH_WORD")
	print(first_encoding_df.count())
	print(second_encoding_df.count())
	print(third_encoding_df.count())
	print(fourth_encoding_df.count())
	df = first_encoding_df.unionByName(second_encoding_df)
	df = df.unionByName(third_encoding_df)
	df = df.unionByName(fourth_encoding_df)
	df.show()
	
	df2 = df.groupBy("HIGH_WORD").agg({"ID":"min"}).withColumnRenamed("min(ID)", "ID")
	df2.show()
	xixi1=df2.toPandas()
	xixi1.to_excel('high_word_seg_zyyin2.xlsx', index = False)
	df2.repartition(1).write.mode("overwrite").parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/high_word_seg")
	print(df2.count())
	print(df.select(df.HIGH_WORD).distinct().count())
	print(df.count())