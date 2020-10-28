# -*- coding: utf-8 -*-
"""alfredyang@pharbers.com.

功能描述：crossJoin

"""

import os
from pyspark.sql import SparkSession
from dataparepare import *
from interfere import *
from pdu_feature import *
from pyspark.sql.types import *
from pyspark.sql.functions import desc
from pyspark.sql.functions import rank
from pyspark.sql.functions import when
from pyspark.sql.functions import array
from pyspark.sql.functions import broadcast
from pyspark.sql import Window
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
import re
import pandas as pd


split_data_path = "s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/splitdata"
result_path = "s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/data3"


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
		.config("spark.sql.autoBroadcastJoinThreshold", 1048576000) \
		.config("spark.sql.files.maxRecordsPerFile", 554432) \
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
	df_standard = load_standard_prod(spark)
	df_interfere = load_interfere_mapping(spark)

	# 1. human interfere 与 数据准备
	modify_pool_cleanning_prod(spark)  # 更高的并发数
	df_cleanning = spark.read.parquet(split_data_path)
	df_cleanning = df_cleanning.repartition(1600)
	df_cleanning = human_interfere(spark, df_cleanning, df_interfere)
	df_cleanning = dosage_standify(df_cleanning)  # 剂型列规范
	df_cleanning = spec_standify(df_cleanning)  # 规格列规范

	df_standard = df_standard.withColumn("SPEC", df_standard.SPEC_STANDARD)
	df_standard = spec_standify(df_standard)
	df_standard = df_standard.withColumn("SPEC_STANDARD", df_standard.SPEC).drop("SPEC")

	# 2. cross join
	df_result = df_cleanning.crossJoin(broadcast(df_standard)).na.fill("")

	# 3. jaccard distance
	# 得到一个list，里面是mole_name 和 doasge 的 jd 数值
	df_result = df_result.withColumn("JACCARD_DISTANCE", \
				efftiveness_with_jaccard_distance( \
					df_result.MOLE_NAME, df_result.MOLE_NAME_STANDARD, \
					df_result.DOSAGE, df_result.DOSAGE_STANDARD \
					))

	# 4. cutting for reduce the calculation
	# df_result = df_result.where((df_result.JACCARD_DISTANCE[0] < 0.6) & (df_result.JACCARD_DISTANCE[1] < 0.9))
	df_result = df_result.where((df_result.JACCARD_DISTANCE[0] < 0.6))  # 目前只取了分子名来判断


	# 5. edit_distance is not very good for normalization probloms
	# we use jaro_winkler_similarity instead
	# if not good enough, change back to edit distance
	df_result = df_result.withColumn("EFFTIVENESS", \
					efftiveness_with_jaro_winkler_similarity( \
						df_result.MOLE_NAME, df_result.MOLE_NAME_STANDARD, \
						df_result.PRODUCT_NAME, df_result.PRODUCT_NAME_STANDARD, \
						df_result.DOSAGE, df_result.DOSAGE_STANDARD, \
						df_result.SPEC, df_result.SPEC_STANDARD, \
						df_result.PACK_QTY, df_result.PACK_QTY_STANDARD, \
						df_result.MANUFACTURER_NAME, df_result.MANUFACTURER_NAME_STANDARD, df_result.MANUFACTURER_NAME_EN_STANDARD \
						))

	df_result = df_result.withColumn("EFFTIVENESS_MOLE_NAME", df_result.EFFTIVENESS[0]) \
					.withColumn("EFFTIVENESS_PRODUCT_NAME", df_result.EFFTIVENESS[1]) \
					.withColumn("EFFTIVENESS_DOSAGE", df_result.EFFTIVENESS[2]) \
					.withColumn("EFFTIVENESS_SPEC", df_result.EFFTIVENESS[3]) \
					.withColumn("EFFTIVENESS_PACK_QTY", df_result.EFFTIVENESS[4]) \
					.withColumn("EFFTIVENESS_MANUFACTURER", df_result.EFFTIVENESS[5]) \
					.drop("EFFTIVENESS")

	# df_result.show()

	# features
	assembler = VectorAssembler( \  # 将多列数据转化为单列的向量列
					inputCols=["EFFTIVENESS_MOLE_NAME", "EFFTIVENESS_PRODUCT_NAME", "EFFTIVENESS_DOSAGE", "EFFTIVENESS_SPEC", \
								"EFFTIVENESS_PACK_QTY", "EFFTIVENESS_MANUFACTURER"], \
					outputCol="features")
	df_result = assembler.transform(df_result)

	df_result = df_result.withColumn("PACK_ID_CHECK_NUM", df_result.PACK_ID_CHECK.cast("int")).na.fill({"PACK_ID_CHECK_NUM": -1})
	df_result = df_result.withColumn("PACK_ID_STANDARD_NUM", df_result.PACK_ID_STANDARD.cast("int")).na.fill({"PACK_ID_STANDARD_NUM": -1})
	df_result = df_result.withColumn("label",
					when((df_result.PACK_ID_CHECK_NUM > 0) & (df_result.PACK_ID_STANDARD_NUM > 0) & (df_result.PACK_ID_CHECK_NUM == df_result.PACK_ID_STANDARD_NUM), 1.0).otherwise(0.0)) \
					.drop("PACK_ID_CHECK_NUM", "PACK_ID_STANDARD_NUM")

	df_result.repartition(10).write.mode("overwrite").parquet(result_path)
  