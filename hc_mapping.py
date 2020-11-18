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
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import regexp_replace
from pyspark.sql.functions import concat
from pyspark.sql.functions import broadcast
from pyspark.sql.functions import isnull
import pandas as pd
import numpy
from pyspark.sql import Window
from pyspark.sql.functions import rank


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


@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def efftiveness_with_jaro_winkler_similarity_in_hc_mapping(cn, sn):
	def jaro_similarity(s1, s2):
		# First, store the length of the strings
		# because they will be re-used several times.
		# len_s1 = 0 if s1 is None else len(s1)
		# len_s2 = 0 if s2 is None else len(s2)
		len_s1, len_s2 = len(s1), len(s2)

		# The upper bound of the distance for being a matched character.
		match_bound = max(len_s1, len_s2) // 2 - 1

		# Initialize the counts for matches and transpositions.
		matches = 0  # no.of matched characters in s1 and s2
		transpositions = 0  # no. of transpositions between s1 and s2
		flagged_1 = []  # positions in s1 which are matches to some character in s2
		flagged_2 = []  # positions in s2 which are matches to some character in s1

		# Iterate through sequences, check for matches and compute transpositions.
		for i in range(len_s1):  # Iterate through each character.
			upperbound = min(i + match_bound, len_s2 - 1)
			lowerbound = max(0, i - match_bound)
			for j in range(lowerbound, upperbound + 1):
				if s1[i] == s2[j] and j not in flagged_2:
					matches += 1
					flagged_1.append(i)
					flagged_2.append(j)
					break
		flagged_2.sort()
		for i, j in zip(flagged_1, flagged_2):
			if s1[i] != s2[j]:
				transpositions += 1

		if matches == 0:
			return 0
		else:
			return (
				1
				/ 3
				* (
					matches / len_s1
					+ matches / len_s2
					+ (matches - transpositions // 2) / matches
				)
			)


	def jaro_winkler_similarity(s1, s2, p=0.1, max_l=4):
		if not 0 <= max_l * p <= 1:
		    print("The product  `max_l * p` might not fall between [0,1].Jaro-Winkler similarity might not be between 0 and 1.")

		# Compute the Jaro similarity
		jaro_sim = jaro_similarity(s1, s2)

		# Initialize the upper bound for the no. of prefixes.
		# if user did not pre-define the upperbound,
		# use shorter length between s1 and s2

		# Compute the prefix matches.
		l = 0
		# zip() will automatically loop until the end of shorter string.
		for s1_i, s2_i in zip(s1, s2):
			if s1_i == s2_i:
				l += 1
			else:
				break
			if l == max_l:
				break
		# Return the similarity value as described in docstring.
		return jaro_sim + (l * p * (1 - jaro_sim))


	frame = {
		"NAME": cn,
		"STANDARD_NAME": sn,
	}
	df = pd.DataFrame(frame)

	df["NAME_JWS"] = df.apply(lambda x: jaro_winkler_similarity(x["NAME"], x["STANDARD_NAME"]), axis=1)
	return df["NAME_JWS"]


if __name__ == '__main__':
	spark = prepare()

	# 1. 利用standard中的中文列通过中文分词
	df_standard = spark.read.csv("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/chc_hc_cleanning/chc_standard.csv", header=True) \
						.withColumnRenamed("TM编码", "TM_CODE") \
						.withColumnRenamed("省", "PROVINCE") \
						.withColumnRenamed("地级市", "CITY") \
						.withColumnRenamed("区[县/县级市]", "DISTRICT") \
						.withColumnRenamed("机构名称", "STANDARD_NAME") \
						.withColumnRenamed("机构类型", "STANDARD_TYPE") \
						.withColumnRenamed("地址", "ADDRESS") \
						.withColumnRenamed("邮编", "MAIL_CODE") \
						.withColumnRenamed("总诊疗人次数", "PER")

	# 2. 看了一下数据，没有必要分词
	# 标准下一下标准里面的数据
	df_standard = df_standard.withColumn("PROVINCE", regexp_replace(df_standard.PROVINCE, "省", ""))
	df_standard = df_standard.withColumn("PROVINCE", regexp_replace(df_standard.PROVINCE, "市", ""))
	df_standard = df_standard.withColumn("PROVINCE", regexp_replace(df_standard.PROVINCE, "自治区", ""))
	df_standard = df_standard.withColumn("CITY", when(df_standard.CITY == "省直辖县级行政区划", df_standard.DISTRICT).otherwise(df_standard.CITY))
	df_standard = df_standard.withColumn("CITY", regexp_replace(df_standard.CITY, "市", ""))
	df_standard = df_standard.withColumn("STANDARD_NAME", concat(df_standard.DISTRICT, df_standard.STANDARD_NAME))

	# 1.1 还需要ID化
	df_cleanning = spark.read.csv("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/chc_hc_cleanning/hc_cleanning.csv", header=True) \
						.withColumnRenamed("省份编号", "PROVINCE_CODE") \
						.withColumnRenamed("省份", "PROVINCE") \
						.withColumnRenamed("城市编号", "CITY_CODE") \
						.withColumnRenamed("城市", "CITY") \
						.withColumnRenamed("机构代码", "CODE") \
						.withColumnRenamed("机构名称", "NAME") \
						.withColumnRenamed("机构类型", "TYPE") \
						.withColumnRenamed("机构子类型", "SUBTYPE") \
						.withColumnRenamed("治疗领域", "AREA") \
						.withColumnRenamed("年", "YEAR") \
						.withColumnRenamed(" 2019FY销量 ", "2019_FY_SALES").repartition(1).withColumn("ID", monotonically_increasing_id()).repartition(8)

	df_cleanning = df_cleanning.withColumn("PROVINCE", regexp_replace(df_cleanning.PROVINCE, "省", ""))
	df_cleanning = df_cleanning.withColumn("PROVINCE", regexp_replace(df_cleanning.PROVINCE, "市", ""))
	df_cleanning = df_cleanning.withColumn("PROVINCE", regexp_replace(df_cleanning.PROVINCE, "自治区", ""))
	df_cleanning = df_cleanning.withColumn("CITY", regexp_replace(df_cleanning.CITY, "市", ""))

	# 2. Join 一下
	df_cleanning = df_cleanning.join(broadcast(df_standard), on=["PROVINCE", "CITY"], how="left")
	df_cleanning.persist()
	df_not_match = df_cleanning.where(isnull(df_cleanning.STANDARD_NAME))
	df_cleanning = df_cleanning.where(~isnull(df_cleanning.STANDARD_NAME))
	df_cleanning = df_cleanning.repartition(800).withColumn("SIMILARITY", efftiveness_with_jaro_winkler_similarity_in_hc_mapping(df_cleanning.NAME, df_cleanning.STANDARD_NAME))
	windowSpec = Window.partitionBy("ID").orderBy(desc("SIMILARITY"))
	df_cleanning = df_cleanning.withColumn("RANK", rank().over(windowSpec))
	df_cleanning = df_cleanning.where(df_cleanning.RANK == 1)
	df_cleanning.repartition(1).write.mode("overwrite").option("header", "true").csv("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/chc_hc_cleanning/hc_result_2")
	df_not_match.repartition(1).write.mode("overwrite").option("header", "true").csv("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/chc_hc_cleanning/hc_result_not_match")
	