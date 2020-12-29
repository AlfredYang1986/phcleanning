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


@pandas_udf(StringType(), PandasUDFType.SCALAR)
def pudf_district_replace(v):
	frame = {
		"TMP": v
	}
	df = pd.DataFrame(frame)
	df["RESULT"] = df["TMP"].apply(lambda x: x.split("-")[0])
	return df["RESULT"]


@pandas_udf(StringType(), PandasUDFType.SCALAR)
def pudf_city_replace(v):
	frame = {
		"TMP": v
	}
	df = pd.DataFrame(frame)


	def city_replace(x):
		dic = {
			"恩施土家族苗族自治区": "恩施土家族苗族自治区",
			"大理白族自治区": "大理白族自治区",
			"文山壮族苗族自治区": "文山壮族苗族自治州",
			"黔东南苗族侗族自治州": "黔东南州",
			"凉山彝族自治州": "凉山州",
			"楚雄彝族自治州": "楚雄州",
			"黔南布依族苗族自治州": "黔南州",
			"湘西土家族苗族自治州": "湘西土家族苗族自治州",
			"西双版纳傣族自治州": "西双版纳州",
			"伊犁哈萨克自治州": "伊犁州",
			"巴音郭楞蒙古自治州": "巴州",
			"昌吉回族自治州": "昌吉州",
			"甘南藏族自治州": "甘南州",
			"临夏回族自治州": "临夏州",
			"海西蒙古族藏族自治州": "海西州",
			"甘孜藏族自治州": "甘孜州",
			"博尔塔拉蒙古自治州": "博尔塔拉州",
			"德宏傣族景颇族自治州": "德宏州",
			"红河哈尼族彝族自治州": "红河哈尼族州",
		}

		if x in dic.keys():
			return dic[x]
		elif x.find("地区") != -1:
				return x
		elif x.find("市") == -1:
			return x + "市"
		else:
			return x


	df["RESULT"] = df["TMP"].apply(lambda x: city_replace(x))
	return df["RESULT"]


@pandas_udf(StringType(), PandasUDFType.SCALAR)
def pudf_province_replace(v):
	frame = {
		"TMP": v
	}
	df = pd.DataFrame(frame)


	def province_replace(x):
		dic = {
			"上海": "上海市",
			"北京": "北京市",
			"天津": "天津市",
			"重庆": "重庆市",
			"广西": "广西壮族自治区",
			"新疆": "新疆维吾尔自治区",
			"宁夏": "宁夏回族自治区",
			"西藏": "西藏自治区",
			"内蒙古": "内蒙古自治区",
		}

		if x in dic.keys():
			return dic[x]
		else:
			return x + "省"


	df["RESULT"] = df["TMP"].apply(lambda x: province_replace(x))
	return df["RESULT"]


@pandas_udf(StringType(), PandasUDFType.SCALAR)
def pudf_name_replace(p, c, d, v):
	frame = {
		"PROVINCE": p,
		"CITY": c,
		"DISTRICT": d,
		"TMP": v
	}
	df = pd.DataFrame(frame)


	def name_replace(px, cx, dx, x):
		r = ""
		if px not in x:
			r = px + r

		if cx not in x:
			r = cx + r

		if dx not in x:
			r = dx + r

		return r + x


	df["RESULT"] = df.apply(lambda x: name_replace(x["PROVINCE"], x["CITY"], x["DISTRICT"], x["TMP"]), axis=1)
	return df["RESULT"]


if __name__ == '__main__':
	spark = prepare()

	# 1. 利用standard中的中文列通过中文分词
	df_standard = spark.read.csv("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/chc_hc_pov_mapping/Universe.csv", header=True) \
							.withColumnRenamed("Prov", "PROVINCE") \
							.withColumnRenamed("City", "CITY") \
							.withColumnRenamed("Disctrict", "DISTRICT") \
							.withColumnRenamed("POV标准化名称", "STANDARD_NAME")

	# 2. 看了一下数据，没有必要分词
	# 标准下一下标准里面的数据
	df_standard = df_standard.where(~isnull(df_standard.PROVINCE))
	df_standard = df_standard.where(~isnull(df_standard.CITY))
	df_standard = df_standard.where(~isnull(df_standard.DISTRICT))
	df_standard = df_standard.where(~isnull(df_standard.STANDARD_NAME))

	df_standard = df_standard.withColumn("CITY", when(df_standard.CITY == "省直辖县级行政区划", df_standard.DISTRICT).otherwise(df_standard.CITY))
	df_standard = df_standard.withColumn("CITY", pudf_city_replace(df_standard.CITY)).distinct()
	df_standard.persist()
	df_standard.show()

	# 1.1 还需要ID化
	df_cleanning = spark.read.csv("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/chc_hc_pov_mapping/POV.csv", header=True) \
						.withColumnRenamed("Province", "PROVINCE") \
						.withColumnRenamed("City.Name", "CITY") \
						.withColumnRenamed("county.city.name", "DISTRICT") \
						.withColumnRenamed("Account.Name", "NAME")
	df_cleanning = df_cleanning.repartition(1).withColumn("ID", monotonically_increasing_id()).repartition(8)

	df_drop = df_cleanning.where(isnull(df_cleanning.DISTRICT))
	df_drop = df_drop.withColumn("STANDARD_NAME", lit("")).withColumn("NAME_CHANGE", lit(""))

	df_cleanning = df_cleanning.where(~isnull(df_cleanning.DISTRICT))
	df_cleanning = df_cleanning.withColumn("PROVINCE", pudf_province_replace(df_cleanning.PROVINCE))
	df_cleanning = df_cleanning.withColumn("DISTRICT", pudf_district_replace(df_cleanning.DISTRICT))
	# df_cleanning = df_cleanning.withColumn("DISTRICT", df_cleanning.DISTRICT[0])
	df_cleanning = df_cleanning.withColumn("NAME_CHANGE", pudf_name_replace(df_cleanning.PROVINCE, df_cleanning.CITY, df_cleanning.DISTRICT, df_cleanning.NAME))

	# 2. Join 一下
	df_cleanning = df_cleanning.join(broadcast(df_standard), on=["PROVINCE", "CITY"], how="left")
	df_cleanning.persist()
	df_not_match = df_cleanning.where(isnull(df_cleanning.STANDARD_NAME))
	df_cleanning = df_cleanning.where(~isnull(df_cleanning.STANDARD_NAME))
	df_cleanning = df_cleanning.withColumn("SIMILARITY", efftiveness_with_jaro_winkler_similarity_in_hc_mapping(df_cleanning.NAME_CHANGE, df_cleanning.STANDARD_NAME))
	windowSpec = Window.partitionBy("ID").orderBy(desc("SIMILARITY"))
	df_cleanning = df_cleanning.withColumn("RANK", rank().over(windowSpec))
	df_cleanning = df_cleanning.where(df_cleanning.RANK == 1)
	df_cleanning.orderBy("ID").repartition(1).write.mode("overwrite").option("header", "true").csv("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/chc_hc_pov_mapping/hc_result")
	# df_not_match.repartition(1).orderBy("ID").write.mode("overwrite").option("header", "true").csv("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/chc_hc_pov_mapping/hc_result_not_match")
	df_not_match.union(df_drop).repartition(1).write.mode("overwrite").option("header", "true").csv("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/chc_hc_pov_mapping/hc_result_not_match")
