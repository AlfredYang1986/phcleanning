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
from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import udf
from pyspark.sql.functions import when
from pyspark.sql.functions import first
from pyspark.sql.functions import array
from pyspark.sql.functions import to_json
from pyspark.sql.functions import explode
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer, StopWordsRemover
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd
import numpy
from math import sqrt
import jieba
import jieba.posseg as pseg
import jieba.analyse as analyse


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


@pandas_udf(StringType(), PandasUDFType.SCALAR)
def manifacture_name_en_standify(en):
	frame = {
		"MANUFACTURER_NAME_EN_STANDARD": en,
	}
	df = pd.DataFrame(frame)

	# @尹 需要换成regex
	df["MANUFACTURER_NAME_EN_STANDARD_STANDIFY"] = df["MANUFACTURER_NAME_EN_STANDARD"].apply(lambda x: x.replace(".", " ").replace("-", " "))
	return df["MANUFACTURER_NAME_EN_STANDARD_STANDIFY"]


@pandas_udf(ArrayType(StringType()), PandasUDFType.SCALAR)
def manifacture_name_pseg_cut(mnf):
	frame = {
		"MANUFACTURER_NAME_STANDARD": mnf,
	}
	df = pd.DataFrame(frame)


	def pseg_cutting(x):
		result = []
		words = pseg.cut(x)
		for word, _ in words:
			result.append(word)

		return result

	df["MANUFACTURER_NAME_STANDARD_WORDS"] = df["MANUFACTURER_NAME_STANDARD"].apply(lambda x: pseg_cutting(x))
	return df["MANUFACTURER_NAME_STANDARD_WORDS"]


@udf
def cosine_distance_between_mnf(array):
	u = array[0].toArray()
	v = array[1].toArray()
	return float(numpy.dot(u, v) / (sqrt(numpy.dot(u, u)) * sqrt(numpy.dot(v, v))))


@pandas_udf(IntegerType(), PandasUDFType.SCALAR)
def dic_words_to_index(words):
	frame = {
		"WORDS": words
	}
	df = pd.DataFrame(frame)


	def is_geo_tag(w):
		t = analyse.extract_tags(w, topK=5, withWeight=True, allowPOS=("ns",))
		if len(t) == 0:
			return 0
		else:
			return 1

	df["GEO_TAG"] = df["WORDS"].apply(lambda x: is_geo_tag(x))
	return df["GEO_TAG"]


if __name__ == '__main__':
	spark = prepare()

	# 1. 利用standard中的中文列通过中文分词
	df_standard = load_standard_prod(spark)
	df_standard = df_standard.withColumn("MANUFACTURER_NAME_EN_STANDARD", manifacture_name_en_standify(df_standard.MANUFACTURER_NAME_EN_STANDARD))
	# df_standard.select("MANUFACTURER_NAME_STANDARD", "MANUFACTURER_NAME_EN_STANDARD").show(truncate=False)

	# 2. 英文的分词方法，tokenizer
	tokenizer = Tokenizer(inputCol="MANUFACTURER_NAME_EN_STANDARD", outputCol="MANUFACTURER_NAME_EN_WORDS")
	df_standard = tokenizer.transform(df_standard)

	# 3. 中文的分词，jieba
	df_standard = df_standard.withColumn("MANUFACTURER_NAME_WORDS", manifacture_name_pseg_cut(df_standard.MANUFACTURER_NAME_STANDARD))
	# df_standard.select("MANUFACTURER_NAME_STANDARD", "MANUFACTURER_NAME_WORDS", "MANUFACTURER_NAME_EN_STANDARD", "MANUFACTURER_NAME_EN_WORDS").show(truncate=False)

	# 4. 分词之后构建词库编码
	# df_standard.show(truncate=False)

	# 4.1 stop word remover 去掉不需要的词
	stopWords = ["股份", "有限", "总公司", "公司", "集团", "制药", "总厂", "厂", "药业", "责任", "健康", "科技", "生物", "工业", "保健", "医药", "(", ")", "（", "）", \
				 "有限公司", "股份", "控股", "集团", "总公司", "总厂", "厂", "责任", "公司", "有限", "有限责任", \
			     "药业", "医药", "制药", "控股集团", "医药集团", "控股集团", "集团股份", "生物医药", "药厂"]
	remover = StopWordsRemover(stopWords=stopWords, inputCol="MANUFACTURER_NAME_WORDS", outputCol="MANUFACTURER_NAME_WORDS_FILTER")

	df_words_cn = df_standard.select("MANUFACTURER_NAME_WORDS")
	df_words_cn = remover.transform(df_words_cn)
	# df_words_cn.show(truncate=False)

	# 4.2 分离地理维度集合
	df_words_cn_dic = df_words_cn.select(explode("MANUFACTURER_NAME_WORDS_FILTER").alias("WORD")).distinct()
	# df_words_cn_dic = df_words_cn_dic.withColumn("GEO_TAG", dic_words_to_index(df_words_cn_dic.WORD))
	# df_words_cn_dic.show()
	# print(df_words_cn_dic.count())

	# 5. 省市相关编码
	# 5.1 地理维度的编码
	schema = StructType([
				StructField("ID", StringType()), \
    			StructField("NAME", StringType()), \
    			StructField("PARENTID", StringType()), \
    			StructField("SN1", StringType()), \
    			StructField("SN2", StringType()), \
    			StructField("SNC", StringType()), \
    			StructField("NOTHING", StringType()), \
    			StructField("LEVEL", StringType()), \
    		])
	df_geo_standard = spark.read.csv(path="s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/district/district-full.csv", sep="\t", schema=schema)
	df_geo_standard = df_geo_standard.select("ID", "NAME", "PARENTID")
	# df_geo_standard.show()

	# 5.2 优先搞定省市问题
	# df_words_cn_dic_encoder = df_words_cn_dic.where(df_words_cn_dic.GEO_TAG == 1)
	df_words_cn_dic_encoder = df_words_cn_dic
	df_words_cn_dic_encoder = df_words_cn_dic_encoder.join(df_geo_standard, df_words_cn_dic_encoder.WORD == df_geo_standard.NAME, how="left")
	df_words_cn_dic_encoder = df_words_cn_dic_encoder.na.fill({"ID": "-1", "NAME": "", "PARENTID": "-1"})
	df_words_cn_dic_encoder = df_words_cn_dic_encoder.select("WORD", df_words_cn_dic_encoder.ID.alias("AREACODE"), df_words_cn_dic_encoder.PARENTID.alias("AREA_PARENT_ID"))

	count = df_words_cn_dic_encoder.where(~(df_words_cn_dic_encoder.AREA_PARENT_ID == "0") & ~(df_words_cn_dic_encoder.AREA_PARENT_ID == "-1")).count()
	while count > 0:
		df_words_cn_dic_encoder = df_words_cn_dic_encoder.join(df_geo_standard, df_words_cn_dic_encoder.AREA_PARENT_ID == df_geo_standard.ID, how="left")
		df_words_cn_dic_encoder = df_words_cn_dic_encoder.na.fill({"ID": "-1", "NAME": "", "PARENTID": "-1"})
		df_words_cn_dic_encoder = df_words_cn_dic_encoder.withColumn("AREACODE", when(~(df_words_cn_dic_encoder.AREA_PARENT_ID == "0"), df_words_cn_dic_encoder.ID).otherwise(df_words_cn_dic_encoder.AREACODE)) \
																.withColumn("AREA_PARENT_ID", when(~(df_words_cn_dic_encoder.AREA_PARENT_ID == "0"), df_words_cn_dic_encoder.PARENTID).otherwise(df_words_cn_dic_encoder.AREA_PARENT_ID)) \
																.select("WORD", "AREACODE", "AREA_PARENT_ID")
																# .select("WORD", "AREACODE", "AREA_PARENT_ID", "NAME")
		count = df_words_cn_dic_encoder.where(~(df_words_cn_dic_encoder.AREA_PARENT_ID == "0") & ~(df_words_cn_dic_encoder.AREA_PARENT_ID == "-1")).count()

	df_words_cn_dic_encoder = df_words_cn_dic_encoder.withColumn("GEO_ENCODE", df_words_cn_dic_encoder.AREACODE.cast(IntegerType()))
	df_words_cn_dic_encoder = df_words_cn_dic_encoder.withColumn("GEO_ENCODE", when(df_words_cn_dic_encoder.GEO_ENCODE > 0, 1000 + df_words_cn_dic_encoder.GEO_ENCODE).otherwise(-1)) \
														.select("WORD", "GEO_ENCODE")
	df_words_cn_dic_encoder.show()

	# 6. 国家编码的问题
	df_country_standard = spark.read.csv(path="s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/district/COUNTRY_NAME.csv", header=True) \
								.repartition(1).withColumn("ID", monotonically_increasing_id())
	df_words_cn_dic_encoder = df_words_cn_dic_encoder.join(df_country_standard, df_words_cn_dic_encoder.WORD == df_country_standard.COUNTRY_CN_NAME, how="left")
	df_words_cn_dic_encoder	= df_words_cn_dic_encoder.withColumn("COUNTRY_ENCODE", df_words_cn_dic_encoder.ID.cast(IntegerType())).na.fill(-1.0)
	df_words_cn_dic_encoder = df_words_cn_dic_encoder.select("WORD", "GEO_ENCODE", "COUNTRY_ENCODE")
	df_words_cn_dic_encoder.show()

	# 7. 高分编码问题
	# 以后在添加拼音等问题