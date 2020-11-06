# -*- coding: utf-8 -*-
"""alfredyang@pharbers.com.

功能描述：TF-IDF for manifacture name

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
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd
import numpy
from math import sqrt
import jieba
import jieba.posseg as pseg
import jieba.analyse as analyse
# from nltk.cluster.util import cosine_distance
# from pyspark.ml.functions import vector_to_array


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


if __name__ == '__main__':
	spark = prepare()

	# 1. 利用standard中的中文列通过中文分词
	df_standard = load_standard_prod(spark)
	df_standard = df_standard.withColumn("MANUFACTURER_NAME_EN_STANDARD", manifacture_name_en_standify(df_standard.MANUFACTURER_NAME_EN_STANDARD))
	df_standard.select("MANUFACTURER_NAME_STANDARD", "MANUFACTURER_NAME_EN_STANDARD").show(truncate=False)

	# 2. 英文的分词方法，tokenizer
	tokenizer = Tokenizer(inputCol="MANUFACTURER_NAME_EN_STANDARD", outputCol="MANUFACTURER_NAME_EN_WORDS")
	df_standard = tokenizer.transform(df_standard)

	# 3. 中文的分词，jieba
	df_standard = df_standard.withColumn("MANUFACTURER_NAME_WORDS", manifacture_name_pseg_cut(df_standard.MANUFACTURER_NAME_STANDARD))
	df_standard.select("MANUFACTURER_NAME_STANDARD", "MANUFACTURER_NAME_WORDS", "MANUFACTURER_NAME_EN_STANDARD", "MANUFACTURER_NAME_EN_WORDS").show(truncate=False)

	# 4. 构建机器学习的feature
	hashingTF_en = HashingTF(inputCol="MANUFACTURER_NAME_EN_WORDS", outputCol="raw_features_mnf_en", numFeatures=1000)
	man_en_idf = IDF(inputCol="raw_features_mnf_en", outputCol="features_mnf_en")

	hashingTF_cn = HashingTF(inputCol="MANUFACTURER_NAME_WORDS", outputCol="raw_features_mnf_cn", numFeatures=1000)
	man_cn_idf = IDF(inputCol="raw_features_mnf_cn", outputCol="features_mnf_cn")

	pipeline = Pipeline(stages=[hashingTF_en, man_en_idf, hashingTF_cn, man_cn_idf])
	idf_model = pipeline.fit(df_standard)
	idf_model.write().overwrite().save("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/idf_model")

	# df_standard = idf_model.transform(df_standard)
	# df_standard.show()

	# --------- 从这里开始应该重启一个job 我偷懒没有写 ------------

	# 5. 利用原有的crossjoin 数据构建，公司的label
	df_result = load_training_data(spark)  # 待清洗数据
	df_result = df_result.withColumn("MANUFACTURER_NAME_EN_STANDARD", manifacture_name_en_standify(df_result.MANUFACTURER_NAME_EN_STANDARD))

	# 为生产厂商构建新label
	# df_mnf_label = df_result.where(df_result.label == 1.0).select("id", "MANUFACTURER_NAME", "MANUFACTURER_NAME_STANDARD","MANUFACTURER_NAME_EN_STANDARD")
	# df_mnf_label = df_mnf_label.groupBy("id").agg(first(df_mnf_label.MANUFACTURER_NAME).alias("MANUFACTURER_NAME_ANSWER"), \
	# 												first(df_mnf_label.MANUFACTURER_NAME_STANDARD).alias("MANUFACTURER_NAME_STANDARD_ANSWER"), \
	# 												first(df_mnf_label.MANUFACTURER_NAME_EN_STANDARD).alias("MANUFACTURER_NAME_EN_STANDARD_ANSWER"))
	# df_result = df_result.join(df_mnf_label, how="left", on="id")
	# df_result = df_result.withColumn("mnf_label", when(
	# 												(df_result.MANUFACTURER_NAME_STANDARD == df_result.MANUFACTURER_NAME_STANDARD_ANSWER) |
	# 												(df_result.MANUFACTURER_NAME_EN_STANDARD == df_result.MANUFACTURER_NAME_EN_STANDARD_ANSWER), 1.0).otherwise(0.0))
	# # df_result.select("id", "MANUFACTURER_NAME", "MANUFACTURER_NAME_STANDARD","MANUFACTURER_NAME_STANDARD_ANSWER", "mnf_label").show()
	# df_result = df_result.drop("MANUFACTURER_NAME_ANSWER", "MANUFACTURER_NAME_STANDARD_ANSWER", "MANUFACTURER_NAME_EN_STANDARD_ANSWER")

	# 5.1 split the words
	# tokenizer = Tokenizer(inputCol="MANUFACTURER_NAME_EN_STANDARD", outputCol="MANUFACTURER_NAME_EN_WORDS")
	df_result = tokenizer.transform(df_result)

	# 5.2. 中文的分词，jieba
	df_result = df_result.withColumn("MANUFACTURER_NAME_WORDS", manifacture_name_pseg_cut(df_result.MANUFACTURER_NAME_STANDARD))
	df_result = idf_model.transform(df_result)
	df_result = df_result.withColumnRenamed("features_mnf_cn", "features_mnf_cn_standard")
	# df_result.select("MANUFACTURER_NAME_STANDARD", "MANUFACTURER_NAME_WORDS", "MANUFACTURER_NAME_EN_STANDARD", "MANUFACTURER_NAME_EN_WORDS").show(truncate=False)

	# 5.3. 带匹配的分词
	df_result = df_result.withColumn("MANUFACTURER_NAME_WORDS", manifacture_name_pseg_cut(df_result.MANUFACTURER_NAME))
	df_result = df_result.drop("raw_features_mnf_en", "raw_features_mnf_cn", "features_mnf_en", "features_mnf_cn")
	df_result = idf_model.transform(df_result)
	df_result = df_result.drop("raw_features_mnf_en", "raw_features_mnf_cn", "features_mnf_en")
	df_result.show(truncate=False)
	# df_result = df_result.withColumn("features_mnf_cn_standard", df_result.features_mnf_cn_standard.toArray()) \
							# .withColumn("features_mnf_cn", df_result.features_mnf_cn.toArray())

	# 6. 直接是用cosine similarity 不要用机器学习 多此一举
	df_result = df_result.limit(100).withColumn("CONSINE_SIMILARITY", cosine_distance_between_mnf(array(df_result.features_mnf_cn, df_result.features_mnf_cn_standard)))
	df_result.select("id", "label", "MANUFACTURER_NAME", "MANUFACTURER_NAME_STANDARD", "CONSINE_SIMILARITY", "EFFTIVENESS_MANUFACTURER").show()
	df_result.select("id", "label", "MANUFACTURER_NAME", "MANUFACTURER_NAME_STANDARD", "EFFTIVENESS_MANUFACTURER", "features_mnf_cn_standard", "features_mnf_cn").show(truncate=False)

