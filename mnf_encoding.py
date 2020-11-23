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


@pandas_udf(StringType(), PandasUDFType.SCALAR)
def convert_array_words_to_string(arr):
	frame = {
		"ARRARY": arr
	}
	df = pd.DataFrame(frame)

	df["RESULT"] = df["ARRARY"].apply(lambda x: ",".join(x.tolist()))
	return df["RESULT"]


if __name__ == '__main__':
	spark = prepare()

	# 1. 利用standard中的中文列通过中文分词
	df_standard = load_standard_prod(spark)
	df_words_cn = phcleanning_mnf_seg(df_standard, "MANUFACTURER_NAME_STANDARD", "MANUFACTURER_NAME_WORDS_FILTER")

	df_words_out = df_words_cn.groupBy("MANUFACTURER_NAME_STANDARD").agg(first(df_words_cn.MANUFACTURER_NAME_WORDS_FILTER).alias("WORDS"))
	df_words_out = df_words_out.withColumn("WORDS", convert_array_words_to_string(df_words_out.WORDS))
	# df_words_out.repartition(1).write.mode("overwrite").option("header", "true").csv("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/district/split-words-out")
	df_words_out.show()

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
	df_geo_standard = df_geo_standard.select("ID", "NAME", "PARENTID")  # 所有地理维度的词
	print("df_geo_standard:")
	df_geo_standard.show()

	# 5.2 优先搞定省市问题
	# df_words_cn_dic_encoder = df_words_cn_dic.where(df_words_cn_dic.GEO_TAG == 1)
	df_words_cn_dic_encoder = df_words_cn_dic
	df_words_cn_dic_encoder = df_words_cn_dic_encoder.join(df_geo_standard, df_words_cn_dic_encoder.WORD == df_geo_standard.NAME, how="left")
	df_words_cn_dic_encoder = df_words_cn_dic_encoder.na.fill({"ID": "-1", "NAME": "", "PARENTID": "-1"})
	df_words_cn_dic_encoder = df_words_cn_dic_encoder.select("WORD", df_words_cn_dic_encoder.ID.alias("AREACODE"), df_words_cn_dic_encoder.PARENTID.alias("AREA_PARENT_ID"))
	
	print("df_words_cn_dic_encoder:")
	df_words_cn_dic_encoder.show()
	
	count = df_words_cn_dic_encoder.where(~(df_words_cn_dic_encoder.AREA_PARENT_ID == "0") & ~(df_words_cn_dic_encoder.AREA_PARENT_ID == "-1")).count()
	while count > 0:
		df_words_cn_dic_encoder = df_words_cn_dic_encoder.join(df_geo_standard, df_words_cn_dic_encoder.AREA_PARENT_ID == df_geo_standard.ID, how="left")
		df_words_cn_dic_encoder = df_words_cn_dic_encoder.na.fill({"ID": "-1", "NAME": "", "PARENTID": "-1"})
		df_words_cn_dic_encoder.show()
		df_words_cn_dic_encoder = df_words_cn_dic_encoder.withColumn("AREACODE", when(~(df_words_cn_dic_encoder.AREA_PARENT_ID == "0"), df_words_cn_dic_encoder.ID).otherwise(df_words_cn_dic_encoder.AREACODE)) \
																.withColumn("AREA_PARENT_ID", when(~(df_words_cn_dic_encoder.AREA_PARENT_ID == "0"), df_words_cn_dic_encoder.PARENTID).otherwise(df_words_cn_dic_encoder.AREA_PARENT_ID)) \
																.select("WORD", "AREACODE", "AREA_PARENT_ID")
																# .select("WORD", "AREACODE", "AREA_PARENT_ID", "NAME")
		count = df_words_cn_dic_encoder.where(~(df_words_cn_dic_encoder.AREA_PARENT_ID == "0") & ~(df_words_cn_dic_encoder.AREA_PARENT_ID == "-1")).count()

	df_words_cn_dic_encoder = df_words_cn_dic_encoder.withColumn("GEO_ENCODE", df_words_cn_dic_encoder.AREACODE.cast(IntegerType()))
	df_words_cn_dic_encoder = df_words_cn_dic_encoder.withColumn("GEO_ENCODE", when(df_words_cn_dic_encoder.GEO_ENCODE > 0, 1000 + df_words_cn_dic_encoder.GEO_ENCODE).otherwise(-1)) \
														.select("WORD", "GEO_ENCODE")
	print("df_words_cn_dic_encoder最终：")
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
	# 去掉区和省名相同的情况
	df_words_cn_dic_encoder = df_words_cn_dic_encoder.groupBy("WORD") \
								.agg(min(df_words_cn_dic_encoder.GEO_ENCODE).alias("GEO_ENCODE"), \
									min(df_words_cn_dic_encoder.COUNTRY_ENCODE).alias("COUNTRY_ENCODE"))
	# df_words_cn_dic_encoder.repartition(1).write.mode("overwrite").csv("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/district/words")

	# 7.1 高分词
	df_high_scroe_word = spark.read.parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/high_word_seg")
	# df_high_scroe_word.show(truncate=False)

	df_words_cn_dic_encoder = df_words_cn_dic_encoder.join(df_high_scroe_word, df_words_cn_dic_encoder.WORD == df_high_scroe_word.HIGH_WORD, how="left").na.fill("-1.0")
	df_words_cn_dic_encoder = df_words_cn_dic_encoder.withColumn("HIGH_WORD_ENCODE", 2001 + df_words_cn_dic_encoder.ID.cast(IntegerType())).na.fill(-1.0)
	df_words_cn_dic_encoder = df_words_cn_dic_encoder.select("WORD", "GEO_ENCODE", "COUNTRY_ENCODE", "HIGH_WORD_ENCODE")

	# 8. 其它分词
	df_words_cn_dic_encoder = df_words_cn_dic_encoder.repartition(1).withColumn("OTHER_WORD_ENCODE", 5001 + monotonically_increasing_id()).na.fill(-1.0)

	# 9. 从所有的编码中，选出所需的
	df_words_cn_dic_encoder = df_words_cn_dic_encoder.repartition(8).withColumn("WORD_ENCODE", \
								when(df_words_cn_dic_encoder.GEO_ENCODE > 0, df_words_cn_dic_encoder.GEO_ENCODE).otherwise( \
									when(df_words_cn_dic_encoder.COUNTRY_ENCODE > 0, df_words_cn_dic_encoder.COUNTRY_ENCODE).otherwise( \
										when(df_words_cn_dic_encoder.HIGH_WORD_ENCODE > 0, df_words_cn_dic_encoder.HIGH_WORD_ENCODE).otherwise( \
											df_words_cn_dic_encoder.OTHER_WORD_ENCODE)))).na.fill(-1.0).select("WORD", "WORD_ENCODE")
	df_words_cn_dic_encoder = df_words_cn_dic_encoder.withColumnRenamed("WORD_ENCODE", "ENCODE")
	df_words_cn_dic_encoder.repartition(1).write.mode("overwrite").parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/word_dict/0.0.4")
	df_words_cn_dic_encoder.show()