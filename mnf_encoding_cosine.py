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
from math import isnan
import numpy


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


@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def mnf_index_word_cosine_similarity(o, v):
	frame = {
		"CLEANNING": o,
		"STANDARD": v
	}
	df = pd.DataFrame(frame)

	def array_to_vector(arr):
		idx = []
		values = []
		s = list(set(arr))
		s.sort()
		for item in s:
			if isnan(item):
				idx.append(5999)
				values.append(1)
				break
			else:
				idx.append(item)
				if item < 2000:
					values.append(2)
				elif (item >= 2000) & (item < 3000):
					values.append(10)
				else:
					values.append(1)

		return Vectors.sparse(6000, idx, values)


	def cosine_distance(u, v):
		u = u.toArray()
		v = v.toArray()
		return float(numpy.dot(u, v) / (sqrt(numpy.dot(u, u)) * sqrt(numpy.dot(v, v))))

	df["CLENNING_FEATURE"] = df["CLEANNING"].apply(lambda x: array_to_vector(x))
	df["STANDARD_FEATURE"] = df["STANDARD"].apply(lambda x: array_to_vector(x))
	df["RESULT"] = df.apply(lambda x: cosine_distance(x["CLENNING_FEATURE"], x["STANDARD_FEATURE"]), axis=1)
	return df["RESULT"]


if __name__ == '__main__':
	spark = prepare()

	df_cleanning = spark.read.parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/district/words-index")
	df_cleanning.show()
	print(df_cleanning.count())
	df_cleanning = df_cleanning.withColumn("COSINE_SIMILARITY", \
					mnf_index_word_cosine_similarity(df_cleanning.MANUFACTURER_NAME_CLEANNING_WORDS, df_cleanning.MANUFACTURER_NAME_STANDARD_WORDS))
	df_cleanning.where((df_cleanning.label == 1.0) & (df_cleanning.EFFTIVENESS_MANUFACTURER < 0.9) & (df_cleanning.COSINE_SIMILARITY > df_cleanning.EFFTIVENESS_MANUFACTURER)) \
		.select("MANUFACTURER_NAME", "MANUFACTURER_NAME_CLEANNING_WORDS", "MANUFACTURER_NAME_STANDARD", \
				"MANUFACTURER_NAME_STANDARD_WORDS", "EFFTIVENESS_MANUFACTURER", "COSINE_SIMILARITY").show(100)
