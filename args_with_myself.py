# -*- coding: utf-8 -*-
"""alfredyang@pharbers.com.

功能描述：job3：left join cpa和prod
  * @author yzy
  * @version 0.0
  * @since 2020/08/12
  * @note  落盘数据：cpa_prod_join

"""

import os
from pyspark.sql import SparkSession
from dataparepare import *
from interfere import *
from pdu_feature import *
from pyspark.sql.types import *
from pyspark.sql.functions import first
from pyspark.sql.functions import sum


def prepare():
	os.environ["PYSPARK_PYTHON"] = "python3"
	# 读取s3桶中的数据
	spark = SparkSession.builder \
		.master("yarn") \
		.appName("CPA&GYC match refactor") \
		.config("spark.driver.memory", "1g") \
		.config("spark.executor.cores", "2") \
		.config("spark.executor.instance", "4") \
		.config("spark.executor.memory", "2g") \
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

	return spark


if __name__ == '__main__':
	spark = prepare()

	df_result = load_training_data(spark)
	df_result = similarity(df_result)
	download_prod_standard(spark)

	# 0. check the result
	total_hit = df_result.where(df_result.label == 1.0).count()
	print("正确数据总数 = " + str(total_hit))

	# 1. count the total number of data
	total_count = df_result.groupBy("id").agg({"label": "first"}).count()
	print("数据总数 = " + str(total_count))

	# 2. count the right hit number
	# 2.1 first hit
	df_result = hit_place_prediction(df_result, 1)
	positive_hit_1 = df_result.where((df_result.prediction_1 == df_result.label) & (df_result.label == 1.0)).count()
	print("第一正确匹配pack id 数量 = " + str(positive_hit_1))

	# 2.2  second hit
	df_result = hit_place_prediction(df_result, 2)
	positive_hit_2 = df_result.where((df_result.prediction_2 == df_result.label) & (df_result.label == 1.0)).count()
	print("第二正确匹配pack id 数量 = " + str(positive_hit_2))

	# 2.3 third hit
	df_result = hit_place_prediction(df_result, 3)
	positive_hit_3 = df_result.where((df_result.prediction_3 == df_result.label) & (df_result.label == 1.0)).count()
	print("第三正确匹配pack id 数量 = " + str(positive_hit_3))

	# 2.4 forth hit
	df_result = hit_place_prediction(df_result, 4)
	positive_hit_4 = df_result.where((df_result.prediction_4 == df_result.label) & (df_result.label == 1.0)).count()
	print("第四正确匹配pack id 数量 = " + str(positive_hit_4))

	# 2.5 forth hit
	df_result = hit_place_prediction(df_result, 5)
	positive_hit_5 = df_result.where((df_result.prediction_5 == df_result.label) & (df_result.label == 1.0)).count()
	print("第五正确匹配pack id 数量 = " + str(positive_hit_5))

	# 2.6 count the accurcy of right hit number
	print("正确匹配pack id 率 = " + str((positive_hit_1 + positive_hit_2 + positive_hit_3 + positive_hit_4 + positive_hit_5) / total_count))
	print("在有pack id的数据中，正确匹配pack id 率 = " + str((positive_hit_1 + positive_hit_2 + positive_hit_3 + positive_hit_4 + positive_hit_5) / total_hit))


	# 3. not_match label & record the validata dataaa
	# df_result.printSchema()
	df_result = df_result.withColumn("prediction", df_result.prediction_1 + df_result.prediction_2 + df_result.prediction_3 + df_result.prediction_4 + df_result.prediction_5)
	df_no_label = df_result.groupBy("id")\
			.agg(sum(df_result.label).alias("label"),\
				first(df_result.MOLE_NAME).alias("MOLE_NAME"), \
				first(df_result.PRODUCT_NAME).alias("PRODUCT_NAME"), \
				first(df_result.DOSAGE).alias("DOSAGE"), \
				first(df_result.SPEC).alias("SPEC"), \
				first(df_result.PACK_QTY).alias("PACK_QTY"), \
				first(df_result.MANUFACTURER_NAME).alias("MANUFACTURER_NAME"), \
				first(df_result.PACK_ID_CHECK).alias("PACK_ID_CHECK")
			)
	df_result = df_result.where((df_result.prediction == 0) & (df_result.label == 1.0))
	# 3.1 完全匹配错误的数据
	print("算法前五没有匹配的数据 = " + str(df_result.count()))
	df_result = df_result.drop("prediction", "prediction_1", "prediction_2", "prediction_3", "prediction_4", "prediction_5").drop("JACCARD_DISTANCE", "features")
	df_result.orderBy("id", "RANK").repartition(1).write.mode("overwrite").option("header", "true").csv("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/validate/error_match")

	# 3.2 本身就没有pack id的数据
	df_no_label = df_no_label.where(df_no_label.label == 0.0)
	print("本身没有label的数据 = " + str(df_no_label.count()))
	df_no_label.orderBy("id").repartition(1).write.mode("overwrite").option("header", "true").csv("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/validate/error_label")
