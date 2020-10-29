# -*- coding: utf-8 -*-
"""alfredyang@pharbers.com.
在没有答案的情况下预测
分三种情况：
	1. 机器判断一定匹配成功 【标准：similarity >= 5.5】
	2. 机器判断模糊匹配 【标准：5 <= similarity < 5.5】
	3. 机器判断无法匹配 【可能的情况：分子名模糊匹配那一步就直接被筛选掉了，或 similarity < 5】

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

error_match_path = "s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/qilu/0.0.2/result_analyse/error_match"
no_label_path = "s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/qilu/0.0.2/result_analyse/no_label"
accuracy_by_mole_path = "s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/qilu/0.0.2/result_analyse/accuracy_by_mole_path"

if __name__ == '__main__':
	spark = prepare()

	df_result = load_training_data(spark)
	df_result = similarity(df_result)
	# download_prod_standard(spark)
	df_result.show(5)
	rank1 = df_result.where(df_result.RANK == 1.0).count()
	print(rank1)
	
	# 0. check the result
	print("算法生成的similarity前五总数 = " + str(df_result.count()))
	
	# 1. count the total number of data
	total_count = df_result.groupBy("id").agg({"SIMILARITY": "first"}).count()
	print("数据总数 = " + str(total_count))
	
	# 2. 第一匹配similarity >= 5.8：
	df_result = df_result.withColumn("match_right_1", when((df_result.SIMILARITY >= 5.8) & (df_result.RANK == 1), 1.0).otherwise(0.0))
	machine_right_1_count = df_result.where(df_result.match_right_1 == 1.0).count()
	print("机器判断第一正确匹配数量 = " + str(machine_right_1_count))
	actual_right_1_count = df_result.where((df_result.match_right_1 == 1.0) & (df_result.label == 1.0)).count()
	print("实际第一正确匹配数量 = " + str(actual_right_1_count))
	print("第一正确匹配率 = " + str(actual_right_1_count / machine_right_1_count))
	
	
	# 3. 第一匹配similarity <= 4：
	df_result = df_result.withColumn("cannot_match", when((df_result.SIMILARITY <= 4) & (df_result.RANK == 1), 1.0).otherwise(0.0))
	cannot_match_count = df_result.where(df_result.cannot_match == 1.0).count()
	print()
	print("机器判断无法匹配数据的数量 = " + str(cannot_match_count))
	actual_cannot_match_count = df_result.where((df_result.cannot_match == 1.0) & (df_result.label == 0.0)).count()
	print("实际无 pack id 数量/第一步算法过滤掉了 = " + str(actual_cannot_match_count))
	print("判断正确匹配率 = " + str(actual_cannot_match_count / cannot_match_count))
	
	# 4. 模糊区域 4 <= similarity <= 5.8：
	df_result = df_result.withColumn("appro_match", when((df_result.SIMILARITY > 4) & (df_result.SIMILARITY < 5.8) & (df_result.RANK == 1), 1.0).otherwise(0.0))
	df_result = df_result.withColumn("appro_match_2", when((df_result.SIMILARITY > 4) & (df_result.SIMILARITY < 5.8) & (df_result.RANK == 2), 1.0).otherwise(0.0))
	df_result = df_result.withColumn("appro_match_3", when((df_result.SIMILARITY > 4) & (df_result.SIMILARITY < 5.8) & (df_result.RANK == 3), 1.0).otherwise(0.0))
	df_result = df_result.withColumn("appro_match_4", when((df_result.SIMILARITY > 4) & (df_result.SIMILARITY < 5.8) & (df_result.RANK == 4), 1.0).otherwise(0.0))
	df_result = df_result.withColumn("appro_match_5", when((df_result.SIMILARITY > 4) & (df_result.SIMILARITY < 5.8) & (df_result.RANK == 5), 1.0).otherwise(0.0))
	
	appro_match_count = df_result.where(df_result.appro_match == 1.0).count()
	print()
	print("机器判断模糊匹配数据的数量 = " + str(appro_match_count))
	print("这部分数据中：")
	df_result = hit_place_prediction(df_result, 1)
	positive_hit_1 = df_result.where((df_result.appro_match == 1.0) & (df_result.prediction_1 == df_result.label) & (df_result.label == 1.0)).count()
	print("第一正确匹配pack id 数量 = " + str(positive_hit_1))

	# 2.2  second hit
	df_result = hit_place_prediction(df_result, 2)
	positive_hit_2 = df_result.where((df_result.appro_match_2 == 1.0) & (df_result.prediction_2 == df_result.label) & (df_result.label == 1.0)).count()
	print("第二正确匹配pack id 数量 = " + str(positive_hit_2))

	# 2.3 third hit
	df_result = hit_place_prediction(df_result, 3)
	positive_hit_3 = df_result.where((df_result.appro_match_3 == 1.0) & (df_result.prediction_3 == df_result.label) & (df_result.label == 1.0)).count()
	print("第三正确匹配pack id 数量 = " + str(positive_hit_3))

	# 2.4 forth hit
	df_result = hit_place_prediction(df_result, 4)
	positive_hit_4 = df_result.where((df_result.appro_match_4 == 1.0) & (df_result.prediction_4 == df_result.label) & (df_result.label == 1.0)).count()
	print("第四正确匹配pack id 数量 = " + str(positive_hit_4))

	# 2.5 forth hit
	df_result = hit_place_prediction(df_result, 5)
	positive_hit_5 = df_result.where((df_result.appro_match_5 == 1.0) & (df_result.prediction_5 == df_result.label) & (df_result.label == 1.0)).count()
	print("第五正确匹配pack id 数量 = " + str(positive_hit_5))
	xixi1=df_result.toPandas()
	xixi1.to_excel('Pfizer_PFZ10_outlier.xlsx', index = False)
	
	

	
	



