# -*- coding: utf-8 -*-
"""alfredyang@pharbers.com.

"""

import os
from pyspark.sql import SparkSession
from dataparepare import *
from interfere import *
from pyspark.sql.types import *
from pyspark.sql.functions import desc
from pyspark.sql.functions import rank, lit, when
from pyspark.sql import Window
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.classification import MultilayerPerceptronClassificationModel
from pyspark.ml.classification import DecisionTreeClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import PipelineModel
from pdu_feature import similarity, hit_place_prediction, dosage_replace, prod_name_replace, pack_replace, mnf_encoding_index, mnf_encoding_cosine
from pyspark.ml.feature import VectorAssembler


def prepare():
	os.environ["PYSPARK_PYTHON"] = "python3"
	# 读取s3桶中的数据
	spark = SparkSession.builder \
		.master("yarn") \
		.appName("CPA&GYC match refactor") \
		.config("spark.driver.memory", "1g") \
		.config("spark.executor.cores", "2") \
		.config("spark.executor.instances", "4") \
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

	# 1. load the data
	df_result = load_training_data(spark)  # 待清洗数据
	# df_result.printSchema()
	df_validate = df_result #.select("id", "label", "features").orderBy("id")
	df_encode = load_word_dict_encode(spark) 

	# 2. load model
	# model = PipelineModel.load("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/dt")
	model = PipelineModel.load("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/pfizer_model/0.0.1/model")

	# 3. compute accuracy on the test set
	predictions = model.transform(df_validate)
	
	# predictions.where((predictions.label == 1.0) & (predictions.prediction == 0.0)).show(truncate=False)
	# predictions.where((predictions.label == 1.0) & (predictions.prediction == 0.0)).select("id", "label", "probability", "prediction").show(truncate=False)
	# predictions.where((predictions.label == 1.0) & (predictions.prediction == 1.0)).select("id", "label", "probability", "prediction").show(truncate=False)
	evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
	accuracy = evaluator.evaluate(predictions)
	print("Test Error = %g " % (1.0 - accuracy))
	print("Test set accuracy = " + str(accuracy))

	# 4. Test with Pharbers defined methods
	result = predictions
	# result.printSchema()
	result = result.withColumn("JACCARD_DISTANCE_MOLE_NAME", result.JACCARD_DISTANCE[0]) \
				.withColumn("JACCARD_DISTANCE_DOSAGE", result.JACCARD_DISTANCE[1]) \
				.drop("JACCARD_DISTANCE", "features", "indexedFeatures").drop("rawPrediction", "probability")
	# result.orderBy("id").repartition(1).write.mode("overwrite").csv("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/result")
	df_ph = result.where((result.prediction == 1.0) | (result.label == 1.0))
	ph_total = result.groupBy("id").agg({"prediction": "first", "label": "first"}).count()
	print("数据总数： " + str(ph_total))
	# result = result.where(result.PACK_ID_CHECK != "")
	# ph_total = result.groupBy("id").agg({"prediction": "first", "label": "first"}).count()
	# print("人工已匹配数据总数: " + str(ph_total))

	# 5. 尝试解决多高的问题
	df_true_positive = similarity(result.where(result.prediction == 1.0))
	df_true_positive = df_true_positive.where(df_true_positive.RANK == 1)
	machine_right_1 = df_true_positive

	ph_positive_prodict = df_true_positive.count()
	print("机器判断第一轮TP条目 = " + str(ph_positive_prodict))
	ph_positive_hit = df_true_positive.where((result.prediction == result.label) & (result.label == 1.0)).count()
	print("其中正确条目 = " + str(ph_positive_hit))
	ph_tp_null_packid = df_true_positive.where(df_true_positive.PACK_ID_CHECK == "")
	# ph_negetive_hit = result.where(result.prediction != result.label).count()
	if ph_positive_prodict == 0:
		print("Pharbers Test set accuracy （机器判断第一轮T比例） = 无 ")
	else:
		print("Pharbers Test set accuracy （机器判断第一轮TP比例） = " + str(ph_positive_prodict / ph_total))
		print("Pharbers Test set precision （机器判断第一轮TP正确率） = " + str(ph_positive_hit / ph_positive_prodict))
		print("人工没有匹配packid = " + str(ph_tp_null_packid.count()))

	
	# for analysis
	# df_true_positive.orderBy("id").repartition(1) \
	# 	.where((result.prediction == 0.0) & (result.label == 1.0)) \
	# 	.write.mode("overwrite").option("header", "true").csv("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/dt_predictions/false_negative")
	# df_true_positive.orderBy("id").repartition(1) \
	# 	.where((result.prediction == 1.0) & (result.label == 0.0)) \
	# 	.write.mode("overwrite").option("header", "true").csv("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/dt_predictions/false_positive")
	# for output
	# df_true_positive.orderBy("id").repartition(1) \
	# 	.where((result.prediction == 1.0) & (result.label == 0.0)) \
	# 	.write.mode("overwrite").option("header", "true").csv("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/dt_predictions/prediction")\
	# 
	# df_pre1_label0 = df_true_positive.where((df_true_positive.prediction == 1.0) & (df_true_positive.label == 0.0))
	# xixi1=df_pre1_label0.toPandas()
	# xixi1.to_excel('round1_pre1_label0.xlsx', index = False)

	# 6. 第二轮筛选TP
	df_true_positive = df_true_positive.select("id").distinct()
	id_local = df_true_positive.toPandas()["id"].tolist()  # list的内容是上一步确定TP的id

	df_candidate = result.where(~result.id.isin(id_local)) # df_candidate 是上一步选出的TP的剩下的数据，进行第二轮
	count_prediction_se = df_candidate.groupBy("id").agg({"prediction": "first", "label": "first"}).count()
	print("第二轮总量= " + str(count_prediction_se))
	
	# 第二轮筛选的方法是：再对dosage等列重新计算eff，要用到dosage_mapping
	df_second_round = df_candidate.drop("prediction", "indexedLabel", "indexedFeatures", "rawPrediction", "probability", "features")
	dosage_mapping = load_dosage_mapping(spark)
	df_second_round = df_second_round.join(dosage_mapping, df_second_round.DOSAGE == dosage_mapping.CPA_DOSAGE, how="left").na.fill("")
	df_second_round = df_second_round.withColumn("EFFTIVENESS_DOSAGE_SE", dosage_replace(df_second_round.MASTER_DOSAGE, \
														df_second_round.DOSAGE_STANDARD, df_second_round.EFFTIVENESS_DOSAGE)) 
	
	df_second_round = df_second_round.withColumn("EFFTIVENESS_PACK_QTY_SE", pack_replace(df_second_round.EFFTIVENESS_PACK_QTY, df_second_round.SPEC_ORIGINAL, \
														df_second_round.PACK_QTY, df_second_round.PACK_QTY_STANDARD))
	df_second_round = mnf_encoding_index(df_second_round, df_encode)
	df_second_round = mnf_encoding_cosine(df_second_round)
	df_second_round = df_second_round.withColumn("EFFTIVENESS_MANUFACTURER_SE", \
										when(df_second_round.COSINE_SIMILARITY >= df_second_round.EFFTIVENESS_MANUFACTURER, df_second_round.COSINE_SIMILARITY) \
										.otherwise(df_second_round.EFFTIVENESS_MANUFACTURER))
	df_second_round = df_second_round.withColumn("EFFTIVENESS_PRODUCT_NAME_SE", \
								prod_name_replace(df_second_round.EFFTIVENESS_MOLE_NAME, df_second_round.EFFTIVENESS_MANUFACTURER_SE, df_second_round.EFFTIVENESS_PRODUCT_NAME))
	
	
	assembler = VectorAssembler( \
				inputCols=["EFFTIVENESS_MOLE_NAME", "EFFTIVENESS_PRODUCT_NAME_SE", "EFFTIVENESS_DOSAGE_SE", "EFFTIVENESS_SPEC", \
							"EFFTIVENESS_PACK_QTY_SE", "EFFTIVENESS_MANUFACTURER_SE"], \
				outputCol="features")
	df_second_round = assembler.transform(df_second_round)
	# df_second_round.repartition(10).write.mode("overwrite").parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/second_round_dt")
	
	predictions_second_round = model.transform(df_second_round)
	# predictions_second_round.write.mode("overwrite").parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/second_round_prediction_1")
	evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
	accuracy = evaluator.evaluate(predictions_second_round)
	print("Test Error = %g " % (1.0 - accuracy))
	print("Test set accuracy = " + str(accuracy))
	
	# xixi1=predictions_second_round.toPandas()
	# xixi1.to_excel('predictions_second_round_mnf.xlsx', index = False)
	
	# 第二轮正确率检测
	df_true_positive_se = predictions_second_round.where(predictions_second_round.prediction == 1.0)
	machine_right_2 = df_true_positive_se
	ph_positive_prodict_se = df_true_positive_se.count()
	print("机器判断第二轮TP条目 = " + str(ph_positive_prodict_se))
	ph_positive_hit_se = df_true_positive_se.where((df_true_positive_se.prediction == df_true_positive_se.label) & (df_true_positive_se.label == 1.0)).count()
	print("其中正确条目 = " + str(ph_positive_hit_se))
	ph_tp_null_packid_se = df_true_positive_se.where(df_true_positive_se.PACK_ID_CHECK == "")
	# ph_negetive_hit = result.where(result.prediction != result.label).count()
	if ph_positive_prodict_se == 0:
		print("Pharbers Test set accuracy （机器判断第二轮TP比例） = 无" )
	else:
		print("Pharbers Test set accuracy （机器判断第二轮TP比例） = " + str(ph_positive_prodict_se / ph_total))
		print("Pharbers Test set precision （机器判断第二轮TP正确率） = " + str(ph_positive_hit_se / ph_positive_prodict_se))
		print("人工没有匹配packid = " + str(ph_tp_null_packid_se.count()))
	
	# for analyses
	# df_pre1_label0 = df_true_positive_se.where((df_true_positive_se.prediction == 1.0) & (df_true_positive_se.label == 0.0))
	# xixi1=df_pre1_label0.toPandas()
	# xixi1.to_excel('round2_pre1_label0.xlsx', index = False)
	
	# 两轮合并整理成表格
	# machine_right_1 = machine_right_1.drop("indexedLabel", "JACCARD_DISTANCE_DOSAGE", "JACCARD_DISTANCE_MOLE_NAME", "SIMILARITY", "RANK")
	# machine_right_2 = machine_right_2.drop("indexedLabel", "JACCARD_DISTANCE_DOSAGE", "JACCARD_DISTANCE_MOLE_NAME", "SIMILARITY", "RANK", "features", \
	# 								"EFFTIVENESS_PRODUCT_NAME_SE", "EFFTIVENESS_DOSAGE_SE", "EFFTIVENESS_PACK_QTY_SE", "indexedFeatures", "rawPrediction", "probability")
	# machine_right_1.printSchema()
	# machine_right_2.printSchema()
	# machine_right = machine_right_1.unionByName(machine_right_2)
	# xixi1=machine_right.toPandas()
	# xixi1.to_excel('pf_true_positive.xlsx', index = False)

	# 7. 两轮估算总量
	ph_positive_prodict = ph_positive_prodict + ph_positive_prodict_se
	print("两轮判断TP总量 = " + str(ph_positive_prodict))
	ph_positive_hit = ph_positive_hit + ph_positive_hit_se
	print("两轮判断TP总正确数量 = " + str(ph_positive_hit))
	# ph_negetive_hit = result.where(result.prediction != result.label).count()
	if ph_positive_prodict_se == 0:
		print("Pharbers Test set accuracy （两轮判断TP总比例） = " + str(ph_positive_prodict / ph_total))
		print("Pharbers Test set precision （两轮判断TP总正确率）= 0")
	else:
		print("Pharbers Test set accuracy （两轮判断TP总比例） = " + str(ph_positive_prodict / ph_total))
		print("Pharbers Test set precision （两轮判断TP总正确率）= " + str(ph_positive_hit / ph_positive_prodict))
		print("人工未匹配总量" + str(ph_tp_null_packid_se.count()+ph_tp_null_packid.count()))
	

	# 8. 第三轮
	df_prediction_se = df_true_positive_se.select("id").distinct()
	id_local_se = df_prediction_se.toPandas()["id"].tolist()
	id_local_total = id_local_se + id_local
	predictions_second_round = predictions_second_round.drop("EFFTIVENESS_PRODUCT_NAME", "EFFTIVENESS_DOSAGE", "EFFTIVENESS_PACK_QTY", "EFFTIVENESS_MANUFACTURER")
	df_candidate_third = predictions_second_round.where(~result.id.isin(id_local_total)).withColumnRenamed("EFFTIVENESS_PRODUCT_NAME_SE", "EFFTIVENESS_PRODUCT_NAME") \
																.withColumnRenamed("EFFTIVENESS_DOSAGE_SE", "EFFTIVENESS_DOSAGE") \
																.withColumnRenamed("EFFTIVENESS_MANUFACTURER_SE", "EFFTIVENESS_MANUFACTURER") \
																.withColumnRenamed("EFFTIVENESS_PACK_QTY_SE", "EFFTIVENESS_PACK_QTY")
	count_third = df_candidate_third.groupBy("id").agg({"prediction": "first", "label": "first"}).count()
	print("第三轮总量= " + str(count_third))
	df_candidate_third = similarity(df_candidate_third)
	
	# 机器判断无法匹配
	# prediction_third_round = df_candidate_third.where(df_candidate_third.SIMILARITY > 3.0)
	df_candidate_third.write.mode("overwrite").parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/third_round_1113qilu2")
	# df_candidate_third.write.mode("overwrite").parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/third_round_1113az")
	# prediction_third_round = df_candidate_third.where(df_candidate_third.SIMILARITY > 4.0)
	# prediction_third_round.write.mode("overwrite").parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/third_round_4")
	# xixi1=df_candidate_third.toPandas()
	# xixi1.to_excel('df_candidate_third.xlsx', index = False)
	prediction_third_round = df_candidate_third.where(df_candidate_third.RANK <= 5).groupBy("id").agg(count("label").alias("label_sum"))
	ph_positive_predict_th = prediction_third_round.count()
	ph_positive_hit_th =  prediction_third_round.where(prediction_third_round.label_sum != 0.0).count()
	print("机器判断模糊数量= " + str(ph_positive_predict_th))
	print("前五正确数量= " + str(ph_positive_hit_th))
	
	
	# 9. 最后一步，给出完全没匹配的结果