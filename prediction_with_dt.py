# -*- coding: utf-8 -*-
"""alfredyang@pharbers.com.

"""

import os
from pyspark.sql import SparkSession
from dataparepare import *
from interfere import *
from pyspark.sql.types import *
from pyspark.sql.functions import desc
from pyspark.sql.functions import rank, lit, when, row_number
from pyspark.sql import Window
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.classification import MultilayerPerceptronClassificationModel
from pyspark.ml.classification import DecisionTreeClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import PipelineModel
from pdu_feature import similarity, hit_place_prediction, dosage_replace, prod_name_replace, pack_replace, mnf_encoding_index, mnf_encoding_cosine, mole_dosage_calculaltion
from pyspark.ml.feature import VectorAssembler


def prepare():
	os.environ["PYSPARK_PYTHON"] = "python3"
	# 读取s3桶中的数据
	spark = SparkSession.builder \
		.master("yarn") \
		.appName("CPA&GYC match refactor") \
		.config("spark.driver.memory", "1g") \
		.config("spark.executor.cores", "2") \
		.config("spark.executor.instances", "2") \
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
	df_result = load_training_data(spark)  # 进入清洗流程的所有数据
	df_validate = df_result #.select("id", "label", "features").orderBy("id")
	df_encode = load_word_dict_encode(spark) 
	df_all = load_split_data(spark)  # 带id的所有数据
	
	resultid = df_result.select("id").distinct()
	resultid_lst = resultid.toPandas()["id"].tolist()
	df_lost = df_all.where(~df_all.id.isin(resultid_lst))  # 第一步就丢失了的数据
	# df_lost.write.mode("overwrite").parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/qilu/0.0.10/lost")
	# print("丢失条目写入完成")

	# 2. load model
	model = PipelineModel.load("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/pfizer_model/0.0.4/model")

	# 3. compute accuracy on the test set
	predictions = model.transform(df_validate)
	evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
	accuracy = evaluator.evaluate(predictions)
	print("Test Error = %g " % (1.0 - accuracy))
	print("Test set accuracy = " + str(accuracy))

	# 4. Test with Pharbers defined methods
	result = predictions
	result_similarity = similarity(result)
	# result_similarity.write.mode("overwrite").parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/azsanofi/0.0.15/for_analysis7")
	# print("用于分析的的条目写入完成")
	result = result.withColumn("JACCARD_DISTANCE_MOLE_NAME", result.JACCARD_DISTANCE[0]) \
				.withColumn("JACCARD_DISTANCE_DOSAGE", result.JACCARD_DISTANCE[1]) \
				.drop("JACCARD_DISTANCE", "indexedFeatures").drop("rawPrediction", "probability")
	# result = result.where(result.PACK_ID_CHECK != "nan")
	ph_total = result.groupBy("id").agg({"prediction": "first", "label": "first"}).count()
	df_label = result.groupBy("id").agg({"label":"max", "prediction":"max"})
	
	# df_label.withColumnRenamed("first(label)", "")
	# machine_cannot = df_label.where(df_label["max(prediction)"] == 0).count()
	# print(machine_cannot)
	# machine_cannot_true = df_label.where(df_label["max(label)"] == 0)
	# machine_cannot_true.show(2)
	# print(machine_cannot_true.count())
	
	# cannot_id = machine_cannot_true.select("id").distinct().toPandas()["id"].tolist()
	# machine_cannot_true = result.where(result.id.isin(cannot_id))
	# machine_cannot_true.printSchema()
	# print(machine_cannot_true.count())
	# machine_cannot_true.write.mode("overwrite").parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/azsanofi/0.0.10/no_label")
	
	
	all_count = df_all.count()
	print("数据总数： " + str(all_count))
	print("进入匹配流程条目： " + str(ph_total))
	print("丢失条目： " + str(df_lost.count()))
	# result = result.where(result.PACK_ID_CHECK != "")
	# ph_total = result.groupBy("id").agg({"prediction": "first", "label": "first"}).count()
	# print("人工已匹配数据总数: " + str(ph_total))

	# 5. 尝试解决多高的问题
	df_true_positive = similarity(result.where(result.prediction == 1.0))
	df_true_positive = df_true_positive.where(df_true_positive.RANK == 1)
	machine_right_1 = df_true_positive
	# df_true_positive.write.mode("overwrite").parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/eia/0.0.2/machine_tp")
	# print("机器判断TP的条目写入完成")

	ph_positive_prodict = df_true_positive.count()
	print("机器判断第一轮TP条目 = " + str(ph_positive_prodict))
	ph_positive_hit = df_true_positive.where((result.prediction == result.label) & (result.label == 1.0)).count()
	print("其中正确条目 = " + str(ph_positive_hit))
	ph_tp_null_packid = df_true_positive.where(df_true_positive.PACK_ID_CHECK == "")
	# ph_negetive_hit = result.where(result.prediction != result.label).count()
	if ph_positive_prodict == 0:
		print("Pharbers Test set accuracy （机器判断第一轮T比例） = 无 ")
	else:
		print("Pharbers Test set accuracy （机器判断第一轮TP比例） = " + str(ph_positive_prodict / ph_total) + " / " + str(ph_positive_prodict / all_count))
		print("Pharbers Test set precision （机器判断第一轮TP正确率） = " + str(ph_positive_hit / ph_positive_prodict))
		print("人工没有匹配packid = " + str(ph_tp_null_packid.count()))


	# 6. 第二轮筛选TP
	df_true_positive = df_true_positive.select("id").distinct()
	id_local = df_true_positive.toPandas()["id"].tolist()  # list的内容是上一步确定TP的id
	
	df_candidate = result.where(~result.id.isin(id_local)) # df_candidate 是上一步选出的TP的剩下的数据，进行第二轮
	df_candidate = df_candidate.drop("prediction", "indexedLabel", "indexedFeatures", "rawPrediction", "probability", "features")
	# df_candidate = df_candidate.where(df_candidate.RANK<=3)
	# df_candidate.write.mode("overwrite").parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/chc/0.0.4/second_round")
	# print("第二轮写入完成")
	count_prediction_se = df_candidate.groupBy("id").agg({"label": "first"}).count()
	print("第二轮总量= " + str(count_prediction_se))
	model = PipelineModel.load("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/pfizer_model/0.0.4/model_without_prod")
	assembler = VectorAssembler( \
					inputCols=["EFFTIVENESS_MOLE_NAME", "EFFTIVENESS_DOSAGE", "EFFTIVENESS_SPEC",\
								"EFFTIVENESS_PACK_QTY", "EFFTIVENESS_MANUFACTURER"], \
					outputCol="features")
	df_candidate = assembler.transform(df_candidate)
	predictions = model.transform(df_candidate)
	evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
	accuracy = evaluator.evaluate(predictions)
	print("Test Error = %g " % (1.0 - accuracy))
	print("Test set accuracy = " + str(accuracy))
	
	df_true_positive_se = similarity(predictions.where((predictions.prediction == 1.0)))
	df_true_positive_se = df_true_positive_se.where(df_true_positive_se.RANK == 1)

	ph_positive_prodict = df_true_positive_se.count()
	print("机器判断第二轮TP条目 = " + str(ph_positive_prodict))
	ph_positive_hit = df_true_positive_se.where((df_true_positive_se.prediction == 1.0) & (df_true_positive_se.label == 1.0)).count()
	print("其中正确条目 = " + str(ph_positive_hit))
	# ph_tp_null_packid = df_true_positive.where(df_true_positive.PACK_ID_CHECK == "")
	if ph_positive_prodict == 0:
		print("Pharbers Test set accuracy （机器判断第一轮T比例） = 无 ")
	else:
		print("Pharbers Test set accuracy （机器判断第一轮TP比例） = " + str(ph_positive_prodict / ph_total) + " / " + str(ph_positive_prodict / all_count))
		print("Pharbers Test set precision （机器判断第一轮TP正确率） = " + str(ph_positive_hit / ph_positive_prodict))
		print("人工没有匹配packid = " + str(ph_tp_null_packid.count()))

	# 8. 第三轮
	df_prediction_se = df_true_positive_se.select("id").distinct()
	id_local_se = df_prediction_se.toPandas()["id"].tolist()
	id_local_total = id_local_se + id_local
	df_candidate_th = result.where(~result.id.isin(id_local_total))
	count_third = df_candidate_th.groupBy("id").agg({"prediction": "first", "label": "first"}).count()
	print("第三轮总量= " + str(count_third))
	df_candidate_th = similarity(df_candidate_th)
	# df_candidate_th.write.mode("overwrite").parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/qilu/0.0.8/no_pack_id")
	rank3_true = df_candidate_th.where((df_candidate_th.RANK <= 3) & (df_candidate_th.label == 1)).count()
	rank5_true = df_candidate_th.where((df_candidate_th.RANK <= 5) & (df_candidate_th.label == 1)).count()
	print("前三数量/正确率 = " + str(rank3_true) + " / " + str(rank3_true / count_third))
	print("前五数量/正确率 = " + str(rank5_true) + " / " + str(rank5_true / count_third))
	
	
	# 机器判断无法匹配
	# prediction_third_round = df_candidate_third.where(df_candidate_third.SIMILARITY > 3.0)
	# df_candidate_third.write.mode("overwrite").parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/third_round_1117qilu")
	# df_candidate_third.write.mode("overwrite").parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/third_round_1113az")
	# prediction_third_round = df_candidate_third.where(df_candidate_third.SIMILARITY > 4.0)
	# prediction_third_round.write.mode("overwrite").parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/third_round_4")
	# xixi1=df_candidate_third.toPandas()
	# xixi1.to_excel('df_candidate_third.xlsx', index = False)
	# prediction_third_round = df_candidate_third.where(df_candidate_third.RANK <= 5).groupBy("id").agg(count("label").alias("label_sum"))
	# ph_positive_predict_th = prediction_third_round.count()
	# ph_positive_hit_th =  prediction_third_round.where(prediction_third_round.label_sum != 0.0).count()
	# print("机器判断模糊数量= " + str(ph_positive_predict_th))
	# print("前五正确数量= " + str(ph_positive_hit_th))
	
	
	# 9. 最后一步，给出完全没匹配的结果