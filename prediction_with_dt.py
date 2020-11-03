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
from pyspark.sql.types import *
from pyspark.sql.functions import desc
from pyspark.sql.functions import rank
from pyspark.sql import Window
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.classification import MultilayerPerceptronClassificationModel
from pyspark.ml.classification import DecisionTreeClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import PipelineModel
from pdu_feature import similarity, hit_place_prediction


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
	df_result = load_training_data(spark)
	df_result.printSchema()
	df_validate = df_result #.select("id", "label", "features").orderBy("id")

	# 2. load model
	model = PipelineModel.load("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/dt")

	# 3. compute accuracy on the test set
	predictions = model.transform(df_validate)
	predictions.where((predictions.label == 1.0) & (predictions.prediction == 0.0)).show(truncate=False)
	# predictions.where((predictions.label == 1.0) & (predictions.prediction == 0.0)).select("id", "label", "probability", "prediction").show(truncate=False)
	# predictions.where((predictions.label == 1.0) & (predictions.prediction == 1.0)).select("id", "label", "probability", "prediction").show(truncate=False)
	evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
	accuracy = evaluator.evaluate(predictions)
	print("Test Error = %g " % (1.0 - accuracy))
	print("Test set accuracy = " + str(accuracy))

	# 4. Test with Pharbers defined methods
	result = predictions
	result.printSchema()
	result = result.withColumn("JACCARD_DISTANCE_MOLE_NAME", result.JACCARD_DISTANCE[0]) \
				.withColumn("JACCARD_DISTANCE_DOSAGE", result.JACCARD_DISTANCE[1]) \
				.drop("JACCARD_DISTANCE", "features", "indexedFeatures").drop("rawPrediction", "probability")
	# result.orderBy("id").repartition(1).write.mode("overwrite").csv("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/result")
	df_ph = result.where((result.prediction == 1.0) | (result.label == 1.0))
	ph_total = result.groupBy("id").agg({"prediction": "first", "label": "first"}).count()
	print(ph_total)

	# 5. 尝试解决多高的问题
	df_true_positive = similarity(result.where(result.prediction == 1.0))
	df_true_positive = df_true_positive.where(df_true_positive.RANK == 1)

	ph_positive_prodict = df_true_positive.count()
	print(ph_positive_prodict)
	ph_positive_hit = result.where((result.prediction == result.label) & (result.label == 1.0)).count()
	print(ph_positive_hit)
	# ph_negetive_hit = result.where(result.prediction != result.label).count()
	print("Pharbers Test set accuracy = " + str(ph_positive_hit / ph_total))
	print("Pharbers Test set precision = " + str(ph_positive_hit / ph_positive_prodict))

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
	# 	.write.mode("overwrite").option("header", "true").csv("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/dt_predictions/prediction")

	# 6. 在未匹配数据中，在剩下的数据中，机器给出建议
	df_true_positive = df_true_positive.select("id").distinct()
	id_local = df_true_positive.toPandas()["id"].tolist()

	df_candidate = result.where(~result.id.isin(id_local))
	df_candidate = similarity(df_candidate)

	df_prediction_se = df_candidate.where((df_candidate.SIMILARITY > 5.0) & (df_candidate.RANK == 1))
	count_prediction_se = df_prediction_se.count()
	print("第二轮总量= " + str(count_prediction_se))

	df_prediction_se = hit_place_prediction(df_prediction_se, 1)
	df_prediction_se_1_tp = df_prediction_se.where((df_prediction_se.prediction_1 == 1.0) & (df_prediction_se.prediction_1 == 1.0))
	df_prediction_se_1_fp = df_prediction_se.where((df_prediction_se.prediction_1 == 1.0) & (df_prediction_se.label == 0.0))
	positive_hit_1_tp = df_prediction_se_1_tp.count()
	positive_hit_1_fp = df_prediction_se_1_fp.count()
	print("第一正确匹配pack id 数量 = " + str(positive_hit_1_tp))
	print("第一正确匹配pack id presicion = " + str(positive_hit_1_tp / (positive_hit_1_fp + positive_hit_1_tp)))

	# 7. 两轮估算总量
	ph_positive_prodict = ph_positive_prodict + positive_hit_1_tp + positive_hit_1_fp
	print(ph_positive_prodict)
	ph_positive_hit = ph_positive_hit + positive_hit_1_tp
	print(ph_positive_hit)
	# ph_negetive_hit = result.where(result.prediction != result.label).count()
	print("第二轮 Pharbers Test set accuracy = " + str(ph_positive_hit / ph_total))
	print("第二轮 Pharbers Test set precision = " + str(ph_positive_hit / ph_positive_prodict))


	# 8. 第三轮，剩下的给出猜测
	# 9. 最后一步，给出完全没匹配的结果