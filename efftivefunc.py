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
from feature import *
from pdu_feature import *
# from similarity import *
# from oldsimi import *
from pyspark.sql.types import *
from pyspark.sql.functions import desc
from pyspark.sql.functions import rank
from pyspark.sql.functions import when
from pyspark.sql.functions import array
from pyspark.sql import Window
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler


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
		.config("spark.sql.autoBroadcastJoinThreshold", 1048576000) \
		.config("spark.sql.files.maxRecordsPerFile", 33554432) \
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


if __name__ == '__main__':
	spark = prepare()
	df_result = load_training_data(spark)
	df_result = dosage_standify(df_result)

	# edit_distance
	df_result = df_result.withColumn("MOLE_NAME_ED", edit_distance_pandas_udf(df_result.MOLE_NAME, df_result.MOLE_NAME_STANDARD))
	df_result = df_result.withColumn("PRODUCT_NAME_ED", edit_distance_with_contains_pandas_udf(df_result.PRODUCT_NAME, df_result.PRODUCT_NAME_STANDARD))
	df_result = df_result.withColumn("DOSAGE_ED", edit_distance_with_contains_pandas_udf(df_result.DOSAGE, df_result.DOSAGE_STANDARD))
	df_result = df_result.withColumn("SPEC_ED", edit_distance_with_contains_pandas_udf(df_result.SPEC, df_result.SPEC_STANDARD))
	df_result = df_result.withColumn("PACK_QTY_ED", edit_distance_with_float_change_pandas_udf(df_result.PACK_QTY, df_result.PACK_QTY_STANDARD))
	df_result = df_result.withColumn("MANUFACTURER_NAME_CH_ED", edit_distance_with_contains_pandas_udf(df_result.MANUFACTURER_NAME, df_result.MANUFACTURER_NAME_STANDARD))
	df_result = df_result.withColumn("MANUFACTURER_NAME_EN_ED", edit_distance_with_contains_pandas_udf(df_result.MANUFACTURER_NAME, df_result.MANUFACTURER_NAME_EN_STANDARD))
	df_result = df_result.withColumn("MANUFACTURER_NAME_ED", \
					when(df_result.MANUFACTURER_NAME_CH_ED < df_result.MANUFACTURER_NAME_EN_ED, df_result.MANUFACTURER_NAME_CH_ED) \
					.otherwise(df_result.MANUFACTURER_NAME_EN_ED))

	# features
	assembler = VectorAssembler( \
					inputCols=["MOLE_NAME_ED", "PRODUCT_NAME_ED", "DOSAGE_ED", "SPEC_ED", "PACK_QTY_ED", "MANUFACTURER_NAME_ED"], \
					outputCol="features")
	df_result = assembler.transform(df_result)

	df_result = df_result.withColumn("PACK_ID_CHECK_NUM", df_result.PACK_ID_CHECK.cast("int")).na.fill({"PACK_ID_CHECK_NUM": -1})
	df_result = df_result.withColumn("PACK_ID_STANDARD_NUM", df_result.PACK_ID_STANDARD.cast("int")).na.fill({"PACK_ID_STANDARD_NUM": -1})
	df_result = df_result.withColumn("label",
					when((df_result.PACK_ID_CHECK_NUM > 0) & (df_result.PACK_ID_STANDARD_NUM > 0) & (df_result.PACK_ID_CHECK_NUM == df_result.PACK_ID_STANDARD_NUM), 1.0).otherwise(0.0))

	df_result.write.mode("overwrite").parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/efftivefunc")
