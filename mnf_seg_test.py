# -*- coding: utf-8 -*-
"""alfredyang@pharbers.com.

功能描述：crossJoin

"""

import os
from pyspark.sql import SparkSession
from dataparepare import *
from interfere import *
from pdu_feature import *
from pyspark.sql.types import *
from pyspark.sql.functions import desc
from pyspark.sql.functions import rank
from pyspark.sql.functions import when
from pyspark.sql.functions import array
from pyspark.sql.functions import broadcast
from pyspark.sql import Window
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
import re
import pandas as pd


def execute():
	print("--"*80)
	print("zyyin-test 程序start")
	
	os.environ["PYSPARK_PYTHON"] = "python3"
		
		
	spark = SparkSession.builder \
		.master("yarn") \
		.appName("data from s3") \
		.config("spark.driver.memory", "1g") \
		.config("spark.executor.cores", "1") \
		.config("spark.executor.instance", "1") \
		.config("spark.executor.memory", "1g") \
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
# 	df = spark.read.parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/azsanofi/raw_data")
# 	df = spec_standify(df)  
# 	df.show()
	cpa_test = [("1", "1", u"Kowa Company, Ltd., Nagoya Factory （JA）", "", "0.0125 MIU 2 ML", u"日本兴和株式会社", 1, "辉瑞", ), 
	("2", "2", u"Patheon Italia S.p.A. （IT）",["利多","卡因"], "2% 20ML", u"比利时优时比制药公司", 1, "辉瑞", )]
	cpa_schema = StructType([StructField('COMPANY',StringType(),True),
		StructField('SOURCE',StringType(),True),
		StructField('MANUFACTURER_NAME',StringType(),True),
		StructField('PRODUCT_NAME', StringType(),True),
		StructField('SPEC',StringType(),True),
		StructField('MANUFACTURER_NAME_STANDARD',StringType(),True),
		StructField('Sigbu',IntegerType(),True),
		StructField('mnf',StringType(),True),])
	cpa_test_df = spark.createDataFrame(cpa_test, schema=cpa_schema).na.fill("").select("MANUFACTURER_NAME", "MANUFACTURER_NAME_STANDARD")
	cpa_test_df.show()
	df_encode = load_word_dict_encode(spark) 
	cpa_test_df = mnf_encoding_index(cpa_test_df, df_encode, spark)
	cpa_test_df = mnf_encoding_cosine(cpa_test_df)
	cpa_test_df.show()
	
	cpa_test_df = cpa_test_df.withColumn("JACCARD_DISTANCE", \
				efftiveness_with_jaccard_distance( \
					cpa_test_df.MANUFACTURER_NAME, cpa_test_df.MANUFACTURER_NAME_STANDARD, \
					cpa_test_df.MANUFACTURER_NAME, cpa_test_df.MANUFACTURER_NAME_STANDARD \
					))
	cpa_test_df.show()
	# df = spark.read.parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/dosage_test").select("MASTER_DOSAGE", "DOSAGE_STANDARD", "EFFTIVENESS_DOSAGE", "JACCARD_DISTANCE")
	# df.show()
	# df.printSchema()
	
	# df = df.withColumn("MASTER_DOSAGE", when(df.MASTER_DOSAGE.isNull(), df.JACCARD_DISTANCE). \
	# 					otherwise(df.MASTER_DOSAGE))
	# df = df.withColumn("eff", dosage_replace(df.MASTER_DOSAGE, df.DOSAGE_STANDARD, df.EFFTIVENESS_DOSAGE))				
	# df.show()
	

    
	
execute()