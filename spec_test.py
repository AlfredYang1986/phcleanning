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
	cpa_test = [("1", "1", u"利多", "", "0.0125 MIU 2 ML", u"凝胶剂", 1, "辉瑞", ), 
	("2", "2", u"卡因",["利多","卡因"], "2% 20ML", u"滴丸", 1, "辉瑞", ), 
	("2.0", "3", u"中药", ["利多","卡因"], "(50UG+100UG)/DOS", u"溶液剂（注射剂）", 1, "辉瑞", ), 
	("2.0", "4", u"胰岛素", ["利多","卡因"], "AERO IN 1mg 0.02l ×1", u"片剂", 1, "辉瑞", ), 
	("5.0", "5", u"咪康唑", ["利多","卡因"], "0.25G/1.25MG", u"TAB FLM CTD", 1, "辉瑞", ), 
	("5","6", u"对乙酰氨基酚", ["利多","卡因"], "50MG:500MG", u"双层片", 1, "辉瑞"), 
	("5.0", "7", u"培美曲塞二钠", ["利多","卡因"], "(2.5MG+1000MG)", u"粉剂（粉剂针）", 1, "辉瑞")]
	cpa_schema = StructType([StructField('COMPANY',StringType(),True),
		StructField('SOURCE',StringType(),True),
		StructField('MOLE_NAME',StringType(),True),
		StructField('PRODUCT_NAME', StringType(),True),
		StructField('SPEC',StringType(),True),
		StructField('DOSAGE',StringType(),True),
		StructField('PACK',IntegerType(),True),
		StructField('mnf',StringType(),True),])
	cpa_test_df = spark.createDataFrame(cpa_test, schema=cpa_schema).na.fill("")
# 	cpa_test_df.show()
	
	cpa_test_df = cpa_test_df.withColumn("SPEC_ORIGINAL", cpa_test_df.SPEC).select("SPEC", "SPEC_ORIGINAL") # 保留原字段内容
	# df_cleanning = dosage_standify(df_cleanning)  # 剂型列规范
	cpa_test_df = spec_standify(cpa_test_df)  # 规格列规范
	cpa_test_df.show()

    
	
execute()