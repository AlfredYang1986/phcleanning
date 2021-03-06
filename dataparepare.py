# -*- coding: utf-8 -*-
"""alfredyang@pharbers.com.

功能描述：
  * @author yzy
  * @version 0.0
  * @since 2020/08/12
  * @note  落盘数据：cpa_prod_join

"""

import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import lit
from pyspark.sql.types import *



# raw_data_path = "s3a://ph-max-auto/2020-08-11/BPBatchDAG/azsanofi_check"
raw_data_path = "s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/qilu/raw_data2"
split_data_path = "s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/qilu/0.0.11/splitdata"
training_data_path = "s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/qilu/0.0.11/tmp/data2"

# raw_data_path = "s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/azsanofi/raw_data"
# split_data_path = "s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/azsanofi/0.0.15/splitdata"
# training_data_path = "s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/azsanofi/0.0.15/tmp/data7"


# raw_data_path = "s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/chc/raw_data"
# split_data_path = "s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/chc/0.0.5/splitdata"
# training_data_path = "s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/chc/0.0.5/tmp/data3"

# raw_data_path = "s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/az/raw_data"
# split_data_path = "s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/az/0.0.1/splitdata"
# training_data_path = "s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/az/0.0.1/tmp/data3"

# raw_data_path = "s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/pf/raw_data_null"
# split_data_path = "s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/pf/0.0.3/splitdata"
# training_data_path = "s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/pf/0.0.3/tmp/data2"
# training_data_path = "s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/pf/0.0.3/tmp/data3"

# raw_data_path = "s3a://ph-stream/common/public/pfizer_check"
# split_data_path = "s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/pfizer_model/0.0.2/splitdata"
# training_data_path = "s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/pfizer_model/0.0.2/tmp/data3"

# raw_data_path = "s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/pfizer_model/0.0.2/raw_data"
# split_data_path = "s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/pfizer_model/0.0.2/splitdata"
# training_data_path = "s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/pfizer_model/0.0.2/tmp/data2"

raw_data_path = "s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/eia/raw_data_2"
split_data_path = "s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/eia/0.0.3/splitdata"
training_data_path = "s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/eia/0.0.3/tmp/data2"


def load_word_dict_encode(spark):
	df_encode = spark.read.parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/word_dict/0.0.12")
	return df_encode

"""
读取标准表WW
"""
def load_standard_prod(spark):
	 df_standard = spark.read.parquet("s3a://ph-stream/common/public/prod/0.0.20") \
					.select("PACK_ID",
							  "MOLE_NAME_CH", "MOLE_NAME_EN",
							  "PROD_DESC", "PROD_NAME_CH",
							  "CORP_NAME_EN", "CORP_NAME_CH", "MNF_NAME_EN", "MNF_NAME_CH",
							  "PCK_DESC", "DOSAGE", "SPEC", "PACK", \
							  "SPEC_valid_digit", "SPEC_valid_unit", "SPEC_gross_digit", "SPEC_gross_unit")
					# .drop("version")

	 df_standard = df_standard.withColumnRenamed("PACK_ID", "PACK_ID_STANDARD") \
					.withColumnRenamed("MOLE_NAME_CH", "MOLE_NAME_STANDARD") \
					.withColumnRenamed("PROD_NAME_CH", "PRODUCT_NAME_STANDARD") \
					.withColumnRenamed("CORP_NAME_CH", "CORP_NAME_STANDARD") \
					.withColumnRenamed("MNF_NAME_CH", "MANUFACTURER_NAME_STANDARD") \
					.withColumnRenamed("MNF_NAME_EN", "MANUFACTURER_NAME_EN_STANDARD") \
					.withColumnRenamed("DOSAGE", "DOSAGE_STANDARD") \
					.withColumnRenamed("SPEC", "SPEC_STANDARD") \
					.withColumnRenamed("PACK", "PACK_QTY_STANDARD") \
					.withColumnRenamed("SPEC_valid_digit", "SPEC_valid_digit_STANDARD") \
					.withColumnRenamed("SPEC_valid_unit", "SPEC_valid_unit_STANDARD") \
					.withColumnRenamed("SPEC_gross_digit", "SPEC_gross_digit_STANDARD") \
					.withColumnRenamed("SPEC_gross_unit", "SPEC_gross_unit_STANDARD")

	 df_standard = df_standard.select("PACK_ID_STANDARD", "MOLE_NAME_STANDARD",
										"PRODUCT_NAME_STANDARD", "CORP_NAME_STANDARD",
										"MANUFACTURER_NAME_STANDARD", "MANUFACTURER_NAME_EN_STANDARD",
										"DOSAGE_STANDARD", "SPEC_STANDARD", "PACK_QTY_STANDARD", \
										"SPEC_valid_digit_STANDARD", "SPEC_valid_unit_STANDARD", "SPEC_gross_digit_STANDARD", "SPEC_gross_unit_STANDARD")

	 # df_standard.show()
	 # df_standard.printSchema()

	 return df_standard


"""
读取待清洗的数据
"""
def load_cleanning_prod(spark):
	 #df_cleanning = spark.read.parquet("s3a://ph-stream/common/public/pfizer_check").drop("version")
	 df_cleanning = spark.read.parquet(raw_data_path).drop("version")  #.limit(3000)

	 # 为了验证算法，保证id尽可能可读性，投入使用后需要删除
	 df_cleanning = df_cleanning.repartition(1).withColumn("id", monotonically_increasing_id())
	 print(df_cleanning.count())
	 #df_cleanning = df_cleanning.readStream.withColumn("id", monotonically_increasing_id())

	 # 为了算法更高的并发，在这里将文件拆分为16个，然后以16的并发数开始跑人工智能
	 # df_cleanning.repartition(16).write.parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/splitdata")
	 return df_cleanning


"""
更高的并发数
"""
def modify_pool_cleanning_prod(spark):
	 # df_cleanning = spark.read.parquet("s3a://ph-stream/common/public/pfizer_check").drop("version")
	 df_cleanning = spark.read.parquet(raw_data_path)  #.limit(3000)

	 # 为了验证算法，保证id尽可能可读性，投入使用后需要删除
	 df_cleanning = df_cleanning.repartition(1).withColumn("id", monotonically_increasing_id())
	 
	 #df_cleanning = df_cleanning.readStream.withColumn("id", monotonically_increasing_id())
	 print("源数据条目： "+ str(df_cleanning.count()))
	 print("源数据：")
	 df_cleanning.show(3)

	 # 为了算法更高的并发，在这里将文件拆分为16个，然后以16的并发数开始跑人工智能
	 df_cleanning.write.mode("overwrite").parquet(split_data_path)
	 # return df_cleanning


"""
流读取待清洗的数据
"""
def load_stream_cleanning_prod(spark):
	schema = \
		StructType([ \
			StructField("id", LongType()), \
			StructField("PACK_ID_CHECK", StringType()), \
			StructField("MOLE_NAME", StringType()), \
			StructField("PRODUCT_NAME", StringType()), \
			StructField("DOSAGE", StringType()), \
			StructField("SPEC", StringType()), \
			StructField("PACK_QTY", StringType()), \
			StructField("MANUFACTURER_NAME", StringType())
		])

	df_cleanning = spark.readStream.schema(schema).parquet(split_data_path)
	return df_cleanning


"""
读取人工干预表
"""
def load_interfere_mapping(spark):
	 # df_interfere = spark.read.parquet("s3a://ph-stream/common/public/human_replace/0.0.15") \
	 #                     .withColumnRenamed("MOLE_NAME", "MOLE_NAME_INTERFERE") \
	 #                     .withColumnRenamed("PRODUCT_NAME", "PRODUCT_NAME_INTERFERE") \
	 #                     .withColumnRenamed("SPEC", "SPEC_INTERFERE") \
	 #                     .withColumnRenamed("DOSAGE", "DOSAGE_INTERFERE") \
	 #                     .withColumnRenamed("PACK_QTY", "PACK_QTY_INTERFERE") \
	 #                     .withColumnRenamed("MANUFACTURER_NAME", "MANUFACTURER_NAME_INTERFERE")
	 #                     # .drop("version")

	 df_interfere = spark.read.parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/human_replace_packid") \
						 .withColumnRenamed("match_MOLE_NAME_CH", "MOLE_NAME_INTERFERE") \
						 .withColumnRenamed("match_PRODUCT_NAME", "PRODUCT_NAME_INTERFERE")  \
						 .withColumnRenamed("match_SPEC", "SPEC_INTERFERE") \
						 .withColumnRenamed("match_DOSAGE", "DOSAGE_INTERFERE") \
						 .withColumnRenamed("match_PACK_QTY", "PACK_QTY_INTERFERE") \
						 .withColumnRenamed("match_MANUFACTURER_NAME_CH", "MANUFACTURER_NAME_INTERFERE") \
						 .withColumnRenamed("PACK_ID", "PACK_ID_INTERFERE")

	 return df_interfere

"""
读取剂型替换表
"""
def load_dosage_mapping(spark):
	df_dosage_mapping = spark.read.parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/cpa_dosage_mapping/0.0.2")
	# df_dosage_mapping = spark.read.parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/cpa_dosage_mapping/cpa_dosage_lst")
	return df_dosage_mapping


"""
	Load CrossJoin
"""
def load_training_data(spark):
	 #return spark.read.parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/data2") # pifer
	 return spark.read.parquet(training_data_path)
	 #return spark.read.parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/azsanofi/0.0.2/tmp/data3") # az
	 #return spark.read.parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/qilu/0.0.1/tmp/data3") # az

def load_split_data(spark):
	 return spark.read.parquet(split_data_path)

"""
download stand
"""
def download_prod_standard(spark):
	 df = load_standard_prod(spark)
	 df.repartition(1).write.mode("overwrite").option("header", "true").csv("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/standard")