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


"""
读取标准表
"""
def load_standard_prod(spark):
     df_standard = spark.read.parquet("s3a://ph-stream/common/public/prod/0.0.18") \
                    .select("PACK_ID",
                              "MOLE_NAME_CH", "MOLE_NAME_EN",
                              "PROD_DESC", "PROD_NAME_CH",
                              "CORP_NAME_EN", "CORP_NAME_CH", "MNF_NAME_EN", "MNF_NAME_CH",
                              "PCK_DESC", "DOSAGE", "SPEC", "PACK")
                    # .drop("version")

     df_standard = df_standard.withColumnRenamed("PACK_ID", "PACK_ID_STANDARD") \
                    .withColumnRenamed("MOLE_NAME_CH", "MOLE_NAME_STANDARD") \
                    .withColumnRenamed("PROD_NAME_CH", "PRODUCT_NAME_STANDARD") \
                    .withColumnRenamed("CORP_NAME_CH", "CORP_NAME_STANDARD") \
                    .withColumnRenamed("MNF_NAME_CH", "MANUFACTURER_NAME_STANDARD") \
                    .withColumnRenamed("MNF_NAME_EN", "MANUFACTURER_NAME_EN_STANDARD") \
                    .withColumnRenamed("DOSAGE", "DOSAGE_STANDARD") \
                    .withColumnRenamed("SPEC", "SPEC_STANDARD") \
                    .withColumnRenamed("PACK", "PACK_QTY_STANDARD")

     df_standard = df_standard.select("PACK_ID_STANDARD", "MOLE_NAME_STANDARD",
                                        "PRODUCT_NAME_STANDARD", "CORP_NAME_STANDARD",
                                        "MANUFACTURER_NAME_STANDARD", "MANUFACTURER_NAME_EN_STANDARD",
                                        "DOSAGE_STANDARD", "SPEC_STANDARD", "PACK_QTY_STANDARD")

     # df_standard.show()
     # df_standard.printSchema()

     return df_standard


"""
读取待清洗的数据
"""
def load_cleanning_prod(spark):
     # df_cleanning = spark.read.parquet("s3a://ph-stream/common/public/pfizer_check").drop("version")
     df_cleanning = spark.read.parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/azsanofi_check/0.0.12/raw_data").drop("version")

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
     df_cleanning = spark.read.parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/azsanofi_check")

     # 为了验证算法，保证id尽可能可读性，投入使用后需要删除
     df_cleanning = df_cleanning.repartition(1).withColumn("id", monotonically_increasing_id())
     #df_cleanning = df_cleanning.readStream.withColumn("id", monotonically_increasing_id())
     print(df_cleanning.count())
     df_cleanning.show()

     # 为了算法更高的并发，在这里将文件拆分为16个，然后以16的并发数开始跑人工智能
     df_cleanning.write.mode("overwrite").parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/splitdata")
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

	df_cleanning = spark.readStream.schema(schema).parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/splitdata")
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
                         .withColumnRenamed("match_PRODUCT_NAME", "PRODUCT_NAME_INTERFERE") \
                         .withColumnRenamed("match_SPEC", "SPEC_INTERFERE") \
                         .withColumnRenamed("match_DOSAGE", "DOSAGE_INTERFERE") \
                         .withColumnRenamed("match_PACK_QTY", "PACK_QTY_INTERFERE") \
                         .withColumnRenamed("match_MANUFACTURER_NAME_CH", "MANUFACTURER_NAME_INTERFERE") \
                         .withColumnRenamed("PACK_ID", "PACK_ID_INTERFERE")

     return df_interfere


"""
	Load CrossJoin
"""
def load_training_data(spark):
    # return spark.read.parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/data2") # pifer
     return spark.read.parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/data3") # az


"""
download stand
"""
def download_prod_standard(spark):
     df = load_standard_prod(spark)
     df.repartition(1).write.mode("overwrite").option("header", "true").csv("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/alfred/tmp/standard")