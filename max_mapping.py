# -*- coding: utf-8 -*-
"""alfredyang@pharbers.com.

功能描述：max 数据匹配表统一schema整理

"""

import os
from pyspark.sql import SparkSession
from dataparepare import *
from interfere import *
from pdu_feature import *
from pyspark.sql.types import *
from pyspark.sql.functions import when, lit
from pyspark.sql.functions import explode
from pyspark.sql.functions import first
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd
import numpy
from math import sqrt
import boto3
from py4j.protocol import Py4JJavaError


def prepare():
	os.environ["PYSPARK_PYTHON"] = "python3"
	# 读取s3桶中的数据
	spark = SparkSession.builder \
		.master("yarn") \
		.appName("CPA&GYC match refactor") \
		.config("spark.driver.memory", "2g") \
		.config("spark.executor.cores", "2") \
		.config("spark.executor.instances", "2") \
		.config("spark.executor.memory", "2g") \
		.config('spark.sql.codegen.wholeStage', False) \
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


def schema_unify(df, company):
	# print(df.count())
	
	# df.printSchema()
	

	if company == "Janssen":
		df = df.drop("通用名", "标准商品名", "剂型", "规格", "包装数量", "生产企业")
	elif company == "Qilu":
		df = df.drop("ims_pck", "pfc")
	elif company == "Astellas":
		df = df.withColumnRenamed(df.columns[4], "包装数量")
		
	# df.show(1)
	df = df.withColumnRenamed("Molecule", "MOLE_NAME_ORIGIN")
	df = df.withColumnRenamed("药品名称", "MOLE_NAME_ORIGIN")
	df = df.withColumnRenamed("原始通用名", "MOLE_NAME_ORIGIN")
	df = df.withColumnRenamed("通用名_原始", "MOLE_NAME_ORIGIN")
	df = df.withColumnRenamed("Molecule_Name", "MOLE_NAME_ORIGIN")
	df = df.withColumnRenamed("molecule_name", "MOLE_NAME_ORIGIN")
	df = df.withColumnRenamed("药品名称1", "MOLE_NAME_ORIGIN")
	
	df = df.withColumnRenamed("Brand", "PROD_NAME_ORIGIN")
	df = df.withColumnRenamed("商品名1", "PROD_NAME_ORIGIN")
	df = df.withColumnRenamed("商品名", "PROD_NAME_ORIGIN")
	df = df.withColumnRenamed("Product_Name", "PROD_NAME_ORIGIN")
	
	df = df.withColumnRenamed("Form", "DOSAGE_ORIGIN")
	df = df.withColumnRenamed("剂型", "DOSAGE_ORIGIN")
	df = df.withColumnRenamed("Dosage", "DOSAGE_ORIGIN")
	df = df.withColumnRenamed("剂型1", "DOSAGE_ORIGIN")
	
	df = df.withColumnRenamed("Specifications", "SPEC_ORIGIN")
	df = df.withColumnRenamed("pack_description", "SPEC_ORIGIN")
	df = df.withColumnRenamed("药品规格1", "SPEC_ORIGIN")
	df = df.withColumnRenamed("规格", "SPEC_ORIGIN")
	df = df.withColumnRenamed("Specification", "SPEC_ORIGIN")
	df = df.withColumnRenamed("Pack", "SPEC_ORIGIN")   ######### TODO
	df = df.withColumnRenamed("Size", "SPEC_ORIGIN")
	
	
	df = df.withColumnRenamed("Pack_Number", "PACK_QTY_ORIGIN")
	df = df.withColumnRenamed("最小包装数量", "PACK_QTY_ORIGIN")
	df = df.withColumnRenamed("包装数量", "PACK_QTY_ORIGIN")
	df = df.withColumnRenamed("包装规格", "PACK_QTY_ORIGIN")
	df = df.withColumnRenamed("PackNumber", "PACK_QTY_ORIGIN")
	# df = df.withColumnRenamed(r"包装1\n数量", "PACK_QTY_ORIGIN")
	# df = df.withColumnRenamed("包装1", "PACK_QTY_ORIGIN")
	
	df = df.withColumnRenamed("Manufacturer", "MANUFACTURE_NAME_ORIGIN")
	df = df.withColumnRenamed("企业名称1", "MANUFACTURE_NAME_ORIGIN")
	df = df.withColumnRenamed("生产企业", "MANUFACTURE_NAME_ORIGIN")
	df = df.withColumnRenamed("CORPORATION", "MANUFACTURE_NAME_ORIGIN")
	df = df.withColumnRenamed("生产厂家", "MANUFACTURE_NAME_ORIGIN")
	df = df.withColumnRenamed("company_name", "MANUFACTURE_NAME_ORIGIN")
	
	df = df.withColumnRenamed("标准通用名", "MOLE_NAME")
	df = df.withColumnRenamed("通用名_标准", "MOLE_NAME")
	df = df.withColumnRenamed("molecule_name_std", "MOLE_NAME")
	
	df = df.withColumnRenamed("标准商品名", "PRODUCT_NAME")
	df = df.withColumnRenamed("商品名_标准", "PRODUCT_NAME")
	df = df.withColumnRenamed("product_name_std", "PRODUCT_NAME")
	
	df = df.withColumnRenamed("标准剂型", "DOSAGE")
	df = df.withColumnRenamed("剂型_标准", "DOSAGE")
	df = df.withColumnRenamed("Form_std", "DOSAGE")
	df = df.withColumnRenamed("S_Dosage", "DOSAGE")

	df = df.withColumnRenamed("标准规格", "SPEC")
	df = df.withColumnRenamed("药品规格_标准", "SPEC")
	df = df.withColumnRenamed("规格_标准", "SPEC")
	df = df.withColumnRenamed("S_Pack", "SPEC")
	df = df.withColumnRenamed("Specifications_std", "SPEC")
	df = df.withColumnRenamed("pack_description_std", "SPEC")
	
	df = df.withColumnRenamed("标准包装数", "PACK_QTY")
	df = df.withColumnRenamed("包装数_标准", "PACK_QTY")
	df = df.withColumnRenamed("标准包装数量", "PACK_QTY")
	df = df.withColumnRenamed("包装数量2", "PACK_QTY")
	df = df.withColumnRenamed("包装数量_标准", "PACK_QTY")
	df = df.withColumnRenamed("Pack_Number_std", "PACK_QTY")
	df = df.withColumnRenamed("毫克数", "PACK_QTY")
	df = df.withColumnRenamed("S_PackNumber", "PACK_QTY")
	df = df.withColumnRenamed("包装数量2", "PACK_QTY")
	
	# df = df.withColumnRenamed("标准集团", "MANUFACTURE_NAME")
	df = df.withColumnRenamed("集团_标准", "MANUFACTURE_NAME")
	df = df.withColumnRenamed("生产企业_标准", "MANUFACTURE_NAME")
	df = df.withColumnRenamed("标准生产厂家", "MANUFACTURE_NAME")
	df = df.withColumnRenamed("S_CORPORATION", "MANUFACTURE_NAME")
	df = df.withColumnRenamed("Manufacturer_std", "MANUFACTURE_NAME")
	df = df.withColumnRenamed("标准生产企业", "MANUFACTURE_NAME")
	df = df.withColumnRenamed("标准企业", "MANUFACTURE_NAME")
	df = df.withColumnRenamed("company_name_std", "MANUFACTURE_NAME")
	
	df = df.withColumnRenamed("PFC（来自于文博的外部版本，文博版本的变动需要加到这里）", "PACK_ID_CHECK")
	df = df.withColumnRenamed("pfc", "PACK_ID_CHECK")
	df = df.withColumnRenamed("Pack_ID", "PACK_ID_CHECK")
	df = df.withColumnRenamed("packcode", "PACK_ID_CHECK")
	df = df.withColumnRenamed("packid", "PACK_ID_CHECK")
	df = df.withColumnRenamed("最终pfc", "PACK_ID_CHECK")
	df = df.withColumnRenamed("PackID", "PACK_ID_CHECK")
	df = df.withColumnRenamed("Pack_Id", "PACK_ID_CHECK")
	df = df.withColumnRenamed("PFC", "PACK_ID_CHECK")
	
	
	if (company == "NHWA") | (company == "Sankyo"):
		df = df.withColumn("PACK_QTY", lit(df.PACK_QTY_ORIGIN))
	elif company == "Janssen":
		df = df.withColumn("PACK_QTY_ORIGIN", lit(""))
		df = df.withColumn("PACK_QTY", lit(""))
		df = df.withColumn("DOSAGE_ORIGIN", lit(""))
		df = df.withColumn("DOSAGE", lit(""))
	
	df = df.withColumn("COMPANY", lit(company))
	df = df.withColumn("COMMENT", lit(""))
	
	df = df.select("MOLE_NAME_ORIGIN", "PROD_NAME_ORIGIN", "DOSAGE_ORIGIN", "SPEC_ORIGIN", "PACK_QTY_ORIGIN", "MANUFACTURE_NAME_ORIGIN", \
				"MOLE_NAME", "PRODUCT_NAME", "DOSAGE", "SPEC", "PACK_QTY", "MANUFACTURE_NAME", \
				"COMPANY", "PACK_ID_CHECK", "COMMENT")

	# df.show(2)
	print("完成")
	return df
	
	
def df_total(spark):
	df_total = [("", "", "", "", "", "", "", "", "", "", "", "", "", "", ""),]
	cpa_schema = StructType([StructField('MOLE_NAME_ORIGIN',StringType(),True),
	StructField('PROD_NAME_ORIGIN',StringType(),True),
	StructField('DOSAGE_ORIGIN',StringType(),True),
	StructField('SPEC_ORIGIN', StringType(),True),
	StructField('PACK_QTY_ORIGIN',StringType(),True),
	StructField('MANUFACTURE_NAME_ORIGIN',StringType(),True),
	StructField('MOLE_NAME',StringType(),True),
	StructField('PRODUCT_NAME',StringType(),True),
	StructField('DOSAGE',StringType(),True),
	StructField('SPEC',StringType(),True),
	StructField('PACK_QTY',StringType(),True),
	StructField('MANUFACTURE_NAME',StringType(),True),
	StructField('PACK_ID_CHECK',StringType(),True),
	StructField('COMMENT',StringType(),True),
	StructField('COMPANY',StringType(),True),])
	df_total = spark.createDataFrame(df_total, schema=cpa_schema)
	
	return df_total


if __name__ == '__main__':
	spark = prepare()
	
	client = boto3.client('s3')
	response = client.list_objects_v2(
	    Bucket="ph-max-auto",
	    Prefix="v0.0.1-2020-06-08/",
	    Delimiter="/"
	)
	
	# 规定整张表的schema
	df_total = df_total(spark)
	df_total.show()
	print(df_total.count())

	for idx in range(0, response["KeyCount"]):
		idx_key = response["CommonPrefixes"][idx]["Prefix"]
		print()
		# print(idx_key)
		company = idx_key.strip("v0.0.1-2020-06-08/").strip("/")
		print(company)
		
		company07_lst = ["Beite", "Janssen", "Novartis", "Sankyo"]
		company_skip_lst = ["Bayer", "Chugai", "Eisai"]
		company_not_lst = ["Common_files", "FileExchange", "New_add_test", "Test", "UDRC"]
		
		
		# try:
		if company in company_skip_lst:
			print("暂时跳过")
		elif company in company_not_lst:
			print("不是项目名称")
		elif company in company07_lst:
			print("07")
			df = spark.read.parquet("s3a://ph-max-auto/" + idx_key + "202007/prod_mapping")
			df_result = schema_unify(df, company)
			df_total = df_total.unionByName(df_result)
			print(df_total.count())
		elif company == "Pfizer":
			print("08")
			df = spark.read.parquet("s3a://ph-max-auto/" + idx_key + "202008/prod_mapping")
			df_result = schema_unify(df, company)
			df_total = df_total.unionByName(df_result)
			print(df_total.count())
		else:
			df = spark.read.parquet("s3a://ph-max-auto/" + idx_key + "202009/prod_mapping")
			df_result = schema_unify(df, company)
			df_total = df_total.unionByName(df_result)
			print(df_total.count())
			
	print(df_total.count())
	df_total.show(2)
	df_total = df_total.repartition(1).withColumn("ID", monotonically_increasing_id())
	df_total.show(2)
	
	df_total.where(df_total.ID>0).write.repartitionByRange("COMPANY").mode("overwrite").parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/max_mapping_with_id")
	print("写入完成")
		# except:
		# 	print(company + " 不是项目名称")
		# 	pass

