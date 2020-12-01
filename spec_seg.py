# -*- coding: utf-8 -*-
"""alfredyang@pharbers.com.

功能描述：one hot encoding for manifacture name

"""

import os
from pyspark.sql import SparkSession
from dataparepare import *
from interfere import *
from pdu_feature import *
from pyspark.sql.types import *
from pyspark.sql.functions import when
from pyspark.sql.functions import explode
from pyspark.sql.functions import min
from pyspark.sql.functions import first
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd
import numpy
from math import sqrt


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


@pandas_udf(StringType(), PandasUDFType.SCALAR)
def convert_array_words_to_string(arr):
	frame = {
		"ARRARY": arr
	}
	df = pd.DataFrame(frame)

	df["RESULT"] = df["ARRARY"].apply(lambda x: ",".join(x.tolist()))
	return df["RESULT"]


if __name__ == '__main__':
	spark = prepare()
	
	# df_spec = spark.read.parquet("")
	cpa_test = [("1", "1", u"西班牙礼来制药公司 （SP)", ["薄膜衣片","肠溶片", '泡腾片'], "10g:200万IU", 1, 1, "利多卡因", ), 
	("1", "2", u"利多",["薄膜衣片","肠溶片", '泡腾片'], "(3MU+1ML)", 2, 1, "", ), 
	("1", "2", u"利多", ["薄膜衣片","肠溶片", '泡腾片'], "3.40MU 1.2ML", 3, 4, "", ), 
	("2", "1", u"咪康唑", ["胶囊","吸入胶囊"], "5% 100ML", 2, 1, "", ), 
	("2", "3", u"咪康唑", ["胶囊","吸入胶囊"], "依折麦布10mg,辛伐他汀20mg", 1, 1, "", ), 
	("2","4", u"咪康唑", ["胶囊","吸入胶囊"], "18ΜG", 1, 1, ""), 
	("2", "1", u"咪康唑", ["胶囊","吸入胶囊"], "CO 1.25 GM", 1, 1, "")]
	cpa_schema = StructType([StructField('id',StringType(),True),
		StructField('SOURCE',StringType(),True),
		StructField('MOLE_NAME',StringType(),True),
		StructField('MASTER_DOSAGE', StringType(),True),
		StructField('SPEC',StringType(),True),
		StructField('DOSAGE',IntegerType(),True),
		StructField('PACK',IntegerType(),True),
		StructField('mnf',StringType(),True),])
	df = spark.createDataFrame(cpa_test, schema=cpa_schema).na.fill("").select("SPEC")
	df.show()
	
	df = df.withColumn("SPEC", regexp_replace("SPEC", r"(万)", "T"))
	df = df.withColumn("SPEC", regexp_replace("SPEC", r"(μ)", "U"))
	df = df.withColumn("SPEC", upper(df.SPEC))
	df = df.replace(" ", "")
	# df = df.withColumn("SPEC_gross", regexp_extract('SPEC', spec_regex, 2))
	# 拆分规格的成分
	df = df.withColumn("SPEC_percent", regexp_extract('SPEC', r'(\d+%)', 1))
	df = df.withColumn("SPEC_co", regexp_extract('SPEC', r'(CO)', 1))
	spec_valid_regex =  r'([0-9]\d*\.?\d*\s*[A-Za-z]*/?\s*[A-Za-z]+)'
	df = df.withColumn("SPEC_valid", regexp_extract('SPEC', spec_valid_regex, 1))
	spec_gross_regex =  r'([0-9]\d*\.?\d*\s*[A-Za-z]*/?\s*[A-Za-z]+)[ ,/:∶+\s][\u4e00-\u9fa5]*([0-9]\d*\.?\d*\s*[A-Za-z]*/?\s*[A-Za-z]+)'
	df = df.withColumn("SPEC_gross", regexp_extract('SPEC', spec_gross_regex, 2))
	spec_third_regex = r'([0-9]\d*\.?\d*\s*[A-Za-z]*/?\s*[A-Za-z]+)[ /:∶+\s]([0-9]\d*\.?\d*\s*[A-Za-z]*/?\s*[A-Za-z]+)[ /:∶+\s]([0-9]\d*\.?\d*\s*[A-Za-z]*/?\s*[A-Za-z]+)'
	df = df.withColumn("SPEC_third", regexp_extract('SPEC', spec_third_regex, 3))
	
	df.show()
	
	pure_number_regex_spec = r'(\s\d+$)'
	df = df.withColumn("SPEC_pure_number", regexp_extract('SPEC', pure_number_regex_spec, 1))

	digit_regex_spec = r'(\d+\.?\d*e?-?\d*?)'
	df = df.withColumn("SPEC_gross_digit", regexp_extract('SPEC_gross', digit_regex_spec, 1))
	df = df.withColumn("SPEC_gross_unit", regexp_replace('SPEC_gross', digit_regex_spec, ""))
	df = df.withColumn("SPEC_valid_digit", regexp_extract('SPEC_valid', digit_regex_spec, 1))
	df = df.withColumn("SPEC_valid_unit", regexp_replace('SPEC_valid', digit_regex_spec, ""))
	df = df.na.fill("")
	
	df.show()
	
	df = df.withColumn("SPEC_valid", transfer_unit_pandas_udf(df.SPEC_valid))
	df = df.withColumn("SPEC_gross", transfer_unit_pandas_udf(df.SPEC_gross))
	df = df.drop("SPEC_gross_digit", "SPEC_gross_unit", "SPEC_valid_digit", "SPEC_valid_unit")
	df = df.withColumn("SPEC_percent", percent_pandas_udf(df.SPEC_percent, df.SPEC_valid, df.SPEC_gross))
	df.show()
	

