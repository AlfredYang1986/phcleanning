import os
from pyspark.sql import SparkSession
from dataparepare import *
from interfere import *
from similarity import *
# from oldsimi import *
from pyspark.sql.types import *
from pyspark.sql import functions as func
from pyspark.sql import Window
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import *


def execute():
	"""
		please input your code below
	"""
	
	print("--"*80)
	print("程序start: spec_reformat")
	
	os.environ["PYSPARK_PYTHON"] = "python3"
	# spark define
	spark = SparkSession.builder \
		.master("yarn") \
		.appName("BPBatchDAG") \
		.config("spark.driver.memory", "1g") \
		.config("spark.executor.cores", "1") \
		.config("spark.executor.instance", "1") \
		.config("spark.executor.memory", "1g") \
		.config('spark.sql.codegen.wholeStage', False) \
		.enableHiveSupport() \
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
		
	df_cleanning = spark.read.parquet("s3a://ph-stream/common/public/pfizer_check").drop("version").na.fill("")
	df_cleanning.show()
	
		
	
	def unit_transform(spec_str):
			# 输入一个数字+单位的str，输出同一单位后的str

			# 拆分数字和单位
			digit_regex = '\d+\.?\d*e?-?\d*?'
			# digit_regex = '0.\d*'
			if spec_str != "":
				value = re.findall(digit_regex, spec_str)[0]
				unit = spec_str.strip(value)  # type = str
				# value = float(value)  # type = float
				try:
					value = float(value)  # type = float
				except ValueError:
					value = 0.0
	
				# value transform
				if unit == "G" or unit == "GM":
					value = value *1000
				elif unit == "UG":
					value = value /1000
				elif unit == "L":
					value = value *1000
				elif unit == "TU" or unit == "TIU":
					value = value *10000
				elif unit == "MU" or unit == "MIU" or unit == "M":
					value = value *1000000
	
				# unit transform
				unit_switch = {
						"G": "MG",
						"GM": "MG",
						"MG": "MG",
						"UG": "MG",
						"L": "ML",
						"AXAU": "U",
						"AXAIU": "U",
						"IU": "U",
						"TU": "U",
						"TIU": "U",
						"MU": "U",
						"MIU": "U",
						"M": "U",
					}
	
				try:
					unit = unit_switch[unit]
				except KeyError:
					pass
				
			else:
				unit = ""
				value = ""

			return str(value) + unit
	
	
	
	@pandas_udf(StringType(), PandasUDFType.SCALAR)
	def transfer_unit_pandas_udf(value):
		row_num = value.shape[0]
		result = []
		for index in range(row_num):
			result.append(unit_transform(value[index]))
	
		return pd.Series(result)
	
	
	def spec_standify(df):
		
		df = df.withColumn("SPEC", regexp_replace("SPEC", r"(万)", "T"))
		df = df.withColumn("SPEC", regexp_replace("SPEC", r"(μ)", "U"))
		df = df.withColumn("SPEC", upper(df.SPEC))
		# df = df.withColumn("SPEC_gross", regexp_extract('SPEC', spec_regex, 2))
		# 拆分规格的成分
		df = df.withColumn("SPEC_percent", regexp_extract('SPEC', r'(\d+%)', 1))
		df = df.withColumn("SPEC_co", regexp_extract('SPEC', r'(CO)', 1))
		spec_valid_regex =  r'([0-9]\d*\.?\d*\s*[A-Za-z]*/?\s*[A-Za-z]+)'
		df = df.withColumn("SPEC_valid", regexp_extract('SPEC', spec_valid_regex, 1))
		spec_gross_regex =  r'([0-9]\d*\.?\d*\s*[A-Za-z]*/?\s*[A-Za-z]+)[ /:+\s]([0-9]\d*\.?\d*\s*[A-Za-z]*/?\s*[A-Za-z]+)'
		df = df.withColumn("SPEC_gross", regexp_extract('SPEC', spec_gross_regex, 2))
		
		digit_regex_spec = r'(\d+\.?\d*e?-?\d*?)'
		df = df.withColumn("SPEC_gross_digit", regexp_extract('SPEC_gross', digit_regex_spec, 1))
		df = df.withColumn("SPEC_gross_unit", regexp_replace('SPEC_gross', digit_regex_spec, ""))
		df = df.withColumn("SPEC_valid_digit", regexp_extract('SPEC_valid', digit_regex_spec, 1))
		df = df.withColumn("SPEC_valid_unit", regexp_replace('SPEC_valid', digit_regex_spec, ""))
		
		df = df.withColumn("SPEC_valid", transfer_unit_pandas_udf(df.SPEC_valid))
		df = df.withColumn("SPEC_gross", transfer_unit_pandas_udf(df.SPEC_gross))
		df = df.drop("SPEC_gross_digit", "SPEC_gross_unit", "SPEC_valid_digit", "SPEC_valid_unit")
		df = df.withColumn("SPEC", concat("SPEC_co", "SPEC_percent", "SPEC_valid", "SPEC_gross")) \
						.drop("SPEC_percent", "SPEC_co", "SPEC_valid", "SPEC_gross")

		return df

	df = spec_standify(df)
	df.show()
	
execute()