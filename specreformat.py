import os
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import re
from pyspark.sql.functions import *
from pyspark.sql import Window
from pyspark.sql.functions import col, pandas_udf, PandasUDFType
from pyspark.sql.types import *
import re
import pandas as pd


@pandas_udf(StringType(), PandasUDFType.SCALAR)
def transfer_unit_pandas_udf(value):
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

	frame = { "SPEC": value }
	df = pd.DataFrame(frame)
	df["RESULT"] = df["SPEC"].apply(unit_transform)
	return df["RESULT"]


@pandas_udf(StringType(), PandasUDFType.SCALAR)
def percent_pandas_udf(percent, valid, gross):
  row_num = percent.shape[0]
  result = []
  digit_regex = '\d+\.?\d*e?-?\d*?'
  for index in range(row_num):
    if percent[index] != "" and valid[index] != "" and gross[index] == "":
      num = int(percent[index].strip("%"))
      value = re.findall(digit_regex, valid[index])[0]
      unit = valid[index].strip(value)  # type = str
      final_num = num*float(value)*0.01
      result.append(str(final_num) + unit)

    elif percent[index] != "" and valid[index] != "" and gross[index] != "":
      result.append("")

    else:
      result.append(percent[index])

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
  spec_gross_regex =  r'([0-9]\d*\.?\d*\s*[A-Za-z]*/?\s*[A-Za-z]+)[ /:∶+\s]([0-9]\d*\.?\d*\s*[A-Za-z]*/?\s*[A-Za-z]+)'
  df = df.withColumn("SPEC_gross", regexp_extract('SPEC', spec_gross_regex, 2))

  digit_regex_spec = r'(\d+\.?\d*e?-?\d*?)'
  df = df.withColumn("SPEC_gross_digit", regexp_extract('SPEC_gross', digit_regex_spec, 1))
  df = df.withColumn("SPEC_gross_unit", regexp_replace('SPEC_gross', digit_regex_spec, ""))
  df = df.withColumn("SPEC_valid_digit", regexp_extract('SPEC_valid', digit_regex_spec, 1))
  df = df.withColumn("SPEC_valid_unit", regexp_replace('SPEC_valid', digit_regex_spec, ""))

  df = df.withColumn("SPEC_valid", transfer_unit_pandas_udf(df.SPEC_valid))
  df = df.withColumn("SPEC_gross", transfer_unit_pandas_udf(df.SPEC_gross))
  df = df.drop("SPEC_gross_digit", "SPEC_gross_unit", "SPEC_valid_digit", "SPEC_valid_unit")
  df = df.withColumn("SPEC_percent", percent_pandas_udf(df.SPEC_percent, df.SPEC_valid, df.SPEC_gross))
  df = df.withColumn("SPEC_ept", lit(" "))
  df = df.withColumn("SPEC", concat("SPEC_co", "SPEC_percent", "SPEC_ept", "SPEC_valid", "SPEC_ept", "SPEC_gross")) \
          .drop("SPEC_percent", "SPEC_co", "SPEC_valid", "SPEC_gross", "SPEC_ept")
  return df
