# -*- codin: utf-8 -*-
"""alfredyan@pharbers.com.

功能描述：
  * @author yzy
  * @version 0.0
  * @since 2020/08/12
  * @note  落盘数据：cpa_prod_join

"""

import os
import boto3
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import *
from pyspark.sql.functions import regexp_replace, regexp_extract
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.functions import udf
from pyspark.sql.functions import upper
from pyspark.sql.functions import lit
from pyspark.sql.functions import concat
from pyspark.sql.functions import desc
from pyspark.sql.functions import rank, row_number
from pyspark.sql.functions import when
from pyspark.sql.functions import col, udf
import numpy
from pyspark.sql.types import *
from pyspark.sql.functions import array, array_contains, array_distinct
from pyspark.sql.functions import broadcast
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import explode
from pyspark.sql.functions import pandas_udf, PandasUDFType
from math import isnan
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
# from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql import Window
from math import sqrt
import re
import numpy as np
import pandas as pd
import pkuseg
from nltk.metrics import edit_distance as ed
from nltk.metrics import jaccard_distance as jd
# from nltk.metrics import jaro_winkler_similarity as jws
import math


def dosage_standify(df):
	df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"（注射剂）", ""))
	df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"（粉剂针）", ""))
	df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"（胶丸、滴丸）", ""))

	# df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"SOLN", "注射液"))
	# df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"POWD", "粉针剂"))
	# df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"SUSP", "混悬"))
	# df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"OINT", "膏剂"))
	# df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"NA", "鼻"))
	# df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"SYRP", "口服"))
	# df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"PATC", "贴膏"))
	# df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"EMUL", "乳"))
	# df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"AERO", "气雾"))
	# df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"RAN", "颗粒"))
	# df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"SUPP", "栓"))
	# df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"PILL", "丸"))
	# df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"MISC", "混合"))
	# df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"LIQD", "溶液"))
	# df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"TAB", "片"))
	# df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"CAP", "胶囊"))

	# cross join (only Chinese)
	# df_dosage = df.crossJoin(df_dosage_mapping.select("CPA_DOSAGE", "MASTER_DOSAGE").distinct())
	# df_dosage = df_dosage.withColumn("DOSAGE_SUB", when(df_dosage.DOSAGE.contains(df_dosage.CPA_DOSAGE) | df_dosage.CPA_DOSAGE.contains(df_dosage.DOSAGE), \
	# 								df_dosage.MASTER_DOSAGE).otherwise("")) \
	# 								.drop("MASTER_DOSAGE", "CPA_DOSAGE")
	# df_dosage = df_dosage.where(df_dosage.DOSAGE_SUB != "")
	# df_dosage = df_dosage.unionByName(df.withColumn("DOSAGE_SUB", df.DOSAGE)) \
	# 			.select("id", "PACK_ID_CHECK", "MOLE_NAME", "PRODUCT_NAME", "DOSAGE", "SPEC", "PACK_QTY", "MANUFACTURER_NAME", "DOSAGE_SUB").distinct()
	# df_dosage = df_dosage.withColumnRenamed("DOSAGE", "DOSAGE_ORIGINAL").withColumnRenamed("DOSAGE_SUB", "DOSAGE")

	return df


"""
	现有的数据相似度的效能函数
	通过计算字符串的编辑距离，以及包含关系来确定某一项的距离

	Calculate the Levenshtein edit-distance between two strings.
	The edit distance is the number of characters that need to be
	substituted, inserted, or deleted, to transform s1 into s2.  For
	example, transforming "rain" to "shine" requires three steps,
	consisting of two substitutions and one insertion:
	"rain" -> "sain" -> "shin" -> "shine".  These operations could have
	been done in other orders, but at least three steps are needed.
"""
@pandas_udf(ArrayType(DoubleType()), PandasUDFType.SCALAR)
def efftiveness_with_edit_distance(mo, ms, po, ps, do, ds, so, ss, qo, qs, mf, mfc, mfe):
	frame = {
		"MOLE_NAME": mo, "MOLE_NAME_STANDARD": ms,
		"PRODUCT_NAME": po, "PRODUCT_NAME_STANDARD": ps,
		"DOSAGE": do, "DOSAGE_STANDARD": ds,
		"SPEC": so, "SPEC_STANDARD": ss,
		"PACK_QTY": qo, "PACK_QTY_STANDARD": qs,
		"MANUFACTURER_NAME": mf, "MANUFACTURER_NAME_STANDARD": mfc, "MANUFACTURER_NAME_EN_STANDARD": mfe
	}
	df = pd.DataFrame(frame)

	df["MOLE_ED"] = df.apply(lambda x: ed(x["MOLE_NAME"], x["MOLE_NAME_STANDARD"]), axis=1)
	df["PRODUCT_ED"] = df.apply(lambda x: 0 if x["PRODUCT_NAME"] in x ["PRODUCT_NAME_STANDARD"] \
										else 0 if x["PRODUCT_NAME_STANDARD"] in x ["PRODUCT_NAME"] \
										else ed(x["PRODUCT_NAME"], x["PRODUCT_NAME_STANDARD"]), axis=1)
	df["DOSAGE_ED"] = df.apply(lambda x: 0 if x["DOSAGE"] in x ["DOSAGE_STANDARD"] \
										else 0 if x["DOSAGE_STANDARD"] in x ["DOSAGE"] \
										else ed(x["DOSAGE"], x["DOSAGE_STANDARD"]), axis=1)
	df["SPEC_ED"] = df.apply(lambda x: 0 if x["SPEC"] in x ["SPEC_STANDARD"] \
										else 0 if x["SPEC_STANDARD"] in x ["SPEC"] \
										else ed(x["SPEC"], x["SPEC_STANDARD"]), axis=1)
	df["PACK_QTY_ED"] = df.apply(lambda x: ed(x["PACK_QTY"], x["PACK_QTY_STANDARD"].replace(".0", "")), axis=1)
	df["MANUFACTURER_NAME_CH_ED"] = df.apply(lambda x: 0 if x["MANUFACTURER_NAME"] in x ["MANUFACTURER_NAME_STANDARD"] \
										else 0 if x["MANUFACTURER_NAME_STANDARD"] in x ["MANUFACTURER_NAME"] \
										else ed(x["MANUFACTURER_NAME"], x["MANUFACTURER_NAME_STANDARD"]), axis=1)
	df["MANUFACTURER_NAME_EN_ED"] = df.apply(lambda x: 0 if x["MANUFACTURER_NAME"] in x ["MANUFACTURER_NAME_EN_STANDARD"] \
										else 0 if x["MANUFACTURER_NAME_EN_STANDARD"] in x ["MANUFACTURER_NAME"] \
										else ed(x["MANUFACTURER_NAME"], x["MANUFACTURER_NAME_EN_STANDARD"]), axis=1)
	df["MANUFACTURER_NAME_MINUS"] = df["MANUFACTURER_NAME_CH_ED"] - df["MANUFACTURER_NAME_EN_ED"]
	df.loc[df["MANUFACTURER_NAME_MINUS"] < 0.0, "MANUFACTURER_NAME_EFFECTIVENESS"] = df["MANUFACTURER_NAME_CH_ED"]
	df.loc[df["MANUFACTURER_NAME_MINUS"] >= 0.0, "MANUFACTURER_NAME_EFFECTIVENESS"] = df["MANUFACTURER_NAME_EN_ED"]

	df["RESULT"] = df.apply(lambda x: [x["MOLE_ED"], \
										x["PRODUCT_ED"], \
										x["DOSAGE_ED"], \
										x["SPEC_ED"], \
										x["PACK_QTY_ED"], \
										x["MANUFACTURER_NAME_EFFECTIVENESS"], \
										], axis=1)
	return df["RESULT"]


"""
	由于高级的字符串匹配算法的时间复杂度过高，
	在大量的数据量的情况下需要通过简单的数据算法过滤掉不一样的数据
	这个是属于数据Cutting过程，所以这两个变量不是精确变量，不放在后期学习的过程中
"""
@pandas_udf(ArrayType(DoubleType()), PandasUDFType.SCALAR)
def efftiveness_with_jaccard_distance(mo, ms, po, ps):
	frame = {
		"MOLE_NAME": mo, "MOLE_NAME_STANDARD": ms,
		"PACK_QTY": po, "PACK_QTY_STANDARD": ps
	}
	df = pd.DataFrame(frame)

	df["MOLE_JD"] = df.apply(lambda x: jd(set(x["MOLE_NAME"]), set(x["MOLE_NAME_STANDARD"])), axis=1)
	df["PACK_JD"] = df.apply(lambda x: jd(set(x["PACK_QTY"].replace(".0", "")), set(x["PACK_QTY_STANDARD"].replace(".0", ""))), axis=1)
	df["RESULT"] = df.apply(lambda x: [x["MOLE_JD"], x["PACK_JD"]], axis=1)
	return df["RESULT"]


"""
	由于Edit Distance不是一个相似度算法，当你在计算出相似度之后还需要通过一定的辅助算法计算
	Normalization。但是由于各个地方的Normalization很有可能产生误差错误，
	需要一个统一的Similarity的计算方法，去消除由于Normalization来产生的误差
	优先使用1989年提出的  Jaro Winkler distance

	The Jaro Winkler distance is an extension of the Jaro similarity in:

			William E. Winkler. 1990. String Comparator Metrics and Enhanced
			Decision Rules in the Fellegi-Sunter Model of Record Linkage.
			Proceedings of the Section on Survey Research Methods.
			American Statistical Association: 354-359.
		such that:

			jaro_winkler_sim = jaro_sim + ( l * p * (1 - jaro_sim) )
"""
@pandas_udf(ArrayType(DoubleType()), PandasUDFType.SCALAR)
def efftiveness_with_jaro_winkler_similarity(mo, ms, po, ps, do, ds, so, ss, qo, qs, mf, mfc, mfe, spec):
	def jaro_similarity(s1, s2):
		# First, store the length of the strings
		# because they will be re-used several times.
		len_s1, len_s2 = len(s1), len(s2)

		# The upper bound of the distance for being a matched character.
		match_bound = max(len_s1, len_s2) // 2 - 1

		# Initialize the counts for matches and transpositions.
		matches = 0  # no.of matched characters in s1 and s2
		transpositions = 0  # no. of transpositions between s1 and s2
		flagged_1 = []  # positions in s1 which are matches to some character in s2
		flagged_2 = []  # positions in s2 which are matches to some character in s1

		# Iterate through sequences, check for matches and compute transpositions.
		for i in range(len_s1):  # Iterate through each character.
			upperbound = min(i + match_bound, len_s2 - 1)
			lowerbound = max(0, i - match_bound)
			for j in range(lowerbound, upperbound + 1):
				if s1[i] == s2[j] and j not in flagged_2:
					matches += 1
					flagged_1.append(i)
					flagged_2.append(j)
					break
		flagged_2.sort()
		for i, j in zip(flagged_1, flagged_2):
			if s1[i] != s2[j]:
				transpositions += 1

		if matches == 0:
			return 0
		else:
			return (
				1
				/ 3
				* (
					matches / len_s1
					+ matches / len_s2
					+ (matches - transpositions // 2) / matches
				)
			)


	def jaro_winkler_similarity(s1, s2, p=0.1, max_l=4):
		if not 0 <= max_l * p <= 1:
			print("The product  `max_l * p` might not fall between [0,1].Jaro-Winkler similarity might not be between 0 and 1.")

		# Compute the Jaro similarity
		jaro_sim = jaro_similarity(s1, s2)

		# Initialize the upper bound for the no. of prefixes.
		# if user did not pre-define the upperbound,
		# use shorter length between s1 and s2

		# Compute the prefix matches.
		l = 0
		# zip() will automatically loop until the end of shorter string.
		for s1_i, s2_i in zip(s1, s2):
			if s1_i == s2_i:
				l += 1
			else:
				break
			if l == max_l:
				break
		# Return the similarity value as described in docstring.
		return jaro_sim + (l * p * (1 - jaro_sim))


	frame = {
		"MOLE_NAME": mo, "MOLE_NAME_STANDARD": ms,
		"PRODUCT_NAME": po, "PRODUCT_NAME_STANDARD": ps,
		"DOSAGE": do, "DOSAGE_STANDARD": ds,
		"SPEC": so, "SPEC_STANDARD": ss,
		"PACK_QTY": qo, "PACK_QTY_STANDARD": qs,
		"MANUFACTURER_NAME": mf, "MANUFACTURER_NAME_STANDARD": mfc, "MANUFACTURER_NAME_EN_STANDARD": mfe,
		"SPEC_ORIGINAL": spec
	}
	df = pd.DataFrame(frame)

	df["MOLE_JWS"] = df.apply(lambda x: jaro_winkler_similarity(x["MOLE_NAME"], x["MOLE_NAME_STANDARD"]), axis=1)
	df["PRODUCT_JWS"] = df.apply(lambda x: 1 if x["PRODUCT_NAME"] in x ["PRODUCT_NAME_STANDARD"] \
										else 1 if x["PRODUCT_NAME_STANDARD"] in x ["PRODUCT_NAME"] \
										else jaro_winkler_similarity(x["PRODUCT_NAME"], x["PRODUCT_NAME_STANDARD"]), axis=1)
	df["DOSAGE_JWS"] = df.apply(lambda x: 1 if x["DOSAGE"] in x ["DOSAGE_STANDARD"] \
										else 1 if x["DOSAGE_STANDARD"] in x ["DOSAGE"] \
										else jaro_winkler_similarity(x["DOSAGE"], x["DOSAGE_STANDARD"]), axis=1)
	df["SPEC_JWS"] = df.apply(lambda x: 1 if x["SPEC"].strip().replace("//", "/") ==  x["SPEC_STANDARD"].strip().replace("//", "/") \
										else 0 if ((x["SPEC"].strip().replace("///", "") == "") or (x["SPEC_STANDARD"].strip().replace("///", "") == "")) \
										else 1 if x["SPEC"].strip().replace("//", "/") in x["SPEC_STANDARD"].strip().replace("//", "/") \
										else 1 if x["SPEC_STANDARD"].strip().replace("//", "/") in x["SPEC"].strip().replace("//", "/") \
										else jaro_winkler_similarity(x["SPEC"].strip().replace("//", "/"), x["SPEC_STANDARD"].strip().replace("//", "/")), axis=1)
	df["PACK_QTY_JWS"] = df.apply(lambda x: 1 if (x["PACK_QTY"].replace(".0", "") == x["PACK_QTY_STANDARD"].replace(".0", "")) \
										| (("喷" in x["PACK_QTY"]) & (x["PACK_QTY"] in x["SPEC_ORIGINAL"])) \
										else 0, axis=1)
	df["MANUFACTURER_NAME_CH_JWS"] = df.apply(lambda x: 1 if x["MANUFACTURER_NAME"] in x ["MANUFACTURER_NAME_STANDARD"] \
										else 1 if x["MANUFACTURER_NAME_STANDARD"] in x ["MANUFACTURER_NAME"] \
										else jaro_winkler_similarity(x["MANUFACTURER_NAME"], x["MANUFACTURER_NAME_STANDARD"]), axis=1)
	df["MANUFACTURER_NAME_EN_JWS"] = df.apply(lambda x: 1 if x["MANUFACTURER_NAME"] in x ["MANUFACTURER_NAME_EN_STANDARD"] \
										else 1 if x["MANUFACTURER_NAME_EN_STANDARD"] in x ["MANUFACTURER_NAME"] \
										else jaro_winkler_similarity(x["MANUFACTURER_NAME"].upper(), x["MANUFACTURER_NAME_EN_STANDARD"].upper()), axis=1)
	df["MANUFACTURER_NAME_MINUS"] = df["MANUFACTURER_NAME_CH_JWS"] - df["MANUFACTURER_NAME_EN_JWS"]
	df.loc[df["MANUFACTURER_NAME_MINUS"] < 0.0, "MANUFACTURER_NAME_JWS"] = df["MANUFACTURER_NAME_EN_JWS"]
	df.loc[df["MANUFACTURER_NAME_MINUS"] >= 0.0, "MANUFACTURER_NAME_JWS"] = df["MANUFACTURER_NAME_CH_JWS"]

	df["RESULT"] = df.apply(lambda x: [x["MOLE_JWS"], \
										x["PRODUCT_JWS"], \
										x["DOSAGE_JWS"], \
										x["SPEC_JWS"], \
										x["PACK_QTY_JWS"], \
										x["MANUFACTURER_NAME_JWS"], \
										], axis=1)
	return df["RESULT"]

@pandas_udf(StringType(), PandasUDFType.SCALAR)
def spec_valid_std_transfer_pandas_udf(value):
	
	def digit_addition(spec_str):
		lst = spec_str.split(",")
		value_total = 0.0
		# if (lst[0] == "nan") | (len(lst) == 0):
		if len(lst) <= 1:
			value = "0.0"
		else:
			# try:
			for num in lst:
				num = float(num)
				value_total = (value_total + num)
				value  = str(value_total)
			# except:
			# 	value = 0.0
		return value
				
	frame = { "SPEC": value }
	df = pd.DataFrame(frame)
	df["RESULT"] = df["SPEC"].apply(digit_addition)
	return df["RESULT"]

@pandas_udf(StringType(), PandasUDFType.SCALAR)
def spec_total_cleanning_pandas_udf(SPEC_valid_digit, SPEC_valid_unit, SPEC_gross_digit, SPEC_gross_unit):
	
	def digit_addition(a, b, valid_unit, gross_unit):
		if (valid_unit == gross_unit) & ("," not in a) & (a != ""):
			a = float(a)
			b = float(b)
			return str(a+b)
		else:
			return ""
				
	frame = { "SPEC_valid_digit": SPEC_valid_digit, "SPEC_valid_unit": SPEC_valid_unit,
			  "SPEC_gross_digit": SPEC_gross_digit, "SPEC_gross_unit": SPEC_gross_unit,	}
	df = pd.DataFrame(frame)
	df["RESULT"] = df.apply(lambda x: digit_addition(x["SPEC_valid_digit"], x["SPEC_gross_digit"], x["SPEC_valid_unit"], x["SPEC_gross_unit"]), axis=1)
	return df["RESULT"]


@pandas_udf(StringType(), PandasUDFType.SCALAR)
def transfer_unit_pandas_udf(value):
	def unit_trans(value, unit):
		# value transform
		if unit == "G" or unit == "GM":
			value = value *1000
		elif unit == "UG" or unit == "UG/DOS":
			value = value /1000
		elif unit == "L":
			value = value *1000
		elif unit == "TU" or unit == "TIU":
			value = value *10000
		elif unit == "MU" or unit == "MIU" or unit == "M":
			value = value *1000000
		elif (unit == "Y"):
			value = value /1000
		if value >= 1:
			value = round(value, 1)
		else:
			value = value

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
				"Y": "MG",
				"MC": "MC",
			}
		try:
			unit = unit_switch[unit]
		except KeyError:
			pass
		return value, unit
	
	
	def unit_transform(spec_str):
		spec_str = spec_str.replace(" ", "")
		# 拆分数字和单位
		digit_regex = '\d+\.?\d*e?-?\d*?'
		# digit_regex = '0.\d*'
		# try:
		if spec_str != "":
			values = re.findall(digit_regex, spec_str)
			if len(values) == 1:
				value = values[0]
				unit = spec_str.strip(value)  # type = str
				value = float(value)  # type = float
				value = unit_trans(value, unit)[0]
				unit = unit_trans(value, unit)[1]
			elif len(values) >= 2:
				# unit = unit
				# value = 12222
				value_result = ""
				unit_regex = '[A-Z]+\d*'
				unit = re.findall(unit_regex, spec_str)[0]
				# value = "000"
				for value in values:
					value = float(value)  # type = float
					value = unit_trans(value, unit)[0]
					value_result = value_result + str(value) + ","
				unit = unit_trans(value, unit)[1]
				value = value_result.strip(",")
				
		else:
			unit = ""
			value = ""

		return str(value) + unit

		# except Exception:
		# 	return spec_str

	frame = { "SPEC": value }
	df = pd.DataFrame(frame)
	df["RESULT"] = df["SPEC"].apply(unit_transform)
	return df["RESULT"]


@pandas_udf(StringType(), PandasUDFType.SCALAR)
def percent_pandas_udf(percent, valid, gross):
	def percent_calculation(percent, valid, gross):
		digit_regex = '\d+\.?\d*e?-?\d*?'
		if percent != "" and valid != "" and gross == "":
			num = float(percent.strip("%"))
			value = re.findall(digit_regex, valid)[0]
			unit = valid.strip(value)  # type = str
			if unit == "ML":
				final_num = round(num*float(value)*10, 3)
				result = str(final_num) + "MG"
			elif unit == "MG":
				final_num = num*float(value)*0.01
				result = str(final_num) + "MG"
			else:
				result = unit

		elif percent != "" and valid!= "" and gross != "":
			result = ""

		else:
			result = percent
		return result

	frame = { "percent": percent, "valid": valid, "gross": gross }
	df = pd.DataFrame(frame)
	df["RESULT"] = df.apply(lambda x: percent_calculation(x["percent"], x["valid"], x["gross"]), axis=1)
	return df["RESULT"]



def spec_standify(df):
	# df = df.withColumn("SPEC_ORIGINAL", df.SPEC)
	df = df.withColumn("SPEC", regexp_replace("SPEC", r"(万)", "T"))
	df = df.withColumn("SPEC", regexp_replace("SPEC", r"(μ)", "U"))
	df = df.withColumn("SPEC", upper(df.SPEC))
	df = df.replace(" ", "")
	# df = df.withColumn("SPEC_gross", regexp_extract('SPEC', spec_regex, 2))
	# 拆分规格的成分s
	df = df.withColumn("SPEC_percent", regexp_extract('SPEC', r'(\d+%)', 1))
	df = df.withColumn("SPEC_co", regexp_extract('SPEC', r'(CO)', 1))
	spec_valid_regex =  r'([0-9]\d*\.?\d*\s*[A-Za-z]*/?\s*[A-Za-z]+)'
	df = df.withColumn("SPEC_valid", regexp_extract('SPEC', spec_valid_regex, 1))
	spec_gross_regex =  r'([0-9]\d*\.?\d*\s*[A-Za-z]*/?\s*[A-Za-z]+)[ ,/:∶+\s][\u4e00-\u9fa5]*([0-9]\d*\.?\d*\s*[A-Za-z]*/?\s*[A-Za-z]+)'
	df = df.withColumn("SPEC_gross", regexp_extract('SPEC', spec_gross_regex, 2))
	spec_third_regex = r'([0-9]\d*\.?\d*\s*[A-Za-z]*/?\s*[A-Za-z]+)[ /:∶+\s]([0-9]\d*\.?\d*\s*[A-Za-z]*/?\s*[A-Za-z]+)[ /:∶+\s]([0-9]\d*\.?\d*\s*[A-Za-z]*/?\s*[A-Za-z]+)'
	df = df.withColumn("SPEC_third", regexp_extract('SPEC', spec_third_regex, 3))

	pure_number_regex_spec = r'(\s\d+$)'
	df = df.withColumn("SPEC_pure_number", regexp_extract('SPEC', pure_number_regex_spec, 1))

	digit_regex_spec = r'(\d+\.?\d*e?-?\d*?)'
	df = df.withColumn("SPEC_gross_digit", regexp_extract('SPEC_gross', digit_regex_spec, 1))
	df = df.withColumn("SPEC_gross_unit", regexp_replace('SPEC_gross', digit_regex_spec, ""))
	df = df.withColumn("SPEC_valid_digit", regexp_extract('SPEC_valid', digit_regex_spec, 1))
	df = df.withColumn("SPEC_valid_unit", regexp_replace('SPEC_valid', digit_regex_spec, ""))
	df = df.na.fill("")
	df = df.withColumn("SPEC_valid", transfer_unit_pandas_udf(df.SPEC_valid))
	df = df.withColumn("SPEC_gross", transfer_unit_pandas_udf(df.SPEC_gross))
	df = df.drop("SPEC_gross_digit", "SPEC_gross_unit", "SPEC_valid_digit", "SPEC_valid_unit")
	df = df.withColumn("SPEC_percent", percent_pandas_udf(df.SPEC_percent, df.SPEC_valid, df.SPEC_gross))
	df = df.withColumn("SPEC_ept", lit("/"))
	df = df.withColumn("SPEC", concat( "SPEC_percent", "SPEC_ept", "SPEC_valid", "SPEC_ept", "SPEC_gross", "SPEC_ept", "SPEC_third")) \
					.drop("SPEC_ept", "SPEC_percent", "SPEC_co", "SPEC_valid", "SPEC_gross", "SPEC_pure_number", "SPEC_third")
	return df

@pandas_udf(StringType(), PandasUDFType.SCALAR)
def dos_pandas_udf(valid, number, dos):
	def dos_calculation(valid, number, dos):
		digit_regex = r'\d*.*\d+'
		if valid != "" and number != "" and dos != "":
			number = int(float(number))
			value = re.findall(digit_regex, valid)[0]
			unit = valid.strip(value)  # type = str
			final_num = round(number*float(value), 2)
			result = str(final_num) + unit
			# result = str(final_num)
		else:
			result = valid

		return result

	frame = { "number": number, "valid": valid, "dos": dos }
	df = pd.DataFrame(frame)
	df["RESULT"] = df.apply(lambda x: dos_calculation(x["valid"], x["number"], x["dos"]), axis=1)
	return df["RESULT"]



def similarity(df):
	df = df.withColumn("SIMILARITY", \
					(df.EFFTIVENESS_MOLE_NAME + df.EFFTIVENESS_PRODUCT_NAME + df.EFFTIVENESS_DOSAGE \
						+ df.EFFTIVENESS_SPEC + df.EFFTIVENESS_PACK_QTY + df.EFFTIVENESS_MANUFACTURER))
	windowSpec = Window.partitionBy("id").orderBy(desc("SIMILARITY"), desc("EFFTIVENESS_MOLE_NAME"), desc("EFFTIVENESS_DOSAGE"), desc("PACK_ID_STANDARD"))
	df = df.withColumn("RANK", row_number().over(windowSpec))
	df = df.where((df.RANK <= 5) | (df.label == 1.0))
	# df.repartition(1).write.format("parquet").mode("overwrite").save("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/qilu/0.0.3/result_analyse/all_similarity_rank5")
	# print("写入完成")
	return df


def hit_place_prediction(df, pos):
	return df.withColumn("prediction_" + str(pos), when((df.RANK == pos), 1.0).otherwise(0.0))
	# return df.withColumn("prediction_" + str(pos), when((df.RANK == pos) & (df.label == 1.0), 1.0).otherwise(0.0))


@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def dosage_replace(dosage_lst, dosage_standard, eff):

	frame = { "MASTER_DOSAGE": dosage_lst, "DOSAGE_STANDARD": dosage_standard, "EFFTIVENESS_DOSAGE": eff }
	df = pd.DataFrame(frame)

	df["EFFTIVENESS"] = df.apply(lambda x: 1.0 if ((x["DOSAGE_STANDARD"] in x["MASTER_DOSAGE"]) ) \
											else x["EFFTIVENESS_DOSAGE"], axis=1)

	return df["EFFTIVENESS"]


def mole_dosage_calculaltion(df):
	
	def jaro_similarity(s1, s2):
		# First, store the length of the strings
		# because they will be re-used several times.
		len_s1, len_s2 = len(s1), len(s2)

		# The upper bound of the distance for being a matched character.
		match_bound = max(len_s1, len_s2) // 2 - 1

		# Initialize the counts for matches and transpositions.
		matches = 0  # no.of matched characters in s1 and s2
		transpositions = 0  # no. of transpositions between s1 and s2
		flagged_1 = []  # positions in s1 which are matches to some character in s2
		flagged_2 = []  # positions in s2 which are matches to some character in s1

		# Iterate through sequences, check for matches and compute transpositions.
		for i in range(len_s1):  # Iterate through each character.
			upperbound = min(i + match_bound, len_s2 - 1)
			lowerbound = max(0, i - match_bound)
			for j in range(lowerbound, upperbound + 1):
				if s1[i] == s2[j] and j not in flagged_2:
					matches += 1
					flagged_1.append(i)
					flagged_2.append(j)
					break
		flagged_2.sort()
		for i, j in zip(flagged_1, flagged_2):
			if s1[i] != s2[j]:
				transpositions += 1

		if matches == 0:
			return 0
		else:
			return (
				1
				/ 3
				* (
					matches / len_s1
					+ matches / len_s2
					+ (matches - transpositions // 2) / matches
				)
			)

	@udf(returnType=DoubleType())
	def jaro_winkler_similarity(s1, s2, p=0.1, max_l=4):
		if not 0 <= max_l * p <= 1:
			print("The product  `max_l * p` might not fall between [0,1].Jaro-Winkler similarity might not be between 0 and 1.")

		# Compute the Jaro similarity
		jaro_sim = jaro_similarity(s1, s2)

		# Initialize the upper bound for the no. of prefixes.
		# if user did not pre-define the upperbound,
		# use shorter length between s1 and s2

		# Compute the prefix matches.
		l = 0
		# zip() will automatically loop until the end of shorter string.
		for s1_i, s2_i in zip(s1, s2):
			if s1_i == s2_i:
				l += 1
			else:
				break
			if l == max_l:
				break
		# Return the similarity value as described in docstring.
		return jaro_sim + (l * p * (1 - jaro_sim))

	# 给df 增加一列：EFF_MOLE_DOSAGE
	df_dosage_explode = df.withColumn("MASTER_DOSAGES", explode("MASTER_DOSAGE"))
	df_dosage_explode = df_dosage_explode.withColumn("MOLE_DOSAGE", concat(df_dosage_explode.MOLE_NAME, df_dosage_explode.MASTER_DOSAGES))
	df_dosage_explode = df_dosage_explode.withColumn("jws", jaro_winkler_similarity(df_dosage_explode.MOLE_DOSAGE, df_dosage_explode.PRODUCT_NAME_STANDARD))
	df_dosage_explode = df_dosage_explode.groupBy('id').agg({"jws":"max"}).withColumnRenamed("max(jws)","EFF_MOLE_DOSAGE")
	df_dosage = df.join(df_dosage_explode, "id", how="left")
	
	return df_dosage

@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def prod_name_replace(eff_mole_name, eff_mnf_name, eff_product_name, mole_name, prod_name_standard, eff_mole_dosage):

	def jaro_similarity(s1, s2):
		# First, store the length of the strings
		# because they will be re-used several times.
		len_s1, len_s2 = len(s1), len(s2)

		# The upper bound of the distance for being a matched character.
		match_bound = max(len_s1, len_s2) // 2 - 1

		# Initialize the counts for matches and transpositions.
		matches = 0  # no.of matched characters in s1 and s2
		transpositions = 0  # no. of transpositions between s1 and s2
		flagged_1 = []  # positions in s1 which are matches to some character in s2
		flagged_2 = []  # positions in s2 which are matches to some character in s1

		# Iterate through sequences, check for matches and compute transpositions.
		for i in range(len_s1):  # Iterate through each character.
			upperbound = min(i + match_bound, len_s2 - 1)
			lowerbound = max(0, i - match_bound)
			for j in range(lowerbound, upperbound + 1):
				if s1[i] == s2[j] and j not in flagged_2:
					matches += 1
					flagged_1.append(i)
					flagged_2.append(j)
					break
		flagged_2.sort()
		for i, j in zip(flagged_1, flagged_2):
			if s1[i] != s2[j]:
				transpositions += 1

		if matches == 0:
			return 0
		else:
			return (
				1
				/ 3
				* (
					matches / len_s1
					+ matches / len_s2
					+ (matches - transpositions // 2) / matches
				)
			)


	def jaro_winkler_similarity(s1, s2, p=0.1, max_l=4):
		if not 0 <= max_l * p <= 1:
			print("The product  `max_l * p` might not fall between [0,1].Jaro-Winkler similarity might not be between 0 and 1.")

		# Compute the Jaro similarity
		jaro_sim = jaro_similarity(s1, s2)

		# Initialize the upper bound for the no. of prefixes.
		# if user did not pre-define the upperbound,
		# use shorter length between s1 and s2

		# Compute the prefix matches.
		l = 0
		# zip() will automatically loop until the end of shorter string.
		for s1_i, s2_i in zip(s1, s2):
			if s1_i == s2_i:
				l += 1
			else:
				break
			if l == max_l:
				break
		# Return the similarity value as described in docstring.
		return jaro_sim + (l * p * (1 - jaro_sim))


	frame = { "EFFTIVENESS_MOLE_NAME": eff_mole_name, "EFFTIVENESS_MANUFACTURER_SE": eff_mnf_name, "EFFTIVENESS_PRODUCT_NAME": eff_product_name,
			  "MOLE_NAME": mole_name, "PRODUCT_NAME_STANDARD": prod_name_standard, "EFF_MOLE_DOSAGE": eff_mole_dosage,}
	df = pd.DataFrame(frame)

	df["EFFTIVENESS_PROD"] = df.apply(lambda x: max((0.5* x["EFFTIVENESS_MOLE_NAME"] + 0.5* x["EFFTIVENESS_MANUFACTURER_SE"]), \
									# (x["EFFTIVENESS_PRODUCT_NAME"])), axis=1)
								(x["EFFTIVENESS_PRODUCT_NAME"]), \
								(jaro_winkler_similarity(x["MOLE_NAME"], x["PRODUCT_NAME_STANDARD"])), \
								(x["EFF_MOLE_DOSAGE"])), axis=1)

	return df["EFFTIVENESS_PROD"]

@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def pack_replace(eff_pack, spec_original, pack_qty, pack_standard):

	frame = { "EFFTIVENESS_PACK_QTY": eff_pack, "SPEC_ORIGINAL": spec_original,
			  "PACK_QTY": pack_qty,  "PACK_QTY_STANDARD": pack_standard}
	df = pd.DataFrame(frame)

	df["EFFTIVENESS_PACK"] = df.apply(lambda x: 1.0 if ((x["EFFTIVENESS_PACK_QTY"] == 0.0) \
														& ("喷" in x["PACK_QTY"]) \
														& (x["PACK_QTY"] in x["SPEC_ORIGINAL"])) \
											else x["EFFTIVENESS_PACK_QTY"], axis=1)

	return df["EFFTIVENESS_PACK"]


@pandas_udf(StringType(), PandasUDFType.SCALAR)
def manifacture_name_en_standify(en):
	frame = {
		"MANUFACTURER_NAME_EN_STANDARD": en,
	}
	df = pd.DataFrame(frame)

	# @尹 需要换成regex
	df["MANUFACTURER_NAME_EN_STANDARD_STANDIFY"] = df["MANUFACTURER_NAME_EN_STANDARD"].apply(lambda x: x.replace(".", " ").replace("-", " "))
	return df["MANUFACTURER_NAME_EN_STANDARD_STANDIFY"]


@pandas_udf(ArrayType(StringType()), PandasUDFType.SCALAR)
def manifacture_name_pseg_cut(mnf):
	frame = {
		"MANUFACTURER_NAME_STANDARD": mnf,
	}
	df = pd.DataFrame(frame)
	# lexicon = df_lexicon.tolist()  # type = list
	# lexicon = df_pd["lexicon"].tolist()  # type = list
	
	# access_key = os.getenv("AWS_ACCESS_KEY_ID")
	# secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
	
	# s3_client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
	# object_file = s3_client.get_object(Bucket='ph-max-auto', Key="2020-08-11/BPBatchDAG/refactor/zyyin/lexicon_csv/0.0.1/part-00000-1fe947d1-fd07-4f0b-a54d-2e64568715fd-c000.csv")
	# data = object_file['Body'].read()
	# pd_df = pd.read_csv(io.BytesIO(data), encoding='utf-8')
	# lexicon = pd_df["lexicon"].tolist()
	
	
	# “南新”是一个高分词，但“海南新中正”可能被拆分为“南新”，所以目前lexicon里面去掉了部分长度小的字段
	lexicon = ['康福来', '一洋', '中美史克', '宝瑞坦', '华信', '金城', '钟根堂', '华裕', '盛基', '佳泰', 'ROHTO', 'AJINOMOTO', 'SHIRE', '海力生', '绿金子', '诚济', '杨森', '圣特', '立方', '诺捷', '道君', '格瑞仕特', ' 优科', '康诺生化', '雪龙海姆普德', '手心', '阿兹奈米特尔法布里克', '博诚', '兴华', '益盛', '百科', '五行', '红日', 'LISAPHARMA', '城市', '3M', '中核高通', 'KNOLL', 'SCHWARZ', '南少林', '八达', '侨光', '新汇', '威瑞斯', '仁和堂', '京豫', '志鹰', '方强', '泰尔茂', '量子高科', '爱的发', '天天乐', '国森', '隆信', '泛德', '莱博通', '神威', '瑞格', '成田', '正清', '中威', 'Amgen', '立华', '国海', 'SHIGAKEN', ' 齐康', '汾河', '唯科', '第四', '太阳石', '美吉斯', '盛泰', 'KYOWA HAKKO KIRIN', '费森尤斯卡比', '秦武田', '托毕西', '康普', '龙华', '金城金素', '天津药业焦作', '龙江龙', '凯健', '澳亚', '华懋', '金柯', ' 康希', '蓝十字', '天和', '富邦', '儿童', '松鹤', '珍宝岛', '国丹', '神龙', '德远', '中侨', '倍的福', '睿鹰先锋', '可可康', '燕京', '雅柏', '恒达', '宝庆隆', '山东省方明', '振东', '德林', '达华', '正达', '爱生', '和明', '复宏汉霖', '北大维信', 'BIOTEST', '健朗', '曙光', '方盛', '北京双鹤', '紫竹', '南疆', '奇林', '益健', '仙乐', '博士', '一格', '山东健康', '四药', '重庆大新', '杰华', '普什', '美创', 'KREUSSLER', '赛诺菲-安万特', '南国', '赛康', '维康子帆', '亿能普', '美诺华', '赛金', '新郑', '第七', '卓峰', '海神联盛', '曼秀雷敦', '长青', '山东省平原', '利拉斯', '辰星', '拜迪', '华瑞', '卫康', '德康', '恒 伟', '钟山', 'SOLCO', '美伦', '诺达', '富春', '康田', '青阳', '裕松源', '高盛亚', 'BESINS-ISCOVESCO', '拉丰', 'SANOFIPASTEUR', '法马卡', '莫高', '泰邦', 'GENZYME', '永昇', '力诺', '巴中普瑞', 'HEMONY', ' 红星', '民族', '青春康源', '海容唐果', '杏辉天力', '锦华', '龙州', '汉方现代', '光明', '山香', '参天', 'TOBISHI', 'NIPPON KAYAKU', '山东东明', '同方', '西莫', 'LIFEPHARM', '青春宝', '三诺生物', '大海阳光', '水产', 'SCHAPER & BRUMMER', '博洲', '元宁', '缔谊', '益康', '誉衡', '华良', '金峰', '兴邦', '澜泰', '利沙', '翔通', '千金湘江', '巨都药业', '安科余良卿', '葡立', '健信', '卓泰', '怀庆堂', '迪龙', '感康', '中丹', '安迪科正', '通天河', 'NANKUANG', '赛生', '万裕', '华沙', '太湖美', '新致', '菊乐', '泰格', '欧立', '华中药业', '东盟', '精华', '碧迪', '恩威', '海恩', '华鼎', '信东生技', '北医联合', '永生堂', '五洲', 'BERNA', '扁鹊', '凤凰', '华富', '景达', '福尔泰', '双药', '润阳', '东抚', '雪兰诺', '玉威', '三精升和', '旭华', '鼎泰', '铜鼓仁和', '未名生物', '北卫药业', '博恩特', '明龙', '天森', '平光', 'FLEET', '双伟', '资福', '子仁', '天恩', '尖峰', '安科生物', '赛特力,碧兰', '海洋渔业', 'Alcon', '金城泰尔', '盛迪', '通德', '基诺', '赛诺菲安万特民生', '安诺', '福乐康', '鹏鹞', '鲁西', '输血', '圣华', '正大天晴', '悦康凯悦', '活力素', '中昊', '天方', '信立邦', '东风', '冀衡', '汇宇', '金药师', '红星药业', '诺亚荣康', '远大德天', '海王', '信邦', '致君', '华夏', '南京东捷', '小营', '亿华', '达因儿童', '仁会', '思富', '新宝', '华立达', '精优', '史克', '云南植物', '瑞得合通', '益民', '花海', '康源', '倍特', '远大医药', '知原', 'ZHONGXIN', '迈迪', '天药本草堂', '精鑫', 'MYLAN', '特丰', '振兴', '矿维', '瑞新', ' 省', '太龙', '绿因', '旭东海普', '京西双鹤', 'UrsaPharm', '恩泽嘉事', '民生', '天心', '生生', 'SPECTRUM', '麦林克罗', '三晋', '白鹿', '兴齐', 'TEVA', '南诏', '博奥', '东日', '天元', '赣南制药', '威仕生物', 'ACTAVIS', '诺华', '库尔', '协和', '汉信', '裕源', '拜特', '江世', '涟水', '本天', '弘益', '蓝天', '小林', '达 因儿童', '云南白药', '天洋', '百路达', '山东信谊', '希埃', '三三', '天宏', '泓春', '九华', '仁齐', '九瑞健康', '帝斯曼', '维康', '医大', 'SAMCHUNDANG', '汉河', '尧王', 'NIPPON SHINYAKU', '依科', '中健康桥', '上药中西', '康特能', '双鹭', '华立', '陇神,戎发', '同盟', '万禾', '迈兰', '大幸', 'Kowa', '本草', '天马', '柏海', '同路', '夏尔', '希望', '艾迪', '瑞迪', '华润双鹤', '红星葡萄', '三菱化学', '青平', '珍奥', 'MEDA', '润丰', '爱科', '泰复', '津升', '大洋', 'J&J', '美商', '长征', 'CHIESI', 'RHODIA', '圣华曦', '都邦', '正和', '汉堂', '力菲克', '彼迪', '诺得胜', '中化联合', '李时珍', '奇尔康', 'RUIBO', '格瑞', '道润', '包头中药', '必康嘉隆', '安国', 'TEMMLER', '澳利达奈德', '诺德', '莱阳生物化学', '方大', '只楚', '华药南方', '快好', '柳韩', '药业', '江山', '朗肽', '阿尔法希格玛', '利群', '阿尔法玛', 'MERCK', '峨眉山健康', 'PIERRE FABRE', '和平', '中外', '潜龙', '双鹤药业', 'BOEHRINGER INGELHEIM', '凯合', '天年', '海康', '灵方', '田边三菱', '巨能乐斯', '九正', '保定古城', '鼎复康', '健民', '玉瑞', '万和', '梯瓦', '金立源', '金山禅心', '华润高科', '东阳光', '新世纪', 'ALPHAPHARM', '赛升', '明华', 'KADE', '绿谷', '海特', '贝斯迪', '日东电工', '中孚', '北大药业', '天安', '葫芦娃', '新五丰', '颈复康', '海纳', '功达', '谷红', '赞邦', '御室', '金赛', '龙翔', '新春都', '原子科兴', '西藏藏药', 'UCB', 'TAD', '新和成皖南', '恒诚', 'KUKJE', '易舒特', '天铖', '欣峰', '九泰', '鲁抗大地', '通药制药', '中原瑞德', 'JANSSEN', '亨达', '赛特力-碧兰', '康达', '百宫', '云阳', '闪亮', '老山', '恒昌隆', '天银', 'MITSUBISHI CHEM.', '云门', '赛保尔', '博爱', '东宝', '佳辰', '宣秦海门', '澳美', '金蟾生化', '中峘本草', 'AWD', '柏强', 'NYCOMED', '希尔康', '威高泰尔茂', '翔宇', '康尔佳', '长澳', '未名新鹏', '利奥', 'NOVO NORDISK', 'MERCK SHARP&DOHME', '滇虹', '福安', '国金', '金沙', 'JEOU SONG', '海思科', '罗益', '美国制药', '大红鹰恒顺', '奈科明', '霸科', '强基', '荷普', '天圣', '洞庭', '久光', '超群', '关爱', '仁民', '津新', '新宝源', '辰欣', '百奥泰', '银建', '山河', '中美天津史克', '金砖', '绅泰春', '天真', '锐擘', '鑫齐', '迪冉', '腾瑞', '唐氏', '统华', '千金协力', '华兰生物疫苗', '扬州生物化学', '黄河中药', '康蒂尼', ' 爱森斯', '前沿', '得安', '天大', '华澳', '艾美罗', '奥力', '利君精华', '三药', '顺鑫祥云', '灵康', '河洛太龙', '九连山', '福源', '灵泰', '健友', '新华', '悉普拉', '三大', '中国医科大学', '亚东', '亚泰', '德商', '广春', '恩华', '麦迪森', '新高', '正大海尔', '灵源', '新宇', '新赣江', '羚锐生物', '艾康', '绿竹', '双吉', 'ALPHARMA', 'Salutas', '九明', '金页', '大中', '康润', '七星山', '昆药', '药联', '亿邦', '广东百科', '生达', '玉川', '万源', '中宝', '骅成', '韩美', '华神生物', '辉南长龙', '锦帝九州', '新黄河', '乐康美的澜', 'SANDOZ', '圣永', '博士曼', '脏器', '以岭', '健生', '久安', '钧安', 'GEDEON RICHTER', '新东日', '信达生物', '控制疗法', '得恩德', '山东绿叶', 'Bayer', '京丰', '康宁', 'LEIRAS', '景峰', '仁德', '聚仁堂', '天之海', '天星', '庆安', '延安', '科创', '千汇', '亿胜', '百奥', '康复',\
	'福抗', '誉东', '九源', '康泰', '安康', '颐生堂', '金鹿', '津康', '三宝', '杭康', '弘盛', '津华晖星', '天一秦昆', '阿兹奈米特', '白云山汤阴东泰', '成都大西南', 'SANOFI', '紫杉园', '苏泊尔南洋', '绿叶', '诺迪康', '福星', '大围山', '泰平', '福瑞达生物', '瑞年前进', '广西医科', '医科大学生物', '鲁抗', '多多', '天泉', '生缘', '济南东方', '莱士', '三金龙', '三才', '第一生物', '生物制品', '赛诺菲民生', '艾希德', '岭南', '康缘', '金马', '科尔沁', '新五洲', '银湖', '益生源', '药都制药', 'Astellas', 'Eisai', 'DR.FALK', '三蓝', '双成', '恩瑞特', '天峰', '新帅克', 'MEIJI', '亚峰', '九洲龙跃', '哥白尼', 'PIZFER', 'NOVO-NORDISK', '百维', '威智', '蜀中', '天药', '晨牌药业', '万嘉', '华信生物', 'ISEI', '贝克诺顿', '巴斯德', '太平洋', '富祥', '大熊', '宏冠', '华北', '灵豹', '味邦', '百济', 'ASPEN', '吉瑞', '中泰', '东方广诚', '芜湖康奇', 'SCHERING PLOUGH', '百信', '惠氏', '金诺', '丽珠', '倍奇', '眼力健', '德福康', '新东港', '亚大', '兰州生物', 'ARZNEIMITTELF', '先锋', '安琪酵母', '博雅', '地奥', '葛兰素史克', '泽尼康', '济安堂', '长联杜勒', '绿丹', '菲德力', '井田', '启瑞', 'NORGINE', '大政', '英格莱', '拜耳先灵', '希尔康泰', '西峰', '远程', '广仁', '施美', '大禹', '力生', '奇正藏药', '华天', '卫生材料', '江西生物', '遂成', '皇隆', '振澳', '奥达', '罗维', '海辰', '福达', '中新', '民鑫', '博士伦', '汕头金石', '禾丰', 'ARTSANA', 'TRIYASA NAGAMAS', '优胜美特', '神经精神病', '龙灯瑞迪', '远力健', '利祥', '上海华氏', 'DAEWON', '住友', '玉安', '新马', '祈健', '达因', '厚生天佐', '长联来福', '嘉德', '赛诺菲圣德拉堡', '舍画阁', '敖东', '明仁', '美优', '三顺', '东陵', '龙泽', '中方', '莱美', '威奇达', '凯茂生物', '世贸天阶', '医药', '冀南', '华神', '本真', '明治', 'KYOWA HAKKO', 'REDDYS', '欧曼福德', '司邦得', '欧加农', '麦迪海', 'SERVIER', '兰药', '鲁亚', '派斯菲科', 'SK', '可济', '国润', '新亚', '海默尼', '生物谷', '植恩', '夏都', 'Arzneittel', '白云山和记黄埔莱达', '康正', '中佳', '千里明', '天福', '长清', '香菊', '石家庄制药', '荣昌', '华仁太医', '五景', '贝达', '正鑫', '爱大', '天士力', '龙晖', 'Geistlich', '昂生', '新星', '信立泰', '顶立', '星斗', '圣和', '华立南湖', '天宇', 'NARCO-MED', '仁合益康', '华康药业', '帝益', '樟树', '衡山', '宇惠', '美大康', '法隆', '粤华', '迪博', '华润赛科', 'ORGANON', 'J & J', '红蝶', '汇天', '长生生物', '华宇', '格兰百克', '三洋', '鲁南贝特', '普康', '强生', '天和惠世', '云鹏', '安宝', '颐和', '奥克特珐玛', '嘉逸', '江中', '康美', 'BECTON DICKINSON', '安必生', '安徽', '上海血液', 'ALL Medicus', '全新生物', '经济特区', '欣明达', '三九', '鼎昌', '湘雅', '中凯', '广信', '陇药', 'NIPPON', 'TAE JOON', '康乐', 'CHEMINOVA', 'BEECHAM', '亚邦生缘', '先求', 'ASTRAPIN', 'PROMEDICA', '海南卓泰', '朝药', '鑫安科技', 'LILLY', '特宝', '世纪', '爱威', 'RECORDATI', '绮丽', '同人泰', '第一三共', 'DAIICHI SANKYO', '南药', '维尔康', '杰瑞', '立健', '华鲁', '金蟾', '回元堂', '华兰生物工程', '长征富民', '英科新创', '卫材', '博瑞', '百利天华', '古田', '尔康', '仙草堂', '普德', '盘龙', 'ROTTA', '龙润', '长红', '汇丰', '康尔佳生物', '比切姆', '事业所', '三元基因', '德药', '广州南新', '三联', '康恩贝', '赛隆', '千寿', '仙河', '福祈', '正方', '美时', 'HEINRICH MACK', 'JW', '奥星', '炎黄', '桔王', '滋贺', '禹王', '富康', 'CHUGAI', '康都', '艾富西', '人人康', '华素', '南方', 'POHL-BOSKAMP', '盖天力', '阳江', '景诚', '喜人', '国帝', '费森尤斯', '仁丰', '朗致', '永正', '大地', '健坤', '迪菲特', '赣药全新', '仁和', '百草', 'Laboratori Guidotti', '三精', '冠城', '百裕', '奥森', '龙桂', 'APOTEX', '晨光', '希力', '金山', 'PIRAMAL ENTERPRISES', 'Actavis', '乐尔康', '长天', '海口市制药', '格林', '萌蒂', '诚信', '上禾', '康瑞', '凯西', '烟台西苑', '益生', '滨湖双鹤', '安生凤凰', '保定金钟', 'FERRING', '妙音春', '爱尔海泰', '力胜', 'SERONO', '迪隆', '亚神', '回音必', '生物化学', '正康', '勃林格殷格翰', 'REDEL', '宏奇', '环宇', '川抗', '中杰', '江波', '南大', '海尔斯', '山东省东明', '旭晖', '第壹', '蓝绿康', '皇城相府', '意大泛马克', '市', '朗依', '塔赫敏', '华生', '德泽', 'SANKYO', 'JEIL', '中晟', '普丽尔', '君山', '惠丰', '亿利', '睿鹰', '盘古', '天吉生物', '华森', '九', '健赞生物', '泰合', '慈象', 'HELIXOR HEILMITTEL', 'ORION', '万顺堂', '斯达', '大元', '东北亚', '金日', '纽迪希亚', '合信', '茂祥', 'ZAMBON', '诺金', '基因泰克', '华润九新', '赫素', '香山堂', '珠江', '弘美', '大唐', '景德', '联合', 'Demo S.A.Pharm', '亚邦', '银诺克', '迪耳', '兴和', '恒生', '泰德', '辉瑞', '神农', '德众', '三益', '澋润', '振国', '安丁', '宝龙', '奥瑞特', '和盈', '爱德', '安士', '艾华', '中惠', 'IPSEN', '凯华', '伊顺', '盐城市第三', '天源', '渊源', '科瑞德', '维奥', '鲁北药业', '新化学', '环球', '可迪', '东信药业', '金兴', '山东新时代', '罗欣', '新兴', '裕腾', '集川', '佑华', '鑫善源', '宝鉴堂', '通和', '广生堂', '科顿', '海神', '东亚', '七台河', '津奉', '联邦', '同仁堂', '双奇', '大场', '康利', '华世丹', '开封制药', '第一药品', '卓谊生物', '华安', 'BERLIN-CHEMIE', '丹尼', '加柏', '宏远', '众生', '邦琪', '科威', '臣功', '山川', '星火', '微芯', 'FOSCAMA', '兰伯西', '一正', '首和金海', '金太阳', '恒康', 'GLAXOSMITHKLINE', '津兰', '阿斯彭', '白敬宇', '天道', '圣大', '长征-欣凯', 'STANDARD', '贝普', '鑫富', '古汉', '康博士', 'Arzneitt', 'Sandoz', '白求恩', 'M.S.D.', '惟精堂', '仁皇', '天龙', '盟生', '巨琪', '一康', 'BIOTON', '余杭', '天目', '金恒', '迈丰', '久松', '中华', '康泰生物', '峨嵋山', '威特', '元生', 'BRACCO', '迪瑞', '正大清江', 'BEN VENUE', '莎普爱思', '法玛西亚普强', '安国亚东', '上海新华联', '河丰', '圣保堂', '天目山', '利鑫', 'GILEAD', '华威', '禾氏', '乐普', '东宇', 'BIOGEN', 'MUNDI', '美瑞', 'SENJU', 'NITTO DENKO', '端纳', '康乃格', '东松', '圣元', '烟台巨先', '万晟', '鑫威格', '康辰', '麦康', 'S-P', '天地', '海泰', '中研', 'TOA', '田边', '武田', 'lek', '南光', '海伯尔', 'NOVARTIS', '翰辉', '嘉博', '新中正', 'SHIN POONG', '恒安', '仁悦', '金鸿', '汇元', 'PHAFAG', '德元', '宝东', '正美', '福斯卡玛', '一新', '万杰', '三精加滨', '海欣', '哈三联', '诺维诺', 'Cephalon', '柏林-化学', '红林', '世桥', '天赐福', '济生', '金龙', '花园', '济川', '迪赛诺', '华泰', '通园', '西克罗', '泛生', '通化东圣', '万泽', '哈药,总', '美达', '成纪', 'GENELEUK', '康业', '华民', '恒新', '长春北华', '百正', '华生元', '华瑞联合', '赛百诺', 'SALUTAS', '邕江', '好博', '惠普森', 'DONG-A', '素高', '兰花', '绵竹', 'DRUG RESEARCH', '春天', '民生药业', '益民堂', '中国药科大学', '绿岛', '参天堂', '韩林', '华北制药', '三石', '显锋', '博士达', '山之内', '鹏海', '瑞安', '新华达', 'Feel Tech', '雅立峰', '安联', '福乐', '氨凌',\
	'爱诺', '太安堂', '新南方,青蒿', '弘和', '杏林白马', '利君方圆', '协和发酵麒麟', '元和', '蓉生', 'FERRER', '良济', '赛立克', '日本中外', '圣邦', '金不换', '普生', 'URSAPHARM ARZNEIM', '神通', '锦瑞', '美图', '安进', '诺成', '黄浦', '希百寿', '西南合成', '四环制药', '瀚邦', '中玉', '联环', '美通', '泽众', 'GRACE', '方舟', '虹光', '皇象', '金伴', '国源', '旺林堂', ' 再鼎', 'AZ', 'JANSSEN-CILAG', '华立金港', '安万特', '诺诚', '凯威', '万岁', '普新', '富华', '新海康', '正大丰海', '澳迩', '飞鹰', '汉瑞', 'Alpharma', '康立', '国嘉', '礼达', '威玛舒培', '法杏', '一洲', ' 朱养心', '羚锐', '纽兰', '汉森', '中药', '积大', '柏阳', '赫尔', '太太', '远大蜀阳', '俞氏', '同济现代', '百会', '吉安三力', '普济', '罗浮山', '欣泰', '海南海药', '辉克', '晶珠', '金丝利', '厚普', 'CSL BEHRING', '蒂人', '三九同达', 'TRB', '灵芝宝', '联合化工', 'HEXAL', '华闽', '慧谷', '全星', '创健', '辅仁堂', '金坦', '高特', '科源', '运佳', '老桐君', '冯了性', '合源', '赫尔布兰德', '康寿', '云中', '永寿', '健康元', 'LEK', '中辰', '麒麟鲲鹏', 'OWEN MUMFORD', '安科恒益', '圣鹰', '安斯泰来', '雷龙', 'FARMAKA', '葛兰素威康', '半宙天龙', '欧化', '东联', '蓝图', '中皇', '福邦', '国大生物', '裕欣', '同联', '生命科技', '参花', '先声中人', '长兴', 'Alfasigma', 'SQUIBB', '济丰', '顺达四海', '延年', '庆发', '海力', '人福', '立达', '九阳', '南街村', '迪智', '南岳', '世康特', '南格尔', '万润', '日升昌', '佛都', '杨森-CILAG', 'SANOFI PASTEUR', '浦北', '朗圣', '精方', '普瑞', '斯奇', '华侨', '奥赛康', '海大科技', '恩成', '华西', '鸿基', '中化', '罗地亚', '善美', 'astellas', '蒙欣', '瀚晖', '大源', '明瑞', '桑海', '科泰', '瑞尔', '亨瑞达', '国正', '康密迪', '梅生', '福克', '灵广', '同源', '施慧达', 'TERUMO', '大恒', '北斗星', '泰康', '广联', '闽东力捷迅', '吉民', '白天鹅', '长安', '广济', '博大伟业', '山禾', 'KAKEN', 'SEIKO EIYO', '顿斯', '德峰', '西施兰', '国光生物', '味之素', '奥邦', 'MERZ', '大华', '天成', '吉贝尔', '悦兴', '圣火', 'LUXEMBERG', 'TAIHO', '华澄', '德元堂', '荣盛', '翰宇', '迪赛', '昂利康', '万正', ' 王牌速效', '在田', '海德润', '山东平原', '赛诺菲温莎', '扶正', '天津史克', '诺美', '汇中', '复升', '东星', '万隆', 'EDMOND', '赛林泰', '鼎恒升', '百年六福堂', '万通复升', '泰邦生物', '华新生物', 'Pharmacosmos', '同德', '海润', '海外', '盛通', '亨元', '清松', '新鹏生物', '瑞康', 'ESSETI', '大连生物', '张江生物', '新兴同仁', '阿尔法韦士曼', '华南', '健宁', '兴源', '济邦', '豫港', '赛诺', 'TAKEDA', '三才石岐', '北杰', '安捷', '博莱科', '同益', '株式会社', 'BAGO', '南湖制药', '盛辉', '通惠', '伊维', '日晖', '天诚', '久联', '八峰', '金迪克', '美国医药伙伴', '弘森', '天舜', '利君康乐', '远大国奥', '安贝特', '百泰', '海王英特龙', '皇甫谧', '鲁安', '山庆', '明日', '海通', '碧兰', '惠德勤', '明兴', '百灵', '国光', '哈尔滨三联', 'PROSTRAKAN', 'DAIICHI YAKUHIN', 'YPSOMED', '力玖', '卫达', '龙科马', '百慧生化', '九和', '摩罗丹', '天辰', '一半天', '吴中', '中兴', '高博京邦', '天朗', '星银', '福瑞', '东海', '施泰福', '众合', 'GRIFOLS', '宝利', 'CIPLA', '民意', '光正', '威门', 'BORYUNG', '大新', '聚荣', '杰特贝林', 'SUN STAR', '杨天', '威尔森', '南开允公', '万邦', '尚善堂', '万森', '上海新黄河', '南美', '山东德州医药', '奇灵', '东康', '微克西', '宝珠', '荣安', 'SUMITOMO', '兆兴', '现代生物', '经济特区生物化学', '白云山 制药总厂', '希格玛', '贞玉', '赛达', '世一堂', '恒久远', '凯蒙', '康立生', '万兴', '奥萨', '回天', '金诃藏药', '三超', '康宝', '雷允上', '大冢', '汉方', '国药集团容生', '永进', '灵佑', '润泽', '众瑞', '可可西里', '惠诺', '天德', '梅花', 'BRUSCHETTINI', '天瑞', '康弘药业', '叶开泰', '东北制药', '康乃尔', '康福生', '北方', '高科', '银朵兰', '众益', 'URSAPHARM Arzneimittel GmbH', '怡成', 'POLFA TARCHOMIN', 'DANIPHARM', '宏盈', '怡翔', '西林', '舒邦', '迪诺', '华阳', '华太', '美联', '宇斯', '卢森堡', '爱美津', '天骄', '正同', 'MENARINI', '佐力', '九发', '珐博进', '现代哈森', '恒瑞', '百特', '洛斯特', '智同', '泰丰', 'MALLINCKRODT', '华山', '台城制药', '兴安', '圣博康', '得能', '王清任', '实普善', 'Pharmacyclics', 'ACTELION', '春风', '默沙东', '歌礼', 'HANLIM', '万邦德', '润弘', '康沁', '健赞', '康泰生 物', ' 汇恩兰德', '科宝', '青山', '源生', '默克-雅柏', '豪森', '马应龙', '博山', '胜利', 'TWOBIENS', '皖北', '东诚', '三峡', '北生', 'BHC', '天衡', '天平胃舒平', '迪康药业', '阿奎坦', '三箭', '白云山', '鮀滨', '天誉', '爱活', '同一堂', '会好', '盐野义', '致和', '昂德', '克莱斯瑞', '龙城', '川力', 'WILLMAR SCHWABE', '科晖', '华津', '正茂', '英联', '汇仁', '梅峰', 'CELGENE', '海虹', '葡萄糖', '广福', '慧宝源', '一品红', '波尔法-塔赫敏', '默克雅柏', '费雷尔', '林恒', '地纳', '金泰', '普众', '金陵', '贝特', '大宇生化', '泰俊', 'MITSUBISHI TANABE', '保时佳', '嘉进', '神奇', '凯程', '珍视明', '百科亨迪', '光大', '康德莱', '御金丹', '方略', '康润生物', 'SANTEN', '丰泽', '泰乐', '梓橦宫', '天威', '长庆', '国镜', '康吉尔', '三爱', '楚天舒', '百慧', '利丰华瑞', '巨能', '基立福', 'ALFASIGMA', '恒星', '华普生', '卫伦', '利能康泰', '恒利', '仁合堂', '福人', '万通', '德成', 'KONSYL', '万泰沧海', '同达', '太极', '君安', '贝得', '亚太', '康刻尔', '九泓', '金创', '华迈士', '莱士血液', '亿帆', '五加参', 'D.R', '北大荒', '大西洋','恒祥', '科田', '国文', '兰太', 'GEISTLICH', '顺天', '乔源', '禾邦', 'LEURQUIN', '普华', '惠迪森', '凯润', '铭康', '康力', '湖北华中', '福瑞堂', 'SHIONOGI', '利生', '大佛', '福宁', '幸福', '赛而', '亚东启天', '信合援生', 'HOSPIRA', '宝树堂', '五洲通', '默克', '同济堂', 'YUHAN', '娃哈哈', '大亚', '银河', '救心', '上海克隆生物', '四环生物', '汇天生物', '赛诺维', '赛达生物', '万特尔', 'STADA', '乐托尔', '中合 ', '粤龙', '瀚钧', '合瑞', '益品', '积华', '华卫', '诺康', '化学', '万全万特', '绿十字', '誉隆亚东', '伊伯萨', '好医生', '必康制药', '贝克', 'janssen', '天府', '迪沙', '南洋', '长恒', '恒基', '健乔', '泉港', '平原', '康欣', '九州', 'ATLANTIC', '和泽', '民生滨江', '润和', '中西三维', '科兴', '空港', '苏威', '星群', '保灵', '创新', '利欣', '奥鸿', '新时代', '三才医药', '国为', '仙琚', '鸿鹤', '东泰', '科艾赛', '甘甘', 'LEO', '史达德', '优德', '罗特', '圣济堂', '普利', '午时', 'SUNOVION', '豪邦', '依比威', '协和发酵', '三生国健', 'BAYER', '泰华', '金宝', '百益', '新基', '陕西海天', 'STIEFEL', '天实', '兵兵',  '司艾特', '马博士', 'SP', '夏菩', 'I-SENS', '松鹿', '高华', '德源', '邦宇', '仟源', '步长', '久铭', '莱恩', '和记黄埔', '民生健康', '居仁堂爱民', '第一生化', 'ALCON', '白云山东泰商丘', '金星', '康华', '圣诺', '奥利安', '博尔纳', '飞弘', '新达',\
	'正浩', '红豆杉', '新世通', '齐都', '非凡', '罗裳山', '双林', '汇瑞', '唯森', '华诺', '城梓', 'PURZER', 'YUNG SHIN', '希杰', '天一堂', '永和', '何济公', '中宝曙光 ', '千红', '贝丽莱斯', '兰陵', '欧林', '泰姆勒', 'AGUETT', '圣都', '益品新五丰', '豫西', '秦岭', '苑东', '葡萄王', '中狮', '中洲', 'C.T.S.', '澳医', '药都仁和', '阿斯利康', '澳诺', '太洋', '麦氏', 'GENZYME POLYCLONAL', 'MAYOLY SPINDLER', 'HANMI', '武罗', '元森', 'Phamaceuticals', '恒金', '四长', '盱迂', '齐鲁', '华能', 'KOWA', '红旗', '明欣', '苏州东瑞', '秦巴', 'POLI', '兆科', '新张药', '大得利', '寿制药', '金虹胜利', '杨凌生物', '凯因科技', '上海信谊', '亚洲', 'LACTEOL', '高科技,卫健', '金岛', '湘中', '鼎业', '汉晨', '云峰', '开元', '东方', '绿洲', '石家庄四药', '石药', '潍坊制药', '百济神州', '三叶', '通用电气', '旭化成', '家和', '上海信宜', '雅培', '国瑞', '新丰', '福森', '吉斯凯', '信元', '葵花', '瑞年百思特', 'DUNNER', '林州', '双鼎', '彩虹', '台裕', 'ALMIRALL', 'OTSUKA', '利泰', '维威', '亿康', '康臣', '瑞全', '长源', '普华克胜', '鸿烁', 'IBSA', '铁力', '凯宝', 'WEIDAR', '吐比阶', '版纳药业', '草仙', '璞元', '景岳堂', '震元', '医创中药', '大药厂', '赫曼', '米雅利桑', '先通', '诺安', '金牛原大', '联合治疗', '京新', '华盛生物', '普安', '格拉雷', '科益', '晋阳', '鑫煜', '东乐', '帝斯曼,江山', '升和', '先灵葆雅', '星湖', '大力神', '华雨', 'DAEWOONG', '远大', '前锋', 'HISAMITSU', '海山', 'Demo', '开开', '北生研', '久和', '博康健', '波尔法', 'Pizfer', '博大', '赛克', '一品', '康诺', '赫士睿', 'NIPRO', '欣凯', 'ITALFARMACO', '凯利', '天泰', '双科', '熙德隆', '百琪', '一心', '端正', '永宁', '九洋', '启元', 'TEIJIN', '健能', '第八', '中联四药', '三叶美好', '同济奔达', '斯利安', '银谷世纪', '方圆', 'BAXTER', 'PIERRE ROLLLAND', '科伦', '哈药', '富东', '一成', '大红鹰', '大鹏', '创诺', '南粤', '普渡', '汉丰', 'DEMO', '新南山', '丹生生物', '东北六药', '国大药业', '施贵宝', '九天', '汇康', '葛洪堂', '倍绣', '博莱', '奇力', '东信医药', '华星', '九典', '龙泰', '圣朗', '力恩普', '华润金蟾', '济民可信', '康美保宁', '和盛堂', '鲁北生物', '和治', '天普', 'EVER Neuro Pharma GmbH', '星昊', '达尔玛', '中联', '九势', '虎泉', '勤奋', '远恒', '罗氏', '康亚', '京西', '康贺威', 'HAMOL', '法玛西亚', '兰生血液', '阿特维斯', 'PHARMACIA', '美罗', '马赫', 'AVENTIS', '纯正堂', '厚德', '中天', '佛慈', '朗天', '中山', '白医', '生化学', '金山生物', '先声生物', '奇运生', '逸舒', '海灵', '龙德', '民康', '利华', '黄河', '青峰', '美化', '海滨', 'CHEIL.JEDANG', '康莱特', '敬修堂', '新南方', 'EVER NEURO', '上海医药', 'BRAUN', '路易坡', 'FIBROGEN', '华海', '普尔康', '正科', 'MEDOCHEMIE', '医科', '亚联', '顺生', '雄飞利通', '康利达', '诚意', '为康', '申高', '北陆', '优华', '甘李', '黄氏', '滨海县制药', '泓源', '新药', '良福', '济民', '多瑞', '爱科来', '明仁福瑞达', '敬一堂', '长生基因', '利康', '大安', '龙海', '天顺', '川奇', '巨泰', '金远', '辉凌', '隆赋', '迈特兴华', '滏荣', '瑞森', '全宇', '鲁南', '黄海', '日本化药', '先声麦得津', '康友', 'KIRIN KUNPENG', '华青', '瑞昇', '百澳', '康和', '博森生物', 'AstraZeneca', '艾珂尔', '联谊', '北京首儿', '阳光', '丹霞', '三风友', '河北凯威', '成功', '丰生', '京通', '卫星', '恒大', '欣科', '太河', '艾美卫信', '余良卿', '山德士', '西南药业', '大有', '荷夫曼', '国康', '安信', '大同盟', '普利瓦', '豪运', '安泰', '新斯顿', 'Janssen', 'GUERBET', '联盛', '诚志', '康麦斯', 'A & Z', '白云', '安盛科兴', '辰龙', '凯夫', '丰安', '珠海润都', '威尔曼', '百泰生物', '森科', '东芝堂', '华龙', '康哲', '人健', 'MEPHA', '远景康业', '龙灯', '奥托康', '华润顺峰', '天地恒一', '双新', '福元', '赛特力', '河西', '益尔', '吉利德', '海宏', '捷众', '蓝汀', '康皓', '惠康', '威世', '第五', '亨利', '益普生', '百姓堂', '浦津林州', '安徽长江', '北医大', '珍宝', '长白山', '大自然', 'CORREVIO', '双鲸', 'GE', '沃森', '百克生物', '原子高科', '惠美佳', '晋新双鹤', '上药新亚', '保定孚泰', '利龄,恒泰', '利尔', '同道堂', '卫华', '天致', '科雷 沃', '礼来', '力卓', '莱柯', '先声', '鲁银', 'SUN', '高 博京邦', '康必得', '华宝通', '北大高科华泰', '科瑞', '心宝', '玉皇', '百康', 'MIYARISAN', '科发', '金洋', '崇明', 'RANBAXY', '无极', '海悦', '赛特多', '麦道甘美', '贝朗', '健今', '长江', '福满', '西岳', '国风', '万乐', '华新', '百特医疗', '川大华西', '永康', '永安', 'DR.GERHARD MANN', '益佰', '瑞福莱', 'LG', '保宁', 'WYETH', '吉春', '天康', '三金', '亚宝', '迈克', 'SKF', '海济', '正大', '美宝', '嘉林', '星海', 'GALDERMA', '金象', 'GENENTECH', '明鑫', '卡德', '渤健', '寿如松', '上药东英', 'RADIUM', '万德', '老拨云堂', '和仁堂', '景康', 'CEPHALON', 'EXCELLA', '施德', '祁连山', '飞云岭', '华诺威', '爱德福', '银涛', '立业', '东英', '泰华堂', 'PANBIOTIC', 'KINGDOM', '隆康', '安健', '比奥罗历加', '康弘生物', '华靳', '振东康远', 'LAFON', '济仁', '景民', '帅克', '三生', '第三', '特瑞', '迪奥', '万翔', '赛维', '药友', '优你特', '宁国国安邦宁', 'Actelion', 'OCTAPHARMA', '北大医药', '朋来', '北京首医', '四环药业', '云岗', '山东信宜', 'EVERS', '赛诺菲安万特', '十万山', '圣泰', '万邦天诚', '康赐尔', '东菱', '京卫', '山大康诺', '赛伦', '华广', '爱民', '石药银湖', 'Kreussler', '华仁', '金花', '依生', 'PLIVA', 'KOTOBUKI', '丹红', '杭州民生', '全安', 'Alpharma ApS','优克', '雷允 上', '百科达', '恒和维康', '艾美康淮', '祥瑞', '长城', '乌斯法码', '碧凯', '中瑞康芝', '汉王', '方心', '华邦', '风华', '罗达', '昊邦', '大德', '华润天和', '晨牌邦德', '三维生物', '宏业', 'FRESENIUS', '高德美', '科辉', '厚生', '天津天达', '易亨', '永大', '惠松', '普洛康裕', '华创', '源首', '万生', '中和', '第一', 'GENZYME BIOSURGERY', '多维', '晨菲', '鲁抗辰欣', '曙辉', '浙北', '希尔安', '民彤', '医创药业', '康缘桔都', '力思特', '利博福尼', '许瓦兹', '光华', '中南', '润邦', '源瑞', '华葆', '怀仁', '悦康', '赛诺菲', '复旦张江', '利福法玛', '拜耳', '蚌埠丰原涂山', '美君', '鹿王', '天济', '圣宝罗', '云南医药工业', '华源', '尼普洛', '联合赛尔', '施莱', '力邦', '乐信', '伊春', '维和', '金美济', '爱尔康', '皮尔法伯', '澳林', '绿色', '北京赛科', '光谱', '田田', '常乐', '德芮可', '先声东元', '民海','潜江', '华纳', '金健', '西点', '众康', '普罗斯特拉肯', '双基', '科力', '东方协和', '生晃荣养', '昊海生物', '优时比', 'PHARMACYCLICS', '九龙', 'SEPTODONT', '华灿', '顺峰', '世信', '新峰', '星鲨', '海达', '山姆士', '海斯', 'LOTUS', '华元', '奥吉娜', '东新', 'GRAPE KING', '永信', '利君', '医科大学制药', '百年汉克', '保定天浩', '海正', 'MADAUS', '闽海', '万寿堂', 'TANABE SEIYAKU', '赛卓', '中央', 'ABC', '瑞邦', '康远', 'BIONORICA', '天台山', 'S.R.L', '中桂', '维伍', '望子隆', '前进',\
	'安格', '施维雅', '福和', 'TRB Pharma S.A.', 'ASTRAZENECA', 'CHONG KUN DANG', '先声卫科', '辉力', 'ARKRAY', '金耀', '?', '万邦,生化', '二叶', '万特', '中美上海施贵宝', 'EBEWE', 'LUNDBECK', '赛诺菲-温莎', '康奇力', '香雪', '金华隆', '万辉双鹤', '苏中', 'UNITED THERAPEUTIC', '宝泰', '卫光', '博康', '沁义南通', '诺和诺德', '邦民', '乌苏里江', '安邦', '九旭', '辅仁药业', '瑞阳', '山东圣鲁', '天风', '金恺威', '铭鼎', '中科生物', '渝生', '和创', '华润三九', '特一', '必康', '赛诺菲-圣德拉堡', '一力', 'ALLERGAN', 'KYUSHIN', '威高', '仁仁', '灵北', '施图伦', '泰恩康', '龙生北药', '苏泊尔', '富兴', '博华', '为民', '奥贝泰克', '中科', '山东方明', '威智百科', '新宁', '吉通', '修正', '生宝', '宣泰海门', 'Ever Neuro', '爱可泰隆', '武汉血液', '万高', '济民可信山禾', '丰原', '欧氏', '路坦', '海鲸', '江原', '新百', '朝晖', '汇伦', '金世', '星吴四创', '泰利森', '先强', '中美华东', '新姿源', '佑众全椒', '福鼎', '天健', '屏山', '复兴', '浦金', '会仙', '紫光辰济', '舒泰神', '紫光', 'KOBAYASHI', '太阳', '政德', '中医药大学', '世彪', '金桥', '天煌', 'YUNGJIN', '长久', '海口制药', '飞马', 'ROCHE', '盈信', '管城', '正东', '普元', '海神同洲', '百时美施贵宝', '科鼎', '恒健', '欧替', '天麦', '利民制药', '保益', '宁国邦宁', '百利', '爱心', '人民', 'Fabrik Kreussler', '日中天', '明和', '伊赛特', '江中高邦', '奥克特', 'INGELHEIM', '葛仙翁', '神华', '盐城生缘','康芝', '培力', '百洋', '通用', '沱牌', '赛洛金', '天达生物', '康德', '柏赛罗', '复旦复华', '天山制药', '韩都', '恒拓', '昂泰', '成大', 'AMGEN', '端信堂', '宏济堂','扬子江']
	seg = pkuseg.pkuseg(user_dict=lexicon)

	df["MANUFACTURER_NAME_STANDARD_WORDS"] = df["MANUFACTURER_NAME_STANDARD"].apply(lambda x: seg.cut(x))
	return df["MANUFACTURER_NAME_STANDARD_WORDS"]




@udf
def cosine_distance_between_mnf(array):
	u = array[0].toArray()
	v = array[1].toArray()
	return float(numpy.dot(u, v) / (sqrt(numpy.dot(u, u)) * sqrt(numpy.dot(v, v))))


@pandas_udf(IntegerType(), PandasUDFType.SCALAR)
def dic_words_to_index(words):
	frame = {
		"WORDS": words
	}
	df = pd.DataFrame(frame)


	def is_geo_tag(w):
		t = analyse.extract_tags(w, topK=5, withWeight=True, allowPOS=("ns",))
		if len(t) == 0:
			return 0
		else:
			return 1

	df["GEO_TAG"] = df["WORDS"].apply(lambda x: is_geo_tag(x))
	return df["GEO_TAG"]


def phcleanning_mnf_seg(df_standard, inputCol, outputCol):
	# 2. 英文的分词方法，tokenizer
	# 英文先不管
	# df_standard = df_standard.withColumn("MANUFACTURER_NAME_EN_STANDARD", manifacture_name_en_standify(col("MANUFACTURER_NAME_EN_STANDARD")))
	# df_standard.select("MANUFACTURER_NAME_STANDARD", "MANUFACTURER_NAME_EN_STANDARD").show(truncate=False)
	# tokenizer = Tokenizer(inputCol="MANUFACTURER_NAME_EN_STANDARD", outputCol="MANUFACTURER_NAME_EN_WORDS")
	# df_standard = tokenizer.transform(df_standard)

	# 3. 中文的分词，
	
	# df_lexicon.printSchema()
	# lexicon_lst = 
	# 3.2 
	df_standard = df_standard.withColumn("MANUFACTURER_NAME_WORDS", manifacture_name_pseg_cut(col(inputCol)))

	# 4. 分词之后构建词库编码
	# 4.1 stop word remover 去掉不需要的词
	stopWords = ["高新", "化学", "生物", "合资", "中外", "工业", "现代", "化学制品" "科技", "国际", "AU", "OF", "US", "FR", "GE", "FI", "JP", "RO", "CA", "UK", "NO", "IS", "SI", "IT", "JA", \
				"省", "市", "股份", "有限", "总公司", "公司", "集团", "制药", "总厂", "厂", "药业", "责任", "医药", "(", ")", "（", "）", \
				 "有限公司", "股份", "控股", "集团", "总公司", "公司", "有限", "有限责任", "大药厂", '经济特区', '事业所', '株式会社', \
				 "药业", "医药", "制药", "制药厂", "控股集团", "医药集团", "控股集团", "集团股份", "药厂", "分公司", "-", ".", "-", "·", ":", ","]
	remover = StopWordsRemover(stopWords=stopWords, inputCol="MANUFACTURER_NAME_WORDS", outputCol=outputCol)

	return remover.transform(df_standard).drop("MANUFACTURER_NAME_WORDS")


@pandas_udf(ArrayType(IntegerType()), PandasUDFType.GROUPED_AGG)
def word_index_to_array(v):
	return v.tolist()


def words_to_reverse_index(df_cleanning, df_encode, inputCol, outputCol):
	df_cleanning = df_cleanning.withColumn("tid", monotonically_increasing_id())
	df_indexing = df_cleanning.withColumn("MANUFACTURER_NAME_STANDARD_WORD_LIST", explode(col(inputCol)))
	df_indexing = df_indexing.join(df_encode, df_indexing.MANUFACTURER_NAME_STANDARD_WORD_LIST == df_encode.WORD, how="left").na.fill(7999)
	df_indexing = df_indexing.groupBy("tid").agg(word_index_to_array(df_indexing.ENCODE).alias("INDEX_ENCODE"))

	df_cleanning = df_cleanning.join(df_indexing, on="tid", how="left")
	df_cleanning = df_cleanning.withColumn(outputCol, df_cleanning.INDEX_ENCODE)
	df_cleanning = df_cleanning.drop("tid", "INDEX_ENCODE", "MANUFACTURER_NAME_STANDARD_WORD_LIST")
	return df_cleanning


def mnf_encoding_index(df_cleanning, df_encode, spark):
	# 增加两列MANUFACTURER_NAME_CLEANNING_WORDS MANUFACTURER_NAME_STANDARD_WORDS - array(string)
	# 读取df_lexicon
	df_lexicon = spark.read.parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/lexicon/0.0.1")
	df_pd = df_lexicon.toPandas()  # type = pd.df
	print("分词测试")
	df_cleanning.printSchema()
	
	df_cleanning = phcleanning_mnf_seg(df_cleanning, "MANUFACTURER_NAME_STANDARD", "MANUFACTURER_NAME_STANDARD_WORDS")
	df_cleanning = phcleanning_mnf_seg(df_cleanning, "MANUFACTURER_NAME", "MANUFACTURER_NAME_CLEANNING_WORDS")
	df_cleanning = df_cleanning.withColumn("MANUFACTURER_NAME_STANDARD_WORDS_SEG", df_cleanning.MANUFACTURER_NAME_STANDARD_WORDS)
	df_cleanning = df_cleanning.withColumn("MANUFACTURER_NAME_CLEANNING_WORDS_SEG", df_cleanning.MANUFACTURER_NAME_CLEANNING_WORDS)
	df_cleanning = words_to_reverse_index(df_cleanning, df_encode, "MANUFACTURER_NAME_STANDARD_WORDS", "MANUFACTURER_NAME_STANDARD_WORDS")
	df_cleanning = words_to_reverse_index(df_cleanning, df_encode, "MANUFACTURER_NAME_CLEANNING_WORDS", "MANUFACTURER_NAME_CLEANNING_WORDS")
	df_cleanning.printSchema
	df_cleanning.show(2)
	df_cleanning = df_cleanning.withColumn("MANUFACTURER_NAME_STANDARD_WORDS", array_distinct(df_cleanning.MANUFACTURER_NAME_STANDARD_WORDS))
	df_cleanning = df_cleanning.withColumn("MANUFACTURER_NAME_CLEANNING_WORDS", array_distinct(df_cleanning.MANUFACTURER_NAME_CLEANNING_WORDS))
	df_cleanning.show(2)
	
	return df_cleanning

@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def mnf_index_word_cosine_similarity(o, v):
	frame = {
		"CLEANNING": o,
		"STANDARD": v
	}
	df = pd.DataFrame(frame)
	def array_to_vector(arr):
		idx = []
		values = []
		# try:
		if type(arr) != np.ndarray:
			s = [8999,]
		# except AttributeError:
		else:
			s = list(set(arr))
		s.sort()
		for item in s:
			if isnan(item):
				idx.append(8999)
				values.append(1)
				break
			else:
				idx.append(item)
				if item < 2000:
					values.append(2)
				elif (item >= 2000) & (item < 5000):
					values.append(10)
				else:
					values.append(1)
		return Vectors.sparse(9000, idx, values)
		#                    (向量长度，索引数组，与索引数组对应的数值数组)
	def cosine_distance(u, v):
		u = u.toArray()
		v = v.toArray()
		return float(numpy.dot(u, v) / (sqrt(numpy.dot(u, u)) * sqrt(numpy.dot(v, v))))
	df["CLENNING_FEATURE"] = df["CLEANNING"].apply(lambda x: array_to_vector(x))
	df["STANDARD_FEATURE"] = df["STANDARD"].apply(lambda x: array_to_vector(x))
	df["RESULT"] = df.apply(lambda x: cosine_distance(x["CLENNING_FEATURE"], x["STANDARD_FEATURE"]), axis=1)
	return df["RESULT"]


def mnf_encoding_cosine(df_cleanning):
	df_cleanning = df_cleanning.withColumn("COSINE_SIMILARITY", \
					mnf_index_word_cosine_similarity(df_cleanning.MANUFACTURER_NAME_CLEANNING_WORDS, df_cleanning.MANUFACTURER_NAME_STANDARD_WORDS))
	# df_cleanning.where((df_cleanning.label == 1.0) & (df_cleanning.EFFTIVENESS_MANUFACTURER < 0.9) & (df_cleanning.COSINE_SIMILARITY > df_cleanning.EFFTIVENESS_MANUFACTURER)) \
	# 	.select("MANUFACTURER_NAME", "MANUFACTURER_NAME_CLEANNING_WORDS", "MANUFACTURER_NAME_STANDARD", \
	# 			"MANUFACTURER_NAME_STANDARD_WORDS", "EFFTIVENESS_MANUFACTURER", "COSINE_SIMILARITY").show(100)
	return df_cleanning
	
	
def second_round_with_col_recalculate(df_second_round, dosage_mapping, df_encode, spark):
	df_second_round = df_second_round.join(dosage_mapping, df_second_round.DOSAGE == dosage_mapping.CPA_DOSAGE, how="left").na.fill("")
	df_second_round = df_second_round.withColumn("MASTER_DOSAGE", when(df_second_round.MASTER_DOSAGE.isNull(), df_second_round.JACCARD_DISTANCE). \
						otherwise(df_second_round.MASTER_DOSAGE))
	df_second_round = df_second_round.withColumn("EFFTIVENESS_DOSAGE_SE", dosage_replace(df_second_round.MASTER_DOSAGE, \
														df_second_round.DOSAGE_STANDARD, df_second_round.EFFTIVENESS_DOSAGE)) 
	df_second_round = mnf_encoding_index(df_second_round, df_encode, spark)
	df_second_round = mnf_encoding_cosine(df_second_round)
	df_second_round = df_second_round.withColumn("EFFTIVENESS_MANUFACTURER_SE", \
										when(df_second_round.COSINE_SIMILARITY >= df_second_round.EFFTIVENESS_MANUFACTURER, df_second_round.COSINE_SIMILARITY) \
										.otherwise(df_second_round.EFFTIVENESS_MANUFACTURER))
	df_second_round = mole_dosage_calculaltion(df_second_round)   # 加一列EFF_MOLE_DOSAGE，doubletype
	df_second_round = df_second_round.withColumn("EFFTIVENESS_PRODUCT_NAME_SE", \
								prod_name_replace(df_second_round.EFFTIVENESS_MOLE_NAME, df_second_round.EFFTIVENESS_MANUFACTURER_SE, \
												df_second_round.EFFTIVENESS_PRODUCT_NAME, df_second_round.MOLE_NAME, \
												df_second_round.PRODUCT_NAME_STANDARD, df_second_round.EFF_MOLE_DOSAGE))
												
	return df_second_round
	
@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def spec_eff_int_or_carry(SPEC_valid_digit_STANDARD, SPEC_valid_total_ORIGINAL, SPEC_valid_unit_STANDARD, SPEC_valid_unit, SPEC_valid_total_STANDARD, EFFTIVENESS_SPEC_SPLIT):

	frame = { "SPEC_valid_digit_STANDARD": SPEC_valid_digit_STANDARD, "SPEC_valid_total_ORIGINAL": SPEC_valid_total_ORIGINAL,
			  "SPEC_valid_unit_STANDARD": SPEC_valid_unit_STANDARD,  "SPEC_valid_unit": SPEC_valid_unit, 
			  "SPEC_valid_total_STANDARD": SPEC_valid_total_STANDARD, "EFFTIVENESS_SPEC_SPLIT": EFFTIVENESS_SPEC_SPLIT}
	df = pd.DataFrame(frame)
	
	
	# try:
	df["EFFTIVENESS_SPEC_SPLIT"] = df.apply(lambda x: 0.99 if ((int(float(x["SPEC_valid_total_ORIGINAL"])) == int(float(x["SPEC_valid_total_STANDARD"]))) \
														& (x["SPEC_valid_unit_STANDARD"] == x["SPEC_valid_unit"])) \
													else 0.999 if ((math.ceil(float(x["SPEC_valid_total_ORIGINAL"])) == math.ceil(float(x["SPEC_valid_total_STANDARD"]))) \
														& (x["SPEC_valid_unit_STANDARD"] == x["SPEC_valid_unit"])) \
											else x["EFFTIVENESS_SPEC_SPLIT"], axis=1)
	# except ValueError:
	# 	df["EFFTIVENESS_SPEC_SPLIT"] = df.apply(lambda x: 0.888, axis=1)

	return df["EFFTIVENESS_SPEC_SPLIT"]


def spec_split_matching(df):
	df.printSchema()
	
	df = df.withColumn("SPEC_valid_total_STANDARD",  spec_valid_std_transfer_pandas_udf(df.SPEC_valid_digit_STANDARD))
	# df = df.where(df.SPEC_STANDARD == "84.5UG  /DOS  60")
	
	df = df.withColumn("SPEC", regexp_replace("SPEC", r"(微克)", "UG"))
	df = df.withColumn("SPEC", regexp_replace("SPEC", r"[()]", ""))
	df = df.withColumn("SPEC", regexp_replace("SPEC", r"(（)", ""))
	df = df.withColumn("SPEC", regexp_replace("SPEC", r"(）)", ""))
	df = df.withColumn("SPEC", regexp_replace("SPEC", r"(Μ)", "M"))
	# df = df.drop("SPEC_valid_digit_STANDARD", "SPEC_valid_unit_STANDARD", "SPEC_gross_digit_STANDARD", "SPEC_gross_unit_STANDARD", "SPEC_STANDARD")

	
	df = df.withColumn("SPEC", regexp_replace("SPEC", r"(万)", "T"))
	df = df.withColumn("SPEC", regexp_replace("SPEC", r"(μ)", "U"))
	df = df.withColumn("SPEC", upper(df.SPEC))
	df = df.replace(" ", "")
	# df = df.where(df.SPEC_STANDARD == "62.5UG+25UG/DOS 30")
	# df = df.withColumn("SPEC_gross", regexp_extract('SPEC', spec_regex, 2))
	# 拆分规格的成分
	# df = df.withColumn("SPEC_percent", regexp_extract('SPEC', r'(\d*.*\d+%)', 1))
	# df = df.withColumn("SPEC_co", regexp_extract('SPEC', r'(CO)', 1))
	spec_valid_regex =  r'([0-9]\d*\.?\d*\s*[A-Za-z]*/?\s*[A-Za-z]+)'
	df = df.withColumn("SPEC_valid", regexp_extract('SPEC', spec_valid_regex, 1))
	spec_gross_regex =  r'([0-9]\d*\.?\d*\s*[A-Za-z]*/?\s*[A-Za-z]+)[ ,/:∶+\s]*[\u4e00-\u9fa5]*([0-9]\d*\.?\d*\s*[A-Za-z]*/?\s*[A-Za-z]+)'
	df = df.withColumn("SPEC_gross", regexp_extract('SPEC', spec_gross_regex, 2))
	
	spec_valid_se_regex =  r'([0-9]\d*\.?\d*\s*[:/+][0-9]\d*\.?\d*\s*[A-Za-z]+)'
	df = df.withColumn("SPEC_valid_2", regexp_extract('SPEC', spec_valid_se_regex, 1))
	df = df.withColumn("SPEC_valid", when((df.SPEC_valid_2 != ""), df.SPEC_valid_2).otherwise(df.SPEC_valid))
	# df = df.drop("SPEC_valid_2")
	
	# spec_third_regex = r'([0-9]\d*\.?\d*\s*[A-Za-z]*/?\s*[A-Za-z]+)[ /:∶+\s]([0-9]\d*\.?\d*\s*[A-Za-z]*/?\s*[A-Za-z]+)[ /:∶+\s]([0-9]\d*\.?\d*\s*[A-Za-z]*/?\s*[A-Za-z]+)'
	# df = df.withColumn("SPEC_third", regexp_extract('SPEC', spec_third_regex, 3))

	# pure_number_regex_spec = r'(\s\d+$)'
	# df = df.withColumn("SPEC_pure_number", regexp_extract('SPEC', pure_number_regex_spec, 1))
	# dos_regex_spec = r'(/DOS)'
	# df = df.withColumn("SPEC_dos", regexp_extract('SPEC', dos_regex_spec, 1))

	
	# df = df.withColumn("SPEC_valid", dos_pandas_udf(df.SPEC_valid, df.SPEC_pure_number, df.SPEC_dos))
	
	# 单位转换
	df = df.withColumn("SPEC_valid", transfer_unit_pandas_udf(df.SPEC_valid))
	df = df.withColumn("SPEC_gross", transfer_unit_pandas_udf(df.SPEC_gross))
	df = df.drop("SPEC_gross_digit", "SPEC_gross_unit", "SPEC_valid_digit", "SPEC_valid_unit")
	# df = df.withColumn("SPEC_percent", percent_pandas_udf(df.SPEC_percent, df.SPEC_valid, df.SPEC_gross))
	
	df = df.na.fill("")
	df.show()
	
	# 把百分号补充到有效成分列中
	# df = df.withColumn("SPEC_percent", lit(""))
	# df = df.withColumn("SPEC_gross", when(((df.SPEC_gross == "") & (df.SPEC_valid != "")), df.SPEC_valid).otherwise(df.SPEC_gross))
	# df = df.withColumn("SPEC_valid", when((df.SPEC_percent != ""), df.SPEC_percent).otherwise(df.SPEC_valid))
	# df = df.withColumn("SPEC_valid", when((df.SPEC_valid == df.SPEC_gross), lit("")).otherwise(df.SPEC_valid))
	
	# 拆分成四列
	# digit_regex_spec = r'(\d+\.?\d*e?-?\d*?)'
	
	# df = df.withColumn("SPEC_valid_digit", regexp_extract('SPEC_valid', digit_regex_spec, 1))
	# df = df.withColumn("SPEC_valid_unit", regexp_replace('SPEC_valid', digit_regex_spec, ""))
	
	# df = df.withColumn("SPEC_gross_digit", regexp_extract('SPEC_gross', digit_regex_spec, 1))
	# df = df.withColumn("SPEC_gross_unit", regexp_replace('SPEC_gross', digit_regex_spec, ""))
	# df = df.na.fill("")
	# df.show()
	
	unit_regex_spec = r'([A-Z]+\d*)'
	
	df = df.withColumn("SPEC_valid_unit", regexp_extract('SPEC_valid', unit_regex_spec, 1))
	df = df.withColumn("SPEC_valid_digit", regexp_replace('SPEC_valid', unit_regex_spec, ""))
	
	df = df.withColumn("SPEC_gross_unit", regexp_extract('SPEC_gross', unit_regex_spec, 1))
	df = df.withColumn("SPEC_gross_digit", regexp_replace('SPEC_gross', unit_regex_spec, ""))
	df = df.withColumn("SPEC_valid_total_ORIGINAL",  spec_valid_std_transfer_pandas_udf(df.SPEC_valid_digit))
	df = df.withColumn("SPEC_total_ORIGINAL",  spec_total_cleanning_pandas_udf(df.SPEC_valid_digit, df.SPEC_valid_unit, df.SPEC_gross_digit, df.SPEC_gross_unit))
	
	df = df.na.fill("")
	df.select("SPEC_valid_digit", "SPEC_valid_total_ORIGINAL", "SPEC_valid_digit_STANDARD", "SPEC_valid_total_STANDARD").show()
	# df.show()
	
	# 开始计算effectiveness的逻辑
	df = df.withColumn("EFFTIVENESS_SPEC_SPLIT", lit(0))
	# 1. 如果 【四列】分别都相等，则eff为1
	df = df.withColumn("EFFTIVENESS_SPEC_SPLIT", when( \
						((df.SPEC_valid_digit_STANDARD == df.SPEC_valid_digit) & (df.SPEC_valid_unit_STANDARD == df.SPEC_valid_unit) \
						& (df.SPEC_gross_digit_STANDARD == df.SPEC_gross_digit) & (df.SPEC_gross_unit_STANDARD == df.SPEC_gross_unit)), \
						lit(1))\
						.otherwise(df.EFFTIVENESS_SPEC_SPLIT))
	# df = df.where(df.EFFTIVENESS_SPEC_SPLIT == 0)
	
	# 2. 如果original/standard【只有valid/只有gross】，且仅有的部分是可以对应的，则eff为1
	df = df.withColumn("EFFTIVENESS_SPEC_SPLIT", when( \
						((df.SPEC_valid_digit == "") & (df.SPEC_valid_unit == "") \
						& ((df.SPEC_gross_digit_STANDARD == df.SPEC_gross_digit) & (df.SPEC_gross_unit_STANDARD == df.SPEC_gross_unit)) \
						| ((df.SPEC_valid_digit_STANDARD == df.SPEC_gross_digit) & (df.SPEC_valid_unit_STANDARD == df.SPEC_gross_unit))), \
						lit(1))\
						.otherwise(df.EFFTIVENESS_SPEC_SPLIT))
	df = df.withColumn("EFFTIVENESS_SPEC_SPLIT", when( \
						((df.SPEC_gross_digit == "") & (df.SPEC_gross_unit == "") \
						& ((df.SPEC_valid_digit_STANDARD == df.SPEC_valid_digit) & (df.SPEC_valid_unit_STANDARD == df.SPEC_valid_unit)) \
						| ((df.SPEC_gross_digit_STANDARD == df.SPEC_valid_digit) & (df.SPEC_gross_unit_STANDARD == df.SPEC_valid_unit))), \
						lit(1))\
						.otherwise(df.EFFTIVENESS_SPEC_SPLIT))
						
	# 3.如果【总量两列】 = 【有效成分两列】& 【有效成分两列】= 【总量两列】，则eff为1
	df = df.withColumn("EFFTIVENESS_SPEC_SPLIT", when( \
						((df.SPEC_valid_digit_STANDARD == df.SPEC_gross_digit) & (df.SPEC_valid_unit_STANDARD == df.SPEC_gross_unit) \
						& (df.SPEC_gross_digit_STANDARD == df.SPEC_valid_digit) & (df.SPEC_gross_unit_STANDARD == df.SPEC_valid_unit)), \
						lit(1))\
						.otherwise(df.EFFTIVENESS_SPEC_SPLIT))
						
	# 4. 如果【源数据valid == 标准数据valid之和】，则eff为1
	df = df.withColumn("EFFTIVENESS_SPEC_SPLIT", when( \
						((df.SPEC_valid_total_STANDARD == df.SPEC_valid_digit) & (df.SPEC_valid_unit_STANDARD == df.SPEC_valid_unit)), \
						lit(1))\
						.otherwise(df.EFFTIVENESS_SPEC_SPLIT))
	# df.withColumn("SPEC_valid_total_STANDARD", when(df.SPEC_valid_total_STANDARD == "nan", lit("")).otherwise(df.SPEC_valid_total_STANDARD))
	# df = df.where(df.SPEC == "CO 1.003 GM")
	# 5. 一些骚操作（目前是针对azsanofi的）：
	# 如果 【源数据有效成分 == 标准有效成分的取整值/四舍五入值】，则eff为0.99
	df = df.withColumn("EFFTIVENESS_SPEC_SPLIT", when(df.EFFTIVENESS_SPEC_SPLIT == 1, df.EFFTIVENESS_SPEC_SPLIT) \
												.otherwise(spec_eff_int_or_carry(df.SPEC_valid_digit_STANDARD, df.SPEC_valid_total_ORIGINAL, df.SPEC_valid_unit_STANDARD, \
																	df.SPEC_valid_unit, df.SPEC_valid_total_STANDARD, df.EFFTIVENESS_SPEC_SPLIT)))
	# df = df.withColumn("EFFTIVENESS_SPEC_SPLIT", when(df.SPEC.contains("162.5"), lit(0.999999)). \
	# 											otherwise(df.EFFTIVENESS_SPEC_SPLIT ))
	
	# 6. 如果【标准总量/标准有效成分 == 源数据总+有效】，则eff为1
	df = df.withColumn("EFFTIVENESS_SPEC_SPLIT", when( \
						(((df.SPEC_total_ORIGINAL == df.SPEC_gross_digit_STANDARD) & (df.SPEC_gross_unit == df.SPEC_gross_unit_STANDARD)) \
						| ((df.SPEC_total_ORIGINAL == df.SPEC_valid_digit_STANDARD) & (df.SPEC_valid_unit == df.SPEC_valid_unit_STANDARD))), \
						lit(1)) \
						.otherwise(df.EFFTIVENESS_SPEC_SPLIT))
	df.show()
	
	
	df.select("SPEC", "SPEC_STANDARD", "EFFTIVENESS_SPEC_SPLIT").show(25)
	
	return df
	