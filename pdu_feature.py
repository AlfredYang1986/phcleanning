# -*- codin: utf-8 -*-
"""alfredyan@pharbers.com.

功能描述：
  * @author yzy
  * @version 0.0
  * @since 2020/08/12
  * @note  落盘数据：cpa_prod_join

"""

import os
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import *
from pyspark.sql.functions import regexp_replace, regexp_extract
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.functions import udf
from pyspark.sql.functions import upper
from pyspark.sql.functions import lit
from pyspark.sql.functions import concat
from pyspark.sql.functions import desc
from pyspark.sql.functions import rank
from pyspark.sql.functions import when
from pyspark.sql.functions import col
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import OneHotEncoderEstimator
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
def efftiveness_with_jaccard_distance(mo, ms, do, ds):
	frame = {
		"MOLE_NAME": mo, "MOLE_NAME_STANDARD": ms,
		"DOSAGE": do, "DOSAGE_STANDARD": ds
	}
	df = pd.DataFrame(frame)

	df["MOLE_JD"] = df.apply(lambda x: jd(set(x["MOLE_NAME"]), set(x["MOLE_NAME_STANDARD"])), axis=1)
	df["DOSAGE_JD"] = df.apply(lambda x: jd(set(x["DOSAGE"]), set(x["DOSAGE_STANDARD"])), axis=1)
	df["RESULT"] = df.apply(lambda x: [x["MOLE_JD"], x["DOSAGE_JD"]], axis=1)
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
def efftiveness_with_jaro_winkler_similarity(mo, ms, po, ps, do, ds, so, ss, qo, qs, mf, mfc, mfe):
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
		"MANUFACTURER_NAME": mf, "MANUFACTURER_NAME_STANDARD": mfc, "MANUFACTURER_NAME_EN_STANDARD": mfe
	}
	df = pd.DataFrame(frame)

	df["MOLE_JWS"] = df.apply(lambda x: jaro_winkler_similarity(x["MOLE_NAME"], x["MOLE_NAME_STANDARD"]), axis=1)
	df["PRODUCT_JWS"] = df.apply(lambda x: 1 if x["PRODUCT_NAME"] in x ["PRODUCT_NAME_STANDARD"] \
										else 1 if x["PRODUCT_NAME_STANDARD"] in x ["PRODUCT_NAME"] \
										else jaro_winkler_similarity(x["PRODUCT_NAME"], x["PRODUCT_NAME_STANDARD"]), axis=1)
	df["DOSAGE_JWS"] = df.apply(lambda x: 1 if x["DOSAGE"] in x ["DOSAGE_STANDARD"] \
										else 1 if x["DOSAGE_STANDARD"] in x ["DOSAGE"] \
										else jaro_winkler_similarity(x["DOSAGE"], x["DOSAGE_STANDARD"]), axis=1)
	df["SPEC_JWS"] = df.apply(lambda x: 1 if x["SPEC"].strip() ==  x["SPEC_STANDARD"].strip() \
										else 0 if ((x["SPEC"].strip() == "") or (x["SPEC_STANDARD"].strip() == "")) \
										else 1 if x["SPEC"].strip() in x["SPEC_STANDARD"].strip() \
										else 1 if x["SPEC_STANDARD"].strip() in x["SPEC"].strip() \
										else jaro_winkler_similarity(x["SPEC"].strip(), x["SPEC_STANDARD"].strip()), axis=1)
	df["PACK_QTY_JWS"] = df.apply(lambda x: 1 if x["PACK_QTY"].replace(".0", "") == x["PACK_QTY_STANDARD"].replace(".0", "") \
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
def transfer_unit_pandas_udf(value):
	def unit_transform(spec_str):
		spec_str = spec_str.replace(" ", "")
		# 拆分数字和单位
		digit_regex = '\d+\.?\d*e?-?\d*?'
		# digit_regex = '0.\d*'
		try:
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
					}
				try:
					unit = unit_switch[unit]
				except KeyError:
					pass

			else:
				unit = ""
				value = ""

			return str(value) + unit

		except Exception:
			return spec_str

	frame = { "SPEC": value }
	df = pd.DataFrame(frame)
	df["RESULT"] = df["SPEC"].apply(unit_transform)
	return df["RESULT"]


@pandas_udf(StringType(), PandasUDFType.SCALAR)
def percent_pandas_udf(percent, valid, gross):
	def percent_calculation(percent, valid, gross):
		digit_regex = '\d+\.?\d*e?-?\d*?'
		if percent != "" and valid != "" and gross == "":
			num = int(percent.strip("%"))
			value = re.findall(digit_regex, valid)[0]
			unit = valid.strip(value)  # type = str
			final_num = num*float(value)*0.01
			result = str(final_num) + unit

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
	# 拆分规格的成分
	df = df.withColumn("SPEC_percent", regexp_extract('SPEC', r'(\d+%)', 1))
	df = df.withColumn("SPEC_co", regexp_extract('SPEC', r'(CO)', 1))
	spec_valid_regex =  r'([0-9]\d*\.?\d*\s*[A-Za-z]*/?\s*[A-Za-z]+)'
	df = df.withColumn("SPEC_valid", regexp_extract('SPEC', spec_valid_regex, 1))
	spec_gross_regex =  r'([0-9]\d*\.?\d*\s*[A-Za-z]*/?\s*[A-Za-z]+)[ /:∶+\s]([0-9]\d*\.?\d*\s*[A-Za-z]*/?\s*[A-Za-z]+)'
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
	df.show()
	df = df.drop("SPEC_gross_digit", "SPEC_gross_unit", "SPEC_valid_digit", "SPEC_valid_unit")
	df = df.withColumn("SPEC_percent", percent_pandas_udf(df.SPEC_percent, df.SPEC_valid, df.SPEC_gross))
	df = df.withColumn("SPEC_ept", lit(" "))
	df = df.withColumn("SPEC", concat( "SPEC_percent", "SPEC_valid", "SPEC_gross", "SPEC_third")) \
					.drop("SPEC_percent", "SPEC_co", "SPEC_valid", "SPEC_gross", "SPEC_ept", "SPEC_pure_number", "SPEC_third")
	return df


def similarity(df):
	df = df.withColumn("SIMILARITY", \
					when(df.PRODUCT_NAME.contains(df.MOLE_NAME) | df.MOLE_NAME.contains(df.PRODUCT_NAME), \
						1.2*df.EFFTIVENESS_MOLE_NAME + 1.2*df.EFFTIVENESS_DOSAGE \
						+ 1.2*df.EFFTIVENESS_SPEC + 1.2*df.EFFTIVENESS_PACK_QTY + 1.2*df.EFFTIVENESS_MANUFACTURER) \
					.otherwise(df.EFFTIVENESS_MOLE_NAME + df.EFFTIVENESS_PRODUCT_NAME + df.EFFTIVENESS_DOSAGE \
						+ df.EFFTIVENESS_SPEC + df.EFFTIVENESS_PACK_QTY + df.EFFTIVENESS_MANUFACTURER))

	windowSpec = Window.partitionBy("id").orderBy(desc("SIMILARITY"), desc("EFFTIVENESS_MOLE_NAME"), desc("EFFTIVENESS_DOSAGE"), desc("PACK_ID_STANDARD"))

	df = df.withColumn("RANK", rank().over(windowSpec))
	df = df.where((df.RANK <= 5) | (df.label == 1.0))

	return df


def hit_place_prediction(df, pos):
	return df.withColumn("prediction_" + str(pos), when((df.RANK == pos), 1.0).otherwise(0.0))
	# return df.withColumn("prediction_" + str(pos), when((df.RANK == pos) & (df.label == 1.0), 1.0).otherwise(0.0))


@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def dosage_replace(dosage_lst, dosage_standard, eff):

	frame = { "MASTER_DOSAGE": dosage_lst, "DOSAGE_STANDARD": dosage_standard, "EFFTIVENESS_DOSAGE": eff }
	df = pd.DataFrame(frame)

	df["EFFTIVENESS"] = df.apply(lambda x: 1.0 if x["DOSAGE_STANDARD"] in x["MASTER_DOSAGE"] \
											else x["EFFTIVENESS_DOSAGE"], axis=1)

	return df["EFFTIVENESS"]


@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def prod_name_replace(mole_name, mole_name_standard, mnf_name, mnf_name_standard, mnf_en_standard):

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


	frame = { "MOLE_NAME": mole_name, "MOLE_NAME_STANDARD": mole_name_standard,
			  "MANUFACTURER_NAME": mnf_name, "MANUFACTURER_NAME_STANDARD": mnf_name_standard, "MANUFACTURER_NAME_EN_STANDARD": mnf_en_standard }
	df = pd.DataFrame(frame)

	df["EFFTIVENESS_PROD"] = df.apply(lambda x: max((jaro_winkler_similarity((x["MOLE_NAME"] + x["MANUFACTURER_NAME"]), \
																		(x["MOLE_NAME_STANDARD"] + x["MANUFACTURER_NAME_STANDARD"]))), \
												(jaro_winkler_similarity((x["MOLE_NAME"] + x["MANUFACTURER_NAME"]), \
																		(x["MOLE_NAME_STANDARD"] + x["MANUFACTURER_NAME_EN_STANDARD"])))), axis=1)

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
	lexicon = ["优时比", "省", "市", "第一三共", "诺维诺", "药业", "医药", "在田", "人人康", \
				"健朗", "鑫威格", "景康", "皇甫谧", "安徽", "江中高邦", "鲁抗", "辰欣", "法玛西亚普强", "正大天晴", "拜耳", "三才"]
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
	df_standard = df_standard.withColumn("MANUFACTURER_NAME_WORDS", manifacture_name_pseg_cut(col(inputCol)))

	# 4. 分词之后构建词库编码
	# 4.1 stop word remover 去掉不需要的词
	stopWords = ["省", "市", "股份", "有限", "总公司", "公司", "集团", "制药", "总厂", "厂", "药业", "责任", "医药", "(", ")", "（", "）", \
				 "有限公司", "股份", "控股", "集团", "总公司", "公司", "有限", "有限责任", \
			     "药业", "医药", "制药", "制药厂", "控股集团", "医药集团", "控股集团", "集团股份", "药厂", "分公司", "-", ".", "-", "·", ":"]
	remover = StopWordsRemover(stopWords=stopWords, inputCol="MANUFACTURER_NAME_WORDS", outputCol=outputCol)

	return remover.transform(df_standard).drop("MANUFACTURER_NAME_WORDS")
