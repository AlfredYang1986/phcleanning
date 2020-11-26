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
from pyspark.sql.functions import rank, row_number
from pyspark.sql.functions import when
from pyspark.sql.functions import col, udf
import numpy
from pyspark.sql.types import *
from pyspark.sql.functions import array, array_contains
from pyspark.sql.functions import broadcast
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import explode
from pyspark.sql.functions import pandas_udf, PandasUDFType
from math import isnan
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
	df["SPEC_JWS"] = df.apply(lambda x: 1 if x["SPEC"].strip() ==  x["SPEC_STANDARD"].strip() \
										else 0 if ((x["SPEC"].strip() == "") or (x["SPEC_STANDARD"].strip() == "")) \
										else 1 if x["SPEC"].strip() in x["SPEC_STANDARD"].strip() \
										else 1 if x["SPEC_STANDARD"].strip() in x["SPEC"].strip() \
										else jaro_winkler_similarity(x["SPEC"].strip(), x["SPEC_STANDARD"].strip()), axis=1)
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
	df = df.drop("SPEC_gross_digit", "SPEC_gross_unit", "SPEC_valid_digit", "SPEC_valid_unit")
	df = df.withColumn("SPEC_percent", percent_pandas_udf(df.SPEC_percent, df.SPEC_valid, df.SPEC_gross))
	df = df.withColumn("SPEC_ept", lit("/"))
	df = df.withColumn("SPEC", concat( "SPEC_percent", "SPEC_ept", "SPEC_valid", "SPEC_ept", "SPEC_gross", "SPEC_ept", "SPEC_third")) \
					.drop("SPEC_ept", "SPEC_percent", "SPEC_co", "SPEC_valid", "SPEC_gross", "SPEC_pure_number", "SPEC_third")
	return df


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
	# df_lexicon = spark.read.parquet("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/zyyin/lexicon")
	# df_pd = df_lexicon.toPandas()  # type = pd.df
	# lexicon = df_pd["HIGH_SCORE_WORDS"].tolist()  # type = list
	lexicon = ['康福来', '一洋', '新海康', '正大丰海', '远大蜀阳', '同济现代', '康诺生化', '雪龙海姆普德', '量子高科', '爱的发', \
	'天天乐', '费森尤斯卡比', '国大生物', '生命科技', '先声中人', '蓝十字', '天和', '倍的福', '睿鹰先锋', '可可康', '和明', \
	'复宏汉霖', '北大维信', 'BIOTEST', '健朗', '一格', '山东健康', '四药', '海神联盛', '曼秀雷敦', '裕松源', '青春康源', '海容唐果', \
	'杏辉天力', '高 博京邦', '三诺生物', '大海阳光', '千金湘江', '巨都药业', '太湖美', '信东生技', '北医联合', '铜鼓仁和', '未名生物', \
	'安科生物', '赛特力,碧兰', '海洋渔业', '正同', '正大天晴', '红星药业', '达 因儿童', '诺亚荣康', '远大德天', '云南植物', '天药本草堂', \
	'省', '太龙', '绿因', '旭东海普', '京西双鹤', '赣南制药', '威仕生物', '云南白药', '九瑞健康', '帝斯曼', '中健康桥', '上药中西', \
	'康特能', '华润双鹤', '红星葡萄', '三菱化学', '都邦', '正和', '包头中药', '必康嘉隆', '华药南方', '快好', '柳韩', '药业', '江山', \
	'田边三菱', '巨能乐斯', '九正', '金山禅心', '华润高科', '东阳光', '日东电工', '中孚', '新春都', '原子科兴', '西藏藏药', '九泰', \
	'鲁抗大地', '通药制药', '中原瑞德', '澳美', '金蟾生化', '长澳', '未名新鹏', '大红鹰恒顺', '奈科明', '津新', '新宝源', '辰欣', \
	'千金协力', '华兰生物疫苗', '扬州生物化学', '黄河中药', '利君精华', '三药', '中国医科大学', '版纳药业', '新赣江', '羚锐生物', \
	'韩美', '华神生物', '辉南长龙', '锦帝九州', '新黄河', '乐康美的澜', '新东日', '信达生物', '控制疗法', '得恩德', '千汇', \
	'亿胜', '津华晖星', '天一秦昆', '福瑞达生物', '瑞年前进', '广西医科', '医科大学生物', '鲁抗', '三才', '第一生物', '生物制品', \
	'益生源', '药都制药', '九洲龙跃', '晨牌药业', '万嘉', '华信生物', '中泰', '东方广诚', '亚大', '兰州生物', '拜耳先灵', '希尔康泰', \
	'卫生材料', '江西生物', '遂成', '神经精神病', '龙灯瑞迪', '远力健', '达因', '厚生天佐', '长联来福', '威奇达', '凯茂生物', \
	'世贸天阶', '医药', '司邦得', '欧加农', '夏都', '华仁太医', '五景', '圣和', '美大康', '长生生物', '颐和', '安徽', '上海血液', \
	'ALL Medicus', '全新生物', '亚邦生缘', '先求', '同人泰', '第一三共', '金蟾', '回元堂', '华兰生物工程', '长征富民', '英科新创', \
	'康尔佳生物', '比切姆', '美时', '康都', '艾富西', '人人康', '赣药全新', '仁和', '百草', 'Laboratori Guidotti', '三精', \
	'PIRAMAL ENTERPRISES', '滨湖双鹤', '安生凤凰', '回音必', '生物化学', '皇城相府', '意大泛马克', '市', '健赞生物', '基因泰克', \
	'华润九新', '兴和', '和盈', '鲁北药业', '新化学', '鑫善源', '宝鉴堂', '通和', '开封制药', '第一药品', '卓谊生物', '一正', \
	'首和金海', '九旭', '中科生物', '康博士', '为民', '宣泰海门', '一新', '一康', '康泰生 物', '正大清江', 'BEN VENUE', '莎普爱思', \
	'法玛西亚普强', '鑫威格', '四环制药', '再鼎', '绿金子', '城市', '3M', '百会', '吉安三力', '老桐君', '佛都', '国光生物', \
	'味之素', '万正', '王牌速效', '在田', '赛林泰', '鼎恒升', '百年六福堂', '万通复升', '泰邦生物', '华新生物', '大连生物', \
	'张江生物', '新兴同仁', '百泰', '海王英特龙', '皇甫谧', '万邦', '尚善堂', '白云山制药总厂', '雷允 上', '金诃藏药', '康弘药业', \
	'叶开泰', '九发', '珐博进', '现代哈森', '得能', '王清任', '爱活', '同一堂', '会好', '盐野义', '致和', '慧宝源', '一品红', \
	'御金丹', '三爱', '楚天舒', '百慧', '利丰华瑞', '巨能', '基立福', '九泓', '金创', '华迈士', '莱士血液', '亿帆', '五加参', 'D.R', \
	'赛而', '亚东启天', '信合援生', '四环生物', '汇天生物', '赛诺维', '赛达生物', '绿十字', '誉隆亚东', '伊伯萨', '好医生', '必康制药', \
	'和泽', '民生滨江', '润和', '中西三维', '东泰', '依比威', '协和发酵', '三生国健', '马博士', '和记黄埔', '民生健康', '居仁堂爱民', \
	'第一生化', '新世通', '齐都', '何济公', '中宝曙光', '千红', '澳医', '药都仁和', '新张药', '大得利', '寿制药', '金虹胜利', '杨凌生物', \
	'百济神州', '三叶', '通用电气', '旭化成', '家和', '医创中药', '大药厂', '金牛原大', '联合治疗', '京新', '华盛生物', '帝斯曼,江山', \
	'升和', '北生研', '久和', '天泰', '中联四药', '三叶美好', '同济奔达', '新南山', '丹生生物', '东北六药', '国大药业', '华润金蟾', \
	'济民可信', '康美保宁', '和盛堂', '鲁北生物', '和治', '天普', '法玛西亚', '兰生血液', '金山生物', '先声生物', '敬修堂', '新南方', \
	'多瑞', '爱科来', '明仁福瑞达', '敬一堂', '长生基因', '百澳', '康和', '博森生物', '艾美卫信', '余良卿', '山德士', '西南药业', \
	'华润顺峰', '天地恒一', '双新', '益普生', '百姓堂', '浦津林州', '原子高科', '惠美佳', '晋新双鹤', '上药新亚', '康必得', '华宝通', \
	'北大高科华泰', '赛特多', '麦道甘美', '百特医疗', '川大华西', '老拨云堂', '和仁堂', '景康', '比奥罗历加', '康弘生物', '优你特', \
	'宁国国安邦宁', '百科达', '恒和维康', '华润天和', '晨牌邦德', '三维生物', '医创药业', '康缘桔都', '力思特', '赛诺菲', '复旦张江', \
	'利福法玛', '拜耳', '金美济', '德芮可', '先声东元', '东方协和', '生晃荣养', '昊海生物', '优时比', '医科大学制药', '百年汉克', '\
	施维雅', '福和', '?', '万邦生化', '二叶', '万特', '和创', '华润三九', '特一', 'Ever Neuro', '爱可泰隆', '武汉血液', '万高', \
	'济民可信山禾', '先强', '中美华东', '新姿源', '中医药大学', '海神同洲', 'Fabrik Kreussler', '日中天', '明和', '伊赛特', \
	'江中高邦', '柏赛罗', '复旦复华', '天山制药', '韩都', '三精加滨', '海欣', '哈三联', '诺维诺', '万泽', '哈药,总', '百正', \
	'华生元', '华瑞联合', '赛百诺', '民生药业', '益民堂', '中国药科大学', 'Feel Tech', '太安堂', '新南方,青蒿', '弘和', '杏林白马', \
	'利君方圆', '协和发酵麒麟', '元和']
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
				 "有限公司", "股份", "控股", "集团", "总公司", "公司", "有限", "有限责任", "大药厂", \
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


def mnf_encoding_index(df_cleanning, df_encode):
	# 增加两列MANUFACTURER_NAME_CLEANNING_WORDS MANUFACTURER_NAME_STANDARD_WORDS - array(string)
	df_cleanning = phcleanning_mnf_seg(df_cleanning, "MANUFACTURER_NAME_STANDARD", "MANUFACTURER_NAME_STANDARD_WORDS")
	df_cleanning = phcleanning_mnf_seg(df_cleanning, "MANUFACTURER_NAME", "MANUFACTURER_NAME_CLEANNING_WORDS")
	# df_cleanning.where((df_cleanning.label == 1.0) & (df_cleanning.EFFTIVENESS_MANUFACTURER < 0.9)) \
	# 	.select("MANUFACTURER_NAME", "MANUFACTURER_NAME_CLEANNING_WORDS", "MANUFACTURER_NAME_STANDARD", "MANUFACTURER_NAME_STANDARD_WORDS", "EFFTIVENESS_MANUFACTURER").show(10)
	df_cleanning = words_to_reverse_index(df_cleanning, df_encode, "MANUFACTURER_NAME_STANDARD_WORDS", "MANUFACTURER_NAME_STANDARD_WORDS")
	df_cleanning = words_to_reverse_index(df_cleanning, df_encode, "MANUFACTURER_NAME_CLEANNING_WORDS", "MANUFACTURER_NAME_CLEANNING_WORDS")
	# df_cleanning.where((df_cleanning.label == 1.0) & (df_cleanning.EFFTIVENESS_MANUFACTURER < 0.9)) \
	# 	.select("MANUFACTURER_NAME", "MANUFACTURER_NAME_CLEANNING_WORDS", "MANUFACTURER_NAME_STANDARD", "MANUFACTURER_NAME_STANDARD_WORDS", "EFFTIVENESS_MANUFACTURER").show(10)
	return df_cleanning

	# df_cleanning.repartition(10).write.mode("overwrite").parquet(words_index_path)


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
		s = list(set(arr))
		s.sort()
		for item in s:
			if isnan(item):
				idx.append(7999)
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
		return Vectors.sparse(8000, idx, values)
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
	
	
def second_round_with_col_recalculate(df_second_round, dosage_mapping, df_encode):
	df_second_round = df_second_round.join(dosage_mapping, df_second_round.DOSAGE == dosage_mapping.CPA_DOSAGE, how="left").na.fill("")
	df_second_round = df_second_round.withColumn("MASTER_DOSAGE", when(df_second_round.MASTER_DOSAGE.isNull(), df_second_round.JACCARD_DISTANCE). \
						otherwise(df_second_round.MASTER_DOSAGE))
	df_second_round = df_second_round.withColumn("EFFTIVENESS_DOSAGE_SE", dosage_replace(df_second_round.MASTER_DOSAGE, \
														df_second_round.DOSAGE_STANDARD, df_second_round.EFFTIVENESS_DOSAGE)) 
	df_second_round = df_second_round.withColumn("EFFTIVENESS_PACK_QTY_SE", pack_replace(df_second_round.EFFTIVENESS_PACK_QTY, df_second_round.SPEC_ORIGINAL, \
														df_second_round.PACK_QTY, df_second_round.PACK_QTY_STANDARD))
	df_second_round = mnf_encoding_index(df_second_round, df_encode)
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