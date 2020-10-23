# -*- codin: utf-8 -*-
"""alfredyan@pharbers.com.

功能描述：
  * @author yzy
  * @version 0.0
  * @since 2020/08/12
  * @note  落盘数据：cpa_prod_join

"""

import os
from pyspark.sql.functions import udf
from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import create_map
from pyspark.sql.types import *
from pyspark.sql.functions import to_json
from pyspark.sql.functions import regexp_replace
from pyspark.sql.functions import array
from pyspark.sql.functions import pandas_udf, PandasUDFType
import re
import numpy as np
import pandas as pd
import jieba
import jieba.posseg as pseg
import jieba.analyse as analyse
from pyspark.sql.functions import broadcast
from nltk.metrics import edit_distance as ed


def dosage_standify(df):
	df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"（注射剂）", ""))
	df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"（粉剂针）", ""))
	df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"（胶丸、滴丸）", ""))

	df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"SOLN", "注射"))
	df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"POWD", "粉针"))
	df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"SUSP", "混悬"))
	df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"OINT", "膏剂"))
	df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"NA", "鼻"))
	df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"SYRP", "口服"))
	df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"PATC", "贴膏"))
	df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"EMUL", "乳"))
	df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"AERO", "气雾"))
	df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"RAN", "颗粒"))
	df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"SUPP", "栓"))
	df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"PILL", "丸"))
	df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"MISC", "混合"))
	df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"LIQD", "溶液"))
	df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"TAB", "片"))
	df = df.withColumn("DOSAGE", regexp_replace("DOSAGE", r"CAP", "胶囊"))
	return df


@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def edit_distance_pandas_udf(a, b):
	frame = { "left": a, "right": b }
	df = pd.DataFrame(frame)
	df["RESULT"] = df.apply(lambda x: ed(x["left"], x["right"]), axis=1)
	return df["RESULT"]


@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def edit_distance_with_contains_pandas_udf(a, b):
	frame = { "left": a, "right": b }
	df = pd.DataFrame(frame)
	df["RESULT"] = df.apply(lambda x: 0 if x["left"] in x ["right"] else 0 if x["right"] in x ["left"] else ed(x["left"], x["right"]), axis=1)
	return df["RESULT"]


@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def edit_distance_with_float_change_pandas_udf(a, b):
	frame = { "left": a, "right": b }
	df = pd.DataFrame(frame)
	df["RESULT"] = df.apply(lambda x: ed(x["left"], x["right"].replace(".0", "")), axis=1)
	return df["RESULT"]


@udf(DoubleType())
def similarity_2_udf(mn, pd, dg, sp, pq, mf):
	return dg + 10*sp+ 60*pq + mf + pd + mn


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
	# df["MANUFACTURER_NAME_EFFECTIVENESS"] = df.apply(lambda x: df["MANUFACTURER_NAME_CH_ED"] if df["MANUFACTURER_NAME_CH_ED"] - df["MANUFACTURER_NAME_EN_ED"] < 0 \
	# 											else df["MANUFACTURER_NAME_EN_ED"], axis=1)
	# df["MANUFACTURER_NAME_EFFECTIVENESS"] = df.apply(lambda x: df["MANUFACTURER_NAME_CH_ED"], axis=1)

	df["RESULT"] = df.apply(lambda x: [x["MOLE_ED"], \
										x["PRODUCT_ED"], \
										x["DOSAGE_ED"], \
										x["SPEC_ED"], \
										x["PACK_QTY_ED"], \
										# x["MANUFACTURER_NAME_EFFECTIVENESS"], \
										], axis=1)
	return df["RESULT"]
