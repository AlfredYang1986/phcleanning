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
