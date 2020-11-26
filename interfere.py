# -*- coding: utf-8 -*-
"""alfredyang@pharbers.com.

功能描述：
  * @author yzy
  * @version 0.0
  * @since 2020/08/12
  * @note  落盘数据：cpa_prod_join
  
"""

from pyspark.sql.functions import concat
from pyspark.sql.functions import udf
from pyspark.sql.types import *


@udf(returnType=StringType())
def interfere_replace_udf(origin, interfere):
	if interfere != "unknown":
		origin = interfere
	return origin


def human_interfere(spark, df_cleanning, df_interfere):
     # 1. 人工干预优先，不太对后期改
     # 干预流程将数据直接替换，在走平常流程，不直接过滤，保证流程的统一性
     df_cleanning = df_cleanning.withColumn("min", concat(df_cleanning["MOLE_NAME"], df_cleanning["PRODUCT_NAME"], df_cleanning["SPEC"], \
     									df_cleanning["DOSAGE"], df_cleanning["PACK_QTY"], df_cleanning["MANUFACTURER_NAME"]))
     
     # 2. join 干预表，替换原有的原始数据列
     df_cleanning = df_cleanning.join(df_interfere, on="min",  how="left") \
     				.na.fill({
     						"MOLE_NAME_INTERFERE": "unknown", 
     						"PRODUCT_NAME_INTERFERE": "unknown",
     						"SPEC_INTERFERE": "unknown",
                        	          "DOSAGE_INTERFERE": "unknown",
                        	          "PACK_QTY_INTERFERE": "unknown",
                        	          "MANUFACTURER_NAME_INTERFERE": "unknown"})

     df_cleanning = df_cleanning.withColumn("MOLE_NAME", interfere_replace_udf(df_cleanning.MOLE_NAME, df_cleanning.MOLE_NAME_INTERFERE)) \
     				.withColumn("PRODUCT_NAME", interfere_replace_udf(df_cleanning.PRODUCT_NAME, df_cleanning.PRODUCT_NAME_INTERFERE)) \
     				.withColumn("SPEC", interfere_replace_udf(df_cleanning.SPEC, df_cleanning.SPEC_INTERFERE)) \
     				.withColumn("DOSAGE", interfere_replace_udf(df_cleanning.DOSAGE, df_cleanning.DOSAGE_INTERFERE)) \
     				.withColumn("PACK_QTY", interfere_replace_udf(df_cleanning.PACK_QTY, df_cleanning.PACK_QTY_INTERFERE)) \
     				.withColumn("MANUFACTURER_NAME", interfere_replace_udf(df_cleanning.MANUFACTURER_NAME, df_cleanning.MANUFACTURER_NAME_INTERFERE))
     				
     df_cleanning = df_cleanning.select("id", "PACK_ID_CHECK", "MOLE_NAME", "PRODUCT_NAME", "DOSAGE", "SPEC", "PACK_QTY", "MANUFACTURER_NAME", \
                                        "MOLE_NAME_ORIGINAL", "PRODUCT_NAME_ORIGINAL", "DOSAGE_ORIGINAL", "SPEC_ORIGINAL", "PACK_QTY_ORIGINAL", "MANUFACTURER_NAME_ORIGINAL")
     # df_cleanning.persist()
     
     return df_cleanning