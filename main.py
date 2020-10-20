# -*- coding: utf-8 -*-
"""alfredyang@pharbers.com.

功能描述：job3：left join cpa和prod
  * @author yzy
  * @version 0.0
  * @since 2020/08/12
  * @note  落盘数据：cpa_prod_join

"""

import os
from pyspark.sql import SparkSession
from dataparepare import *
from interfere import *
from similarity import *
# from oldsimi import *
from pyspark.sql.types import *
from pyspark.sql.functions import desc
from pyspark.sql.functions import rank
from pyspark.sql import Window


def prepare():
	os.environ["PYSPARK_PYTHON"] = "python3"
	# 读取s3桶中的数据
	spark = SparkSession.builder \
		.master("yarn") \
		.appName("CPA&GYC match refactor") \
		.config("spark.driver.memory", "1g") \
		.config("spark.executor.cores", "2") \
		.config("spark.executor.instance", "4") \
		.config("spark.executor.memory", "2g") \
		.config('spark.sql.codegen.wholeStage', False) \
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


@udf(returnType=IntegerType())
def check_similarity(packid_check, packid_standard, similarity):
	# if v < 0.7:
	# 	return 0
	if (packid_check == "") & (packid_standard == ""):
		return 1
	elif len(packid_check) == 0:
		return 0
	elif len(packid_standard) == 0:
		return 0
	else:
		try:
			if int(packid_check) == int(packid_standard):
				return 1
			else:
				return 0
		except ValueError:
			return 0


if __name__ == '__main__':
	spark = prepare()
	df_standard = load_standard_prod(spark)
	df_cleanning = load_cleanning_prod(spark)
	# df_cleanning = df_cleanning.limit(100)
	df_interfere = load_interfere_mapping(spark)

	# 1. human interfere
	df_cleanning = human_interfere(spark, df_cleanning, df_interfere)
	df_cleanning.persist()

	# 2. 以MOLE_NAME为主键JOIN
	df_result = similarity(spark, df_cleanning, df_standard)
	df_result.persist()
	df_result.show()
	print(df_result.count())

	# 3. 对每个需要匹配的值做猜想排序
	windowSpec  = Window.partitionBy("id").orderBy(desc("SIMILARITY"))
	# windowSpec  = Window.partitionBy("id").orderBy("SIMILARITY")

	df_match = df_result.withColumn("RANK", rank().over(windowSpec))
	df_match = df_match.where(df_match.RANK <= 5)

	# df_match.show()
	df_match.persist()
	df_match.show()
	print(df_match.count())
	# df_match.printSchema()

	df_match = df_match.withColumn("check", check_similarity(df_match.PACK_ID_CHECK, df_match.PACK_ID_STANDARD, df_match.SIMILARITY))
	df_match.show(5)
	df_match = df_match.orderBy("id").drop("ORIGIN", "STANDARD")
	df_match.persist()
	df_match.repartition(1).write.mode("overwrite").csv("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/0.0.3/all")

	df_replace = df_match.filter(df_match.check == 1)

	df_no_replace = df_match.filter(df_match.check == 0)
	df_no_replace.repartition(1).write.mode("overwrite").csv("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/0.0.3/no_replace")

	print(df_replace.count())
	# df_replace = df_replace.where(df_replace.SIMILARITY > 0.7)
	print(df_no_replace.count())
	df_replace.repartition(1).write.format("parquet").mode("overwrite").save("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/0.0.3/replace")
	
	# df_match = df_match.orderBy("id").drop("ORIGIN", "STANDARD")
	# df_match.persist()
	# df_match.repartition(1).write.mode("overwrite").csv("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/0.0.3/all")
	# df_no_replace.repartition(1).write.mode("overwrite").csv("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/0.0.3/no_replace")

	# print(df_replace.count())
	# df_replace = df_replace.where(df_replace.SIMILARITY > 0.7)
	# print(df_replace.count())
	# df_replace.repartition(1).write.mode("overwrite").csv("s3a://ph-max-auto/2020-08-11/BPBatchDAG/refactor/replace")
