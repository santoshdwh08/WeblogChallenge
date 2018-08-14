from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import split
from pyspark.sql.window import Window
from pyspark.sql import functions as F
import sys

# Spark Session Object
spark = SparkSession\
    .builder\
    .appName("PayTM")\
    .getOrCreate()


# Creating Schema
logSchema=StructType([StructField("request_timestamp", TimestampType(), True),
                      StructField("elb", StringType(), True),
                      StructField("client_port", StringType(), True),
                      StructField("backend_port", StringType(), True),
                      StructField("request_processing_time", StringType(), True),
                      StructField("backend_processing_time", StringType(), True),
                      StructField("response_processing_time", StringType(), True),
                      StructField("elb_status_code", IntegerType(), True),
                      StructField("backend_status_code", IntegerType(), True),
                      StructField("received_bytes", IntegerType(), True),
                      StructField("sent_bytes", IntegerType(), True),
                      StructField("request", StringType(), True),
                      StructField("user_agent", StringType(), True),
                      StructField("ssl_cipher", StringType(), True),
                      StructField("ssl_protocol", StringType(), True)])


# Reading the entire data set from the data file into a dataFrame:
logDF = spark\
    .read\
    .option("sep"," ")\
    .schema(logSchema)\
    .csv("hdfs:///root/pydev/dataset/2015_07_22_mktplace_shop_web_log_sample.log")



# Extract the IP value as "client_ip" from "client_port" column
# Convert the "request_timestamp" into Epoch time.
# Drop the original "client_port" and "request" columns from the dataFrame.

logFormattedDF = logDF\
    .withColumn("client_ip", split("client_port", ":").getItem(0))\
    .drop("client_port")\
    .withColumn("rTS",F.unix_timestamp("request_timestamp","yyyy-MM-dd HH:mm:ss.SS"))\
    .drop("request_timestamp")\
    .withColumn("URL", split("request", " ").getItem(1))\
    .drop("request")

# Create a temp table from the existing dataFrame
logTable = logFormattedDF.createOrReplaceTempView("logs")


# Sessionize the data into distinct sessions by IP(=users)
# using 15 min window as session threshold.

diffDF = spark.sql("SELECT    client_ip,                                          "
                   "          rTS,                                                "
                   "          URL,                                                "
                   "          new_session,                                        "
                   "          client_ip || '_' ||SUM(new_session) over(partition by client_ip order by rTS) SESSION_ID "
                   "FROM      (                                                   "
                   "           SELECT     client_ip,                              "
                   "                      rTS,                                    "
                   "                      URL,                                    "
                   "                      CASE WHEN rTS - lag(rTS) over(partition by client_ip   "
                   "                                     order by rTS) >=900      "
                   "                      THEN 1 else 0 END new_session           "
                   "           FROM       logs                                    "
                   "          )                                                   "
                   "ORDER BY   client_ip, rTS                                     "
                  )

# Persisting the data to improve performance, by eliminating re-processing of the same DataFrame
diffDF.persist()


# Solution for Question 1 & 3
# 1> count(URL) group by SESSION_ID
# 3> count(distinct URL) group by SESSION_ID
# Calculating - Total Hits and Unique URL visits by IP/Users

hitsCount = diffDF.select("URL", "session_id")\
    .groupBy("session_id")\
    .agg(F.count("URL"),F.countDistinct("URL"))

# Saving output of hitsCount dataframe to disk
hitsCount.coalesce(1).write.mode("overwrite").csv("hdfs:///root/pydev/dataset/hit_count", header='true')


# Solution for Question 2
# Generating Avg session time as "avg(SESSION_TIME) group by client_ip", where
# SESSION_TIME is calculated as difference between max and min of event time (rTS) grouped by client_ip, SESSION_ID


sessionTimeBySessionID = diffDF.select("rTS", "client_ip", "session_id")\
                               .groupBy("client_ip", "session_id")\
                               .agg(F.max("rTS") - F.min("rTS"))\
                               .withColumnRenamed("(max(rTS) - min(rTS))", "SESSION_TIME")

avgSessionTimeByUser = sessionTimeBySessionID.select("client_ip", "SESSION_TIME")\
                                             .groupBy("client_ip")\
                                             .agg(F.avg("SESSION_TIME"))

# Saving output of avgSessionTimeByUser dataframe to disk
avgSessionTimeByUser.coalesce(1).write.mode("overwrite").csv("hdfs:///root/pydev/dataset/AvgSession_Time", header='true')


# Question 4
# Most engaged users, i.e. the IP with the longest session time
# Fetching IP with max(SESSION_TIME) within entire WebLog dataset

windowSpec = Window\
    .rangeBetween(-sys.maxsize, sys.maxsize)

longestSession = sessionTimeBySessionID.select("client_ip",
                                               "SESSION_TIME",
                                               F.max("SESSION_TIME").over(windowSpec).alias('LONGEST_SESSION_TIME'))\
                                        .where("SESSION_TIME=LONGEST_SESSION_TIME")\
                                        .select("client_ip")

# Saving output of longestSession dataframe to disk
longestSession.coalesce(1).write.mode("overwrite").csv("hdfs:///root/pydev/dataset/Longest_User", header='true')