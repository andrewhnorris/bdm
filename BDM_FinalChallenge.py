from pyspark import SparkContext
from pyspark.sql.session import SparkSession
import pyspark.sql.functions as f 
from pyspark.sql.types import IntegerType
from itertools import chain
from pyspark.sql.functions import broadcast
import statsmodels.api as sm
import sys

def violations_per_streetline(output_folder):
	spark = SparkSession.builder.getOrCreate()
	# create a pyspark df from Parking Violations
	violations = spark.read.csv('hdfs:///tmp/bdm/nyc_parking_violation/', header=True, inferSchema=True).cache()
	# simplify dataframe, drop null vals
	violations = violations.select(f.to_date(violations['Issue Date'], 'MM-dd-yyyy').alias('Date'), f.lower(violations['Violation County']).alias('County'), violations['House Number'], f.lower(violations['Street Name']).alias('Street Name')).na.drop()
	# extract year
	violations = violations.withColumn('Year', f.year(violations['Date']))
	# filter years 2015-2019
	violations = violations.where(f.col("Year").isin({2015,2016,2017,2018,2019}))
	# clean house numbers
	violations = violations.withColumn('House Number', f.regexp_replace('House Number', '-', '').cast(IntegerType()))
	# map county vals to borocode
	mn = ['man','mh','mn','newy','new','y','ny']
	bk = ['bk','k','king','kings']
	bx = ['bronx','bx']
	qu = ['q','qn','qns','qu','queen']
	si = ['r','richmond']

	counties_list = [mn, bx, bk, qu, si]
	county_dic = {}
	for i, co in enumerate(counties_list):
		for name in co:
			county_dic[name] = i+1

	mapping_expr = f.create_map([f.lit(x) for x in chain(*county_dic.items())])
		
	violations = violations.withColumn("BOROCODE", mapping_expr.getItem(f.col("County")))
	# drop unneeded cols
	columns_to_drop = ['Date', 'County']
	violations = violations.drop(*columns_to_drop)
	# add even/odd col
	violations = violations.withColumn('odd', f.when(f.col('House Number')%2==0, 0).otherwise(1))
	
	# create a pyspark df from Street Centerlines
	centerlines = spark.read.csv('hdfs:///tmp/bdm/nyc_cscl.csv', header=True, escape='"', inferSchema=True).cache()
	# select cols
	centerlines = centerlines.select(centerlines['PHYSICALID'], \
		centerlines['L_LOW_HN'], centerlines['L_HIGH_HN'], centerlines['R_LOW_HN'],\
		centerlines['R_HIGH_HN'], f.lower(centerlines['ST_LABEL']).alias('ST_LABEL'),\
		f.lower(centerlines['FULL_STREE']).alias('FULL_STREE'),\
		centerlines['BOROCODE'])
	# clean house numbers, cast as integers
	centerlines = centerlines.withColumn('L_LOW_HN', f.regexp_replace('L_LOW_HN', '-', '').cast(IntegerType()))
	centerlines = centerlines.withColumn('L_HIGH_HN', f.regexp_replace('L_HIGH_HN', '-', '').cast(IntegerType()))
	centerlines = centerlines.withColumn('R_LOW_HN', f.regexp_replace('R_LOW_HN', '-', '').cast(IntegerType()))
	centerlines = centerlines.withColumn('R_HIGH_HN', f.regexp_replace('R_HIGH_HN', '-', '').cast(IntegerType()))

	# join Violations and Centerline data frames on conditions
	violations_joined = violations.join(f.broadcast(centerlines),
		(violations['BOROCODE'] == centerlines['BOROCODE']) & 
		((violations['Street Name'] == centerlines['ST_LABEL']) | (violations['Street Name'] == centerlines['FULL_STREE'])) &
		( ((violations['odd'] == 0) & 
			(centerlines['R_LOW_HN'] <= violations['House Number']) & 
			(violations['House Number'] <= centerlines['R_HIGH_HN']))
			|
		((violations['odd'] == 1) & 
			(centerlines['L_LOW_HN'] <= violations['House Number']) & 
			(violations['House Number'] <= centerlines['L_HIGH_HN']))
		) )
	violations_joined.show()
	# group on PHYSICALID, pivot on Year
	violations_joined = violations_joined.groupBy("PHYSICALID").pivot("YEAR", ['2015','2016','2017','2018','2019']).count()
	# fill na's with 0
	violations_joined = violations_joined.na.fill(0)
	# rename pivoted columns for output
	violations_joined = violations_joined.withColumnRenamed('2015', 'COUNT_2015')\
		.withColumnRenamed('2016', 'COUNT_2016')\
		.withColumnRenamed('2017', 'COUNT_2017')\
		.withColumnRenamed('2018', 'COUNT_2018')\
		.withColumnRenamed('2019', 'COUNT_2019')
	# join remaining centerlines (without violations)
	full_violations_joined = violations_joined.join(broadcast(centerlines), ['PHYSICALID'], how='right')
	# drop unneeded cols
	columns_to_keep = ['PHYSICALID','COUNT_2015','COUNT_2016','COUNT_2017','COUNT_2018','COUNT_2019']
	full_violations_joined = full_violations_joined.select(*columns_to_keep)
	# fill na's with 0
	full_violations_joined = full_violations_joined.na.fill(0)
	# function to calculate OLS R-squared
	def ols_coef(a,b,c,d,e):
		x = ([2015,2016,2017,2018,2019])
		x = sm.add_constant(x)
		y = ([a,b,c,d,e])
		model = sm.OLS(y,x)
		results = model.fit()
		return results.params[1]
	# add col with OLS coeff for each street segment
	full_violations_joined = full_violations_joined.withColumn('OLS_COEF', ols_coef(full_violations_joined['COUNT_2015'], full_violations_joined['COUNT_2016'], full_violations_joined['COUNT_2017'], full_violations_joined['COUNT_2018'], full_violations_joined['COUNT_2019']))
	# order by PHYSICALID
	full_violations_joined = full_violations_joined.orderBy('PHYSICALID')

	# remove dups:
	# full_violations_joined = full_violations_joined.distinct()
	# write to csv
	full_violations_joined.write.csv(output_folder)

if __name__ == '__main__':
	output_folder = sys.argv[1]
	violations_per_streetline(output_folder) 






