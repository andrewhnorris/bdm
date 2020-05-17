from pyspark import SparkContext
from pyspark.sql.session import SparkSession
import pyspark.sql.functions as f 
from pyspark.sql.types import IntegerType
from itertools import chain
from pyspark.sql.functions import broadcast
import statsmodels.api as sm
import sys
import datetime as dt

def violations_per_streetline(output_folder):
	spark = SparkSession.builder.getOrCreate()
	# create a pyspark df from Parking Violations
	violations = spark.read.csv('hdfs:///tmp/bdm/nyc_parking_violation/', header=True, inferSchema=True).cache()
	# simplify dataframe, drop null vals
	violations = violations.select(violations['Issue Date'].alias('Date'), f.lower(violations['Violation County']).alias('County'), violations['House Number'], f.lower(violations['Street Name']).alias('Street Name')).na.drop()
	# extract year
	violations = violations.withColumn('Year', violations['Date'].substr(-4,4))
	# filter years 2015-2019
	# violations = violations.where(f.col("Year").isin({2015,2016,2017,2018,2019}))
	# clean house numbers
	violations = violations.withColumn('House Number', f.regexp_replace('House Number', '-', '').cast(IntegerType()))
	violations = violations.withColumn('House Number', f.regexp_replace(f.col('House Number'), '[ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz]', ''))
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
    # Find Total Complaints by Product and Year
	violations = violations.groupBy('BOROCODE','Street Name','House Number').pivot('Year', ['2015','2016','2017','2018','2019']).count()
    # fill na's with 0
	violations = violations.na.fill(0)
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
    # Select cols:  PHYSICALID, L_LOW_HN, L_HIGH_HN, R_LOW_HN, R_HIGH_HN, ST_LABEL, FULL_STREE, BOROCODE
	centerlines_street = centerlines.select(centerlines['PHYSICALID'], \
                                 centerlines['L_LOW_HN'], centerlines['L_HIGH_HN'], centerlines['R_LOW_HN'],\
                                 centerlines['R_HIGH_HN'], centerlines['BOROCODE'], f.lower(centerlines['ST_LABEL']).alias('ST_LABEL'))

    # Select cols:  PHYSICALID, L_LOW_HN, L_HIGH_HN, R_LOW_HN, R_HIGH_HN, ST_LABEL, FULL_STREE, BOROCODE
	centerlines_full = centerlines.select(centerlines['PHYSICALID'], \
                                 centerlines['L_LOW_HN'], centerlines['L_HIGH_HN'], centerlines['R_LOW_HN'],\
                                 centerlines['R_HIGH_HN'], centerlines['BOROCODE'], f.lower(centerlines['FULL_STREE']).alias('FULL_STREE'))

    # union centerlines, so all labels are in one col 
	centerlines = centerlines_street.union(centerlines_full)
    
	# uncache data
	violations = violations.unpersist()
	centerlines = centerlines.unpersist()

	# join Violations and Centerline data frames on conditions
	violations_joined = violations.join(f.broadcast(centerlines),
		(violations['BOROCODE'] == centerlines['BOROCODE']) & 
		(violations['Street Name'] == centerlines['ST_LABEL']) &
		( ((violations['odd'] == 0) & 
			(centerlines['R_LOW_HN'] <= violations['House Number']) & 
			(violations['House Number'] <= centerlines['R_HIGH_HN']))
			|
		((violations['odd'] == 1) & 
			(centerlines['L_LOW_HN'] <= violations['House Number']) & 
			(violations['House Number'] <= centerlines['L_HIGH_HN']))
		) )

	# drop unneeded cols
	columns_to_keep = ['PHYSICALID','2015','2016','2017','2018','2019']
	violations_joined = violations_joined.select(*columns_to_keep)
    # drop duplicates caused by union
	violations_joined = violations_joined.dropDuplicates(['PHYSICALID', '2015','2016','2017','2018','2019'])
    # group, sum on physical ID
	violations_joined = violations_joined.groupBy('PHYSICALID').agg({'2015':'sum','2016':'sum','2017':'sum','2018':'sum','2019':'sum'})
	# fill na's with 0
	violations_joined = violations_joined.na.fill(0)
	# function to calculate OLS R-squared
	def ols_coef(a,b,c,d,e):
		x = ([2015,2016,2017,2018,2019])
		x = sm.add_constant(x)
		y = ([a,b,c,d,e])
		model = sm.OLS(y,x)
		results = model.fit()
		return results.params[1]
	# add col with OLS coeff for each street segment
	violations_joined = violations_joined.withColumn('OLS_COEF', ols_coef(violations_joined['sum(2015)'], violations_joined['sum(2016)'], violations_joined['sum(2017)'], violations_joined['sum(2018)'], violations_joined['sum(2019)']))
	# order by PHYSICALID
	violations_joined = violations_joined.orderBy('PHYSICALID')
	# write to csv
	violations_joined.write.csv(output_folder)

if __name__ == '__main__':
	output_folder = sys.argv[1]
	starttime = dt.datetime.now()
	elapsed = dt.datetime.now() - starttime
	print("Done, Elapsed: {} (secs)".format(elapsed.total_seconds()))
	violations_per_streetline(output_folder) 






