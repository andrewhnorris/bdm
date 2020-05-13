
from pyspark import SparkContext
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]

if __name__=='__main__':
	sc = SparkContext()

	complaints = sc.textFile(input_path).cache() 

	def extractInfo(partId, records):
		if partId==0:
			next(records)
		import csv
		import dateutil
		from dateutil import parser
		reader = csv.reader(records)
		for row in reader:
			try:
				(year, prod, company) = (dateutil.parser.parse(row[0]).year, row[1].lower(), row[7].lower())
				yield ((prod, year, company), 1)
			except Exception:
				print(row)

	def to_csv_line(record):
	    if "," in record[0]:
	        return '"'+str(record[0])+'"'+','+str(record[1])+','+str(record[2])+','+str(record[3])+','+str(record[4])
	    else:
	        return str(record[0])+','+str(record[1])+','+str(record[2])+','+str(record[3])+','+str(record[4])

	comp = complaints.mapPartitionsWithIndex(extractInfo) \
	    .reduceByKey(lambda x,y: x+y) \
	    .map(lambda x: ((x[0][0],x[0][1]), [x[1]])) \
	    .reduceByKey(lambda x,y: x+y) \
	    .map(lambda x: ((x[0]), (sum(x[1]), len(x[1]), round((max(x[1])*100/sum(x[1])))))) \
	    .sortByKey() \
	    .map(lambda x: (x[0][0], x[0][1], x[1][0], x[1][1], x[1][2])) \
	    .map(to_csv_line) 

	comp.collect()
	comp.saveAsTextFile(output_path)