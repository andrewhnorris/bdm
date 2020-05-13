from pyspark import SparkContext

if __name__=='__main__':
    sc = SparkContext()
    sc.textFile('book.txt') \
        .flatMap(lambda x: x.split()) \
        .map(lambda x: (x,1)) \
        .reduceByKey(lambda x,y: x+y) \
        .saveAsTextFile('output')
