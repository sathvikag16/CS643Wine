import quinn
import requests

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, Normalizer, StandardScaler

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

from pyspark.sql.session import SparkSession
from pyspark.context import SparkContext

scon = SparkContext('local')
spark = SparkSession(scon)

training_df = spark.read.format('csv').options(header='true', inferSchema='true', 
sep=';').load('TrainingDataset.csv')


test_df = spark.read.format('csv').options(header='true', inferSchema='true', 
sep=';').load('ValidationDataset.csv')

def remove_quotations(s):
 return s.replace('"', '')

testing_df = quinn.with_columns_renamed(remove_quotations)(testing_df)
testing_df = testing_df.withColumnRenamed('quality', 'label')

print("Data has been formatted.")
print(training_df.toPandas().head())

assemble = VectorAssembler(
 inputCols=["fixed acidity",
 "volatile acidity",
 "citric acid",
 "residual sugar",
 "chlorides",
 "free sulfur dioxide",
 "total sulfur dioxide",
 "density",
 "pH",
 "sulphates",
 "alcohol"],
 outputCol="inputFeatures")
scaler = Normalizer(inputCol="inputFeatures", outputCol="features")

pline1 = Pipeline(stages=[assembler, scaler, lr])
paramgrid = ParamGridBuilder().build()
evaluator = MulticlassClassificationEvaluator(metricName="f1")
crossval = CrossValidator(estimator=pline1, 
 estimatorParamMaps=paramgrid,
 evaluator=evaluator, 
 numFolds=3
 )
cvModel1 = crossval.fit(training_df) 
print("Docker run Successful")
print("F1 Score of this Model ", evaluator.evaluate(cvModel1.transform(testing_df)))
