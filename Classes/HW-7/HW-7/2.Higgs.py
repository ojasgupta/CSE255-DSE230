# Name: Ojas Gupta  
# Email: ogupta@eng.ucsd.edu
# PID: A53201624
from pyspark import SparkContext
sc = SparkContext()

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint

from string import split,strip

from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.tree import RandomForest, RandomForestModel

from pyspark.mllib.util import MLUtils


# In[13]:

# Read the file into an RDD
# If doing this on a real cluster, you need the file to be available on all nodes, ideally in HDFS.
path='/HIGGS/HIGGS.csv'
#sampling 10% of dataset
inputRDD=sc.textFile(path).sample(False,0.1).cache()


# In[14]:

# Transform the text RDD into an RDD of LabeledPoints
Data=inputRDD.map(lambda line: [float(strip(x)) for x in line.split(',')]).map(lambda x: LabeledPoint(x[0], x[1:])).cache()


# In[15]:

(trainingData,testData)=Data.randomSplit([0.7,0.3])


# In[ ]:

from time import time
errors={}
start=time()
depth = 8
model=GradientBoostedTrees.trainClassifier(trainingData, {}, numIterations=35, maxDepth = depth)
#     print model.toDebugString()
errors[depth]={}
dataSets={'train':trainingData,'test':testData}
for name in dataSets.keys():  # Calculate errors on train and test sets
    data=dataSets[name]
    Predicted=model.predict(data.map(lambda x: x.features))
    LabelsAndPredictions=data.map(lambda x: x.label).zip(Predicted)
    Err = LabelsAndPredictions.filter(lambda (v,p):v != p).count()/float(data.count())
    errors[depth][name]=Err
print depth,errors[depth],int(time()-start),'seconds'
