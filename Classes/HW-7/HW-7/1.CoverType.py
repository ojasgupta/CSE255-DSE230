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


# In[17]:

# Read the file into an RDD
# If doing this on a real cluster, you need the file to be available on all nodes, ideally in HDFS.
path='/covtype/covtype.data'
inputRDD=sc.textFile(path).cache()


# In[18]:

Label=2.0
Data=inputRDD.map(lambda line: [float(x) for x in line.split(',')])     .map(lambda V: LabeledPoint(1.0, V[:-1]) if V[-1] == Label else LabeledPoint(0.0, V[:-1])).cache()


# ### Reducing data size
# In order to see the effects of overfitting more clearly, we reduce the size of the data by a factor of 10

# In[19]:

(trainingData,testData)=Data.randomSplit([0.7,0.3],seed=255)


# In[ ]:

from time import time
errors={}
start=time()
depth = 15
model=GradientBoostedTrees.trainClassifier(trainingData, {}, loss =  "leastSquaresError", numIterations=10, maxDepth = depth)
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
