import pandas as pd
import numpy as np
import pickle


url='https://raw.githubusercontent.com/Sinchiguano/ai_googleColab/main/datasetUleamMl/IRIS.csv'
df=pd.read_csv(url)
tmp=list()
for specie in df['species']:
  if specie =='Iris-setosa':
    tmp.append(0)
  elif specie=='Iris-versicolor':
    tmp.append(1)
  else:
    tmp.append(2)

df['target']=tmp

x=df.drop(['species','target'],axis=1)
x=x.to_numpy()[:,(2,3)]
y=df['target']

#SPLIT THE DATA  INTO TRAIN AND TEST DATASETS
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.8,random_state=45)

# Load the pickled model
pickled_model = pickle.load(open('iris.pkl', 'rb'))
predictions=pickled_model.predict(xtest)


labels=list()
labels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
from sklearn import preprocessing
decipher=preprocessing.LabelEncoder()
decipher.fit(labels)


tmpPredictions=decipher.inverse_transform(predictions)

for i in tmpPredictions[:10]:
  print('The prediction is : {}'.format(i))




print('done......')