
url='https://raw.githubusercontent.com/Sinchiguano/ai_googleColab/main/datasetUleamMl/heart.csv'
import pandas as pd
dataset=pd.read_csv(url)

print('Sex: {}'.format(dataset['Sex'].unique()))
print ('ChestPainType: {}'.format(dataset['ChestPainType'].unique()))
print ('RestingECG: {}'.format(dataset['RestingECG'].unique()))
print ('ExerciseAngina: {}'.format(dataset['ExerciseAngina'].unique()))
print ('ST_Slope: {}'.format(dataset['ST_Slope'].unique()))
print("done....")
print("done....")
print("done....")

SexNum=[]
print('Row Sixze: {}'.format(len(dataset['HeartDisease'])))
for i in range(len(dataset['HeartDisease'])):
  if dataset['Sex'][i]=='M':
    SexNum.append(0)
  else:
    SexNum.append(1)

ChestPainTypeNum=[]
for i in range(len(dataset['HeartDisease'])):
  if dataset['ChestPainType'][i]=='ATA':
    ChestPainTypeNum.append(0)
  elif dataset['ChestPainType'][i]=='NAP':
    ChestPainTypeNum.append(1)
  elif dataset['ChestPainType'][i]=='ASY':
     ChestPainTypeNum.append(2)
  else:
    ChestPainTypeNum.append(3)

    RestingEGGNum=[]
#print('Row Sixze: {}'.format(len(dataset['HeartDisease'])))
for i in range(len(dataset['HeartDisease'])):
  if dataset['RestingECG'][i]=='Normal':
    RestingEGGNum.append(0)
  elif dataset['RestingECG'][i]=='ST':
     RestingEGGNum.append(1)
  else:
    RestingEGGNum.append(2)

    ExerciseAnginaNum=[]
#print('Row Sixze: {}'.format(len(dataset['HeartDisease'])))
for i in range(len(dataset['HeartDisease'])):
  if dataset['ExerciseAngina'][i]=='N':
   ExerciseAnginaNum.append(0)
  else:
    ExerciseAnginaNum.append(1)

    ST_SlopeNum=[]
#print('Row Sixze: {}'.format(len(dataset['HeartDisease'])))
for i in range(len(dataset['HeartDisease'])):
  if dataset['ST_Slope'][i]=='Up':
    ST_SlopeNum.append(0)
  elif dataset['ST_Slope'][i]=='Flat':
     ST_SlopeNum.append(1)
  else:
    ST_SlopeNum.append(2)

datasetHeart=dataset.copy(deep=True)

# print(dataset['Sex'].head(5))

datasetHeart['Sex']=SexNum
# print(datasetHeart['Sex'].head(5))

datasetHeart['Sex']=SexNum
datasetHeart['ChestPainType']=ChestPainTypeNum
datasetHeart['RestingECG']=RestingEGGNum
datasetHeart['ExerciseAngina']=ExerciseAnginaNum
datasetHeart['ST_Slope']=ST_SlopeNum


# print(dataset.head(5))

# print(datasetHeart.head(5))

# print (dataset.info)

# print (datasetHeart.info)

datasetHeart.to_csv('nuevo.csv')

# import matplotlib.pyplot as plt
# datasetHeart.plot(kind='box',subplots=True,layout=(4,4),sharex=False)
# plt.show()

# datasetHeart.hist()
# plt.show()

#datasetHeart = datasetHeart.drop(['Cholesterol','Oldpeak','RestingECG'],axis=1)
x=datasetHeart.drop('HeartDisease',axis=1)
y=datasetHeart['HeartDisease']
print(x.head(3))

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
x=scaler.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.75)

# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier()
# model.fit(x_train,y_train)
# yPredictionOnTesting=model.predict(x_test)
# yPredictionOnTraining=model.predict(x_train)


from sklearn import metrics
# print('TRAIN ACCURACY SCORE:{}'.format(metrics.accuracy_score(y_train,yPredictionOnTraining)))
# print('TEST ACCURACY SCORE:{}'.format(metrics.accuracy_score(y_test,yPredictionOnTesting)))
# print('CONFUSION MATRIX :\n{}'.format(metrics.confusion_matrix(y_test,yPredictionOnTesting)))


# from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train)
yPredictionOnTesting=model.predict(x_test)
yPredictionOnTraining=model.predict(x_train)
from sklearn import metrics
print('TRAIN ACCURACY SCORE:{}'.format(metrics.accuracy_score(y_train,yPredictionOnTraining)))
print('TEST ACCURACY SCORE:{}'.format(metrics.accuracy_score(y_test,yPredictionOnTesting)))
print('CONFUSION MATRIX :\n{}'.format(metrics.confusion_matrix(y_test,yPredictionOnTesting)))


import pickle
filename='model1.pkl'
pickle.dump(model,open (filename, 'wb'))
print('done')


pickled_model = pickle.load(open('model1.pkl', 'rb'))
pickled_model.predict(x_test)
