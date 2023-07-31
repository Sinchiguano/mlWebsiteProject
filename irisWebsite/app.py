from flask import Flask, render_template, request
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn import preprocessing

app = Flask(__name__)

# Load the pickled model
model = pickle.load(open('iris.pkl', 'rb'))


labels=list()
labels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

decipher=preprocessing.LabelEncoder()
decipher.fit(labels)




@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # features = [float(x) for x in request.form.values()]
    a=float(request.form["petal_length"])
    b=float(request.form["petal_width"])
    input_features = [[a,b]]
    prediction = model.predict(input_features)
    predictionLabel = decipher.inverse_transform(prediction)
    return render_template('result.html', prediction=predictionLabel)

if __name__ == '__main__':
    app.run(debug=True)



