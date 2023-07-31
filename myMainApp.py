from flask import Flask, render_template, request, redirect

app = Flask(__name__)

# Hardcoded credentials for demonstration purposes
USERNAME = 'sinchiguano'
PASSWORD = 'sinchiguano'




from flask import Flask, render_template, request
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train a K-Nearest Neighbors classifier
knn = KNeighborsClassifier()
knn.fit(X, y)

@app.route('/')
def index():
    return render_template('login.html')




@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    if username == USERNAME and password == PASSWORD:
        # Redirect to the classifier page
        return render_template('classifier.html')
    else:
        error_message = 'Invalid username or password.'
        return render_template('login.html', error_message=error_message)


@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Make a prediction using the input values
    X_input = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = knn.predict(X_input)
    target_names = iris.target_names

    return render_template('result.html', prediction=prediction[0], target_names=target_names)



if __name__ == '__main__':
    app.run()
