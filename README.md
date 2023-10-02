# Obesity Prediction: Python Web Applications by Flask

Code snippets supplementing the [Python Web Applications: Deploy Your Script as a Flask App](https://realpython.com/python-web-applications/) tutorial.

## Running Locally

Create and activate a Python virtual environment:

```shell
$ python -m venv venv
$ source venv/bin/activate
```

Update `pip` and install the required dependencies:

```shell
(venv) $ pip install -U pip
(venv) $ pip install -r requirements.txt
```

Start the Flask server:

```shell
(venv) $ python main.py
 * Serving Flask app "main" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
 * Running on http://127.0.0.1:8080/ (Press CTRL+C to quit)
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 339-986-221
```

Navigate your web browser to this address: <http://127.0.0.1:8080/>

# Model Training
This machine learning model predicts the chance of obesity. Prediction choses between diagnosised-obesity and not-obesity. The dataset is taken from ScienceDirect Data in Brief (https://www.sciencedirect.com/science/article/pii/S2352340919306985?via%3Dihub). Here are the key features of this project:

The dataset and the current model is tracked using a GCP (Google Cloud) bucket.

The main.py saves all required information to run the model in another machines through a container.

The exam2.csv is the dataset used to train the model.

The test.py is used to make sure the server and the model is running without bugs. 

Attribute Information:
Prediction (1 = obesity, 0 = not obesity)

11 features are inputs to make prediction:
Age, 
Height (m,for example 1.80), 
Weight (kg, for example 70kg), 
family_history_with_overweight (Has a family member suffered or suffers from overweight, 1 means yes, 0 means no), 
FAVC (Frequent consumption of high caloric food, 1 means yes, 0 means no),
FCVC (Frequency of consumption of vegetables, Do you usually eat vegetables in your meals, 1 means never, 2 means sometimes, 3 means always),
NCP (Number of main meals per day, 1 means between one-two, 2 means three, 3 means more than three),
SMOKE (0 means no, 1 means yes),
CH2O (How much water do you drink daily, 1 means less than 1L, 2 means 1-2L, 3 means >2L),
FAF (Physical activity frequency per week, 0 means do not have, 2 means one or two days, 3 means 2 or 3 days, 4 means 4 or 5 days),
TUE (How much time do you use technological devices such as cell phone, videogames, television, computer and others, 0 means 0-2h, 1 means 3-5, 2 means >5h).

Feature Importances are: (from Age to TUE)
[0.09307822 0.08364308 0.53864099 0.08428796 0.0160771  0.02188126  0.03630806 0.00243559 0.03470068 0.05677933 0.03216774]

Based on Scikit-Learn modules and functions such like:
from sklearn.model_selection import train_test_split
        X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.25, random_state=0)
        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc_X = StandardScaler()
        X_Train = sc_X.fit_transform(X_Train)
        # Fitting the classifier into the Training set
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators=200, random_state=0)
        classifier.fit(X_Train, Y_Train)
The model got a 0.9816 of f1 score and a 0.973 of accuracy.

The confusion matrix is:
[[141   6]
 [  8 373]]
