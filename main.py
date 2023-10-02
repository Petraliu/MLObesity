import joblib
from flask import Flask
from flask import request
import pandas as pd


app = Flask(__name__)

@app.route("/")
#collect data
def index():
    k = request.args.get("k", "")
    a = request.args.get("a", "")
    b = request.args.get("b", "")
    c = request.args.get("c", "")
    d = request.args.get("d", "")
    e = request.args.get("e", "")
    f = request.args.get("f", "")
    g = request.args.get("g", "")
    h = request.args.get("h", "")
    i = request.args.get("i", "")
    j = request.args.get("j", "")
    if k:
        if a:
            if b:
                if c:
                    if d:
                        if e:
                            if f:
                                if g:
                                    if h:
                                        if i:
                                            if j:
                                                fahrenheit = fahrenheit_from(k,a,b,c,d,e,f,g,h,i,j)

#data input
    else:
        fahrenheit = "missing input"
    return (
        """<form action="" method="get">
                Age : <input type="text" name="k">
                Height:<input type="text" name="a">
                Weight:<input type="text" name="b">
                family_history_with_overweight:<input type="text" name="c">
                FAVC:<input type="text" name="d">
                FCVC:<input type="text" name="e">
                NCP:<input type="text" name="f">
                SMOKE:<input type="text" name="g">
                CH2O:<input type="text" name="h">
                FAF:<input type="text" name="i">
                TUE:<input type="text" name="j">
                <input type="submit" value="Test your chance">
            </form>"""
        + "Suggestion: "
        + fahrenheit
    )

#run model and predict after getting data
def fahrenheit_from(k,a,b,c,d,e,f,g,h,i,j):
    try:
        datasets = pd.read_csv('exam2.csv')
        X = datasets.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]].values
        Y = datasets.iloc[:, 11].values
        # Splitting the dataset into the Training set and Test set

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
        #joblib.dump(classifier,"my_model.joblib")
        #loaded_model=joblib.load("my_model.joblib")
        input_data = pd.DataFrame([[k, a, b, c, d, e, f, g, h, i, j]],
                                  columns=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6',
                                           'Feature7', 'Feature8', 'Feature9', 'Feature10', 'Feature11'])
        input_data = sc_X.transform(input_data)
        output = classifier.predict(input_data)
        if output == 1:
            return str("yes, you may have obesity")
        if output == 0:
            return str("no, you may not have obesity")
    except ValueError:
        return "invalid input"

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)