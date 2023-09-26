import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('/Users/petra/Desktop/study/UCSF/exam0/new/venv/exam2.csv')

X = data.iloc[:, [0,1,2,3,4,5,6,7,8,9,10]].values
y = data.iloc[:, 11].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

rf_model = RandomForestClassifier(n_estimators=200, random_state=0)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print(y_pred)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

accuracy = accuracy_score(y_test, y_pred)

rf_classifier = RandomForestClassifier()
rf_classifier.fit(X, y)

print(rf_model.feature_importances_)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

def predict_obesity(new_data):
    predictions = rf_model.predict(new_data)
    return predictions
