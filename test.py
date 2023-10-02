import joblib
import numpy as np
def RandomForest(Age,Height,Weight,family_history_with_overweight,FAVC,FCVC,NCP,SMOKE,CH2O,FAF,TUE):
    load_rf = joblib.load("my_test.joblib")
    input = np.array([Age, Height, Weight, family_history_with_overweight, FAVC, FCVC, NCP, SMOKE, CH2O, FAF, TUE])
    input = input.reshape(1, -1)
    output = load_rf.predict(input)

    return str(output)
#a= RandomForest(23,1.5,55,1,1,3,3,0,2,1,0)

load_rf = joblib.load("my_test.joblib")
input = np.array([23,1.5,55,1,1,3,3,0,2,1,0])
input = input.reshape(1, -1)
a = load_rf.predict(input)
output = str(a)
print(output)
