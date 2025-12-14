import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,accuracy_score
from sklearn.preprocessing import LabelEncoder
file =  pd.read_csv(r"C:\Users\sumit\OneDrive\Documents\student_lifestyle_dataset.csv")
frame =  pd.DataFrame(file)
print(frame)
label = LabelEncoder()
frame["Stress_Level"] = label.fit_transform(frame["Stress_Level"])
print(frame)
inputs = frame[["Study_Hours_Per_Day","Extracurricular_Hours_Per_Day","Physical_Activity_Hours_Per_Day",
               "GPA"]]
output = frame["Stress_Level"]
x_train , x_test , y_train , y_test = train_test_split(inputs,output , train_size=0.8 , test_size=0.2, random_state=42 )
algo = XGBClassifier()
algo.fit(inputs,output)
value =  algo.predict(x_test)
print("accuracy:",accuracy_score(y_test,value))
print("precision:",precision_score(y_test,value,average="macro"))
# Take user input
study_hours = float(input("Enter Study Hours Per Day: "))
extra_hours = float(input("Enter Extracurricular Hours Per Day: "))
physical_hours = float(input("Enter Physical Activity Hours Per Day: "))
gpa = float(input("Enter GPA: "))
final = algo.predict([[study_hours,extra_hours,physical_hours,gpa]])
label.fit(file["Stress_Level"])
encoded = label.inverse_transform([final[0]])[0]
print("Your Stress lavel is ::",encoded)