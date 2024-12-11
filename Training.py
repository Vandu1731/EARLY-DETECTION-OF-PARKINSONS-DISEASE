import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score  # Import accuracy_score

# Read the data
data = pd.read_csv('parkinsons.csv')

# Select features and target
X = data[['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:RAP','RPDE']]  # Features
Y = data['status']  # Target variable

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2)
X_train_res, y_train_res = sm.fit_resample(X,Y)


# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred = dt_model.predict(X_test)

# Calculate accuracy
accuracy1 = accuracy_score(Y_test, Y_pred)

# Print accuracy
print(f"DT Accuracy: {accuracy1 * 100:.2f}%")

# Save the trained model to a file (optional, for reusability)
import joblib
joblib.dump(dt_model, 'decision_tree_model.pkl')

#print("Model training completed and saved as 'decision_tree_model.pkl'")

from sklearn.ensemble import RandomForestClassifier
Rf = RandomForestClassifier()
Rf.fit(X_train,Y_train)
pred=Rf.predict(X_test)
accuracy2 = accuracy_score(Y_test,pred)
#print(f"Random forest accuracy: {accuracy}")
print(f"RF Accuracy: {accuracy2 * 100:.2f}%")
import joblib
joblib.dump(pred, 'Random_forest_model.pkl')



import xgboost as xgb
from xgboost.sklearn import XGBClassifier
Xgb = XGBClassifier()
Xgb.fit(X_train,Y_train)
predXgb=Xgb.predict(X_test)
accuracy3 = accuracy_score(Y_test,predXgb)
#print(f"Random forest accuracy: {accuracy}")
print(f"xgb Accuracy: {accuracy3 * 100:.2f}%")
import joblib
joblib.dump(predXgb, 'Xgboost_model.pkl')







from sklearn.neighbors import KNeighborsClassifier
Knn = KNeighborsClassifier()
Knn.fit(X_train,Y_train)
predKnn=Knn.predict(X_test)
accuracy5 = accuracy_score(Y_test,predKnn)
print(f"Model Accuracy: {accuracy5 * 100:.2f}%")
import joblib
joblib.dump(predKnn, 'Knn.pkl')

import numpy as np
import matplotlib.pyplot as plt
 
  
# creating the dataset
data = {'DT':accuracy1, 'RT':accuracy2, 'Xgb':accuracy3,'Knn':accuracy5}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color ='maroon',
        width = 0.4)
 
plt.xlabel("Different Models")
plt.ylabel("Training Accuracy of Models")
plt.title("Accuracy Comparison of Models")
plt.show()


