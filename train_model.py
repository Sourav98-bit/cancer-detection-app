import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

print("Loading data...")
try:
    df = pd.read_csv('breast_cancer_dataframe.csv')
except FileNotFoundError:
    print("ERROR: 'breast_cancer_dataframe.csv' not found in this folder.")
    exit()

print("Training model...")
X = df.drop(['target'], axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

model = XGBClassifier()
model.fit(X_train, y_train)

print("Saving model...")
pickle.dump(model, open('breast_cancer_detector.pickle', 'wb'))
print("Success! New model generated.")