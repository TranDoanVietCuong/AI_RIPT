import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


iris = load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target 
print(df)

print("\nTên các loài hoa:", iris.target_names)

from sklearn.model_selection import train_test_split

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Số mẫu để học: {X_train.shape[0]}")
print(f"Số mẫu để thi: {X_test.shape[0]}")


model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)

#X_new = np.array([[6.1, 3.22, 5.0, 2.058]])
y_pred = model.predict(X_test)
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác của mô hình: {accuracy * 100:.2f}%")

