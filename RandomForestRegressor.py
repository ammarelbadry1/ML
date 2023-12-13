import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.tree import plot_tree

# Reading the dataset
df = pd.read_csv('Battery_RUL.csv')

# Preprocessing
X = df.drop("RUL", axis=1)
y = df["RUL"]

## Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
# model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', random_state=42, oob_score=True)
model.fit(X_train, y_train)


# Make predictions
predictions = model.predict(X_test)


# Evaluate the model
oob_score = model.oob_score_
r2 = r2_score(y_test, predictions)

print('Model Evaluation:')
print(f'Out-of-Bag Score: {oob_score}')
print(f'R2 Score: {r2}')
print()

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature Ranking:")
for feature in range(X.shape[1]):
    print("%d. feature %d (%f)" % (feature + 1, indices[feature], importances[indices[feature]]))


# Visualizing Predictions VS Actual Values
X_test = X_test.iloc[:, 0:1].values
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, predictions, color='red')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()


# Visualizing Error Distribution
errors = y_test - predictions
plt.hist(errors, bins=200)
plt.xlabel('Error')
plt.ylabel('Count')
plt.show()
