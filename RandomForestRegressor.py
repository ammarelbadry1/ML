import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Reading the dataset
df = pd.read_csv('Battery_RUL.csv')

# Preprocessing
X = df.drop("RUL", axis=1).drop("Cycle_Index", axis=1)
y = df["RUL"]

## Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train the model
model = RandomForestRegressor(n_estimators=20, oob_score=True)
model.fit(X_train, y_train)


# Make predictions
predictions = model.predict(X_test)


# Evaluate the model
oob_score = model.oob_score_
r2 = r2_score(y_test, predictions)

print('Model Evaluation:')
print(f'Out-of-Bag Score: {oob_score}')
print(f'R2 Score: {r2 * 100:.2f}%')
print()

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature Ranking:")
for feature in range(X.shape[1]):
    print("%d. feature %d (%f)" % (feature + 1, indices[feature], importances[indices[feature]]))


# Visualizing Predictions VS Actual Values
plt.plot(y_test, y_test, color='blue')
plt.scatter(y_test, predictions, color='red')
plt.title("Visualizing Predictions VS Actual Values")
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()


# Visualizing Error Distribution
errors = y_test - predictions
plt.hist(errors, bins=100)
plt.title("Visualizing Error Distribution")
plt.xlabel('Error')
plt.ylabel('Count')
plt.show()
