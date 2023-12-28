from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd

# ____________KNN Algorithm____________


data = pd.read_csv("Battery_RUL.csv")

features = [
    "Discharge Time (s)",
    "Decrement 3.6-3.4V (s)",
    "Max. Voltage Dischar. (V)",
    "Min. Voltage Charg. (V)",
    "Time at 4.15V (s)",
    "Time constant current (s)",
    "Charging time (s)",
]

X = data[features]
y = data["RUL"]

"""
Split the data into training and testing sets
X_train: 80% of the dataSet with the features
y_train: 80% of the dataSet targets
X_test: 20% of the dataSet with the features
y_test: 20% of the dataSet targets for testing and comparing
"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

knn = KNeighborsRegressor(n_neighbors=4)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean squared error:", mse)
print("r2 score: ", r2)


plt.plot(y_test, y_test, color='blue')
plt.scatter(y_test, y_pred, color='red', s=3)
plt.title("Test Values VS Prediction Values")
plt.xlabel("True Values")
plt.ylabel("Prediction Values")
plt.legend()
plt.show()
