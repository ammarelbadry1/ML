# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('Battery_RUL.csv')

# Split the data into features and target variable
X = data.drop('RUL', axis=1)
y = data['RUL']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the decision tree regressor model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate the accuracy of the predictions
accuracy = model.score(X_test, y_test)

print(f'Accuracy: {accuracy * 100:.2f}%')

# Visualize the predictions and actual values
plt.scatter(range(len(y_test)), y_test, 
            c='b', 
            label='Actual values')

plt.scatter(range(len(y_pred)), y_pred, 
            c='r', 
            marker='x', 
            label='Predicted values')

plt.xlabel('Data Points')
plt.ylabel('Target Variable')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()
