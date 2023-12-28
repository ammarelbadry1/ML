import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import os

# Get the current working directory
current_directory = os.getcwd()

# Construct the file path using os.path.join
file_path = os.path.join(current_directory, 'Battery_RUL.csv')

# Load the dataset
dataset = pd.read_csv(file_path)

# Define the features and target
features = ['Cycle_Index', 'Discharge Time (s)', 'Decrement 3.6-3.4V (s)',
            'Max. Voltage Dischar. (V)', 'Min. Voltage Charg. (V)',
            'Time at 4.15V (s)', 'Time constant current (s)',
            'Charging time (s)']
target = 'RUL'

X = dataset[features].values
y = dataset[target].values

# Feature scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1, 1)).ravel()

# SVR model
# SVR model
regressor = SVR(kernel='rbf')

# Fit the model
regressor.fit(X, y)

# Predictions using the original feature set
y_pred_orig = regressor.predict(X)

# Calculate R^2 score
r2 = r2_score(y, y_pred_orig)


# Predictions using entire scaled feature set
X_pred = np.linspace(-2, 2, 1000).reshape(-1, len(features))  # Change -2 and 2 based on your data range
y_pred = regressor.predict(X_pred)
y_pred = sc_y.inverse_transform(y_pred.reshape(-1, 1))

# Reshape X and y for plotting
X_for_plot = sc_X.inverse_transform(X)
y_for_plot = sc_y.inverse_transform(y.reshape(-1, 1)).ravel()


# Plotting
plt.scatter(X_for_plot[:, 0], y_for_plot, color='red', label='Data')
plt.plot(sc_X.inverse_transform(X_pred)[:, 0], y_pred, color='blue', label='SVR model')
plt.title('Battery Remaining Useful Life (RUL)')
plt.xlabel('Cycle_Index')  # Change this label based on the feature you're plotting against RUL
plt.ylabel('RUL')
plt.legend()
plt.show()

# Calculate the errors
errors = y - y_pred_orig

# Plot the error distribution
plt.hist(errors, bins=30)
plt.xlabel('Prediction Error')
plt.ylabel('Count')
plt.title('Distribution of Prediction Errors')
plt.show()
print(f'Accuracy: {r2 * 100:.2f}%')
