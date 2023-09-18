import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

# Question 1
# Get the version of Pandas
print("Pandas version:", pd.__version__)

# Question 2
# Read the dataset
df = pd.read_csv('housing.csv')

# Get the number of columns in the dataset
print("Number of columns:", len(df.columns))

# Question 3
# Check for missing values in each column
print("Columns with missing values:", df.columns[df.isnull().any()].tolist())

# Question 4
# Get the number of unique values in the 'ocean_proximity' column
print("Number of unique values in 'ocean_proximity':", len(df['ocean_proximity'].unique()))

# Question 5
# Calculate the average of 'median_house_value' for houses near the bay
avg = df[df['ocean_proximity'] == 'NEAR BAY']['median_house_value'].mean()
print("Average median house value for houses near the bay:", avg)

# Question 6

# Fill missing values in 'median_house_value' with the mean value
# Calculate the average of 'median_house_value' column again
print("Average of 'median_house_value' column after filling missing values:",
      df[df['ocean_proximity'] == 'NEAR BAY']['median_house_value'].fillna(avg).mean())

# Question 7
# Select options located on islands and specific columns
islands = df[df['ocean_proximity'] == 'ISLAND'][['housing_median_age', 'total_rooms', 'total_bedrooms']]

# Get the underlying NumPy array
X = islands.values

# Compute matrix-matrix multiplication between the transpose of X and X
XTX = np.dot(X.T, X)

# Compute the inverse of XTX
XTX_inverse = np.linalg.inv(XTX)

# Create an array y
y = np.array([950, 1300, 800, 1000, 1300])

# Multiply the inverse of XTX with the transpose of X, and then multiply the result by y
w = np.dot(np.dot(XTX_inverse, X.T), y)

# Get the value of the last element of w
print("Value of the last element of w:", w[-1])
