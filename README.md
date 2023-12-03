# House-Pricing-Prediction-Using-Linear-Regression

## Import all libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

## Step 1: Dataset Exploration and preprocessing
#Load dataset from csv
df = pd.read_csv("data.csv")

# Exploratory Data Analysis (EDA) 
print(df.head())

# Summary statistics of the dataset
print(df.describe())

# check for missing values
df.isnull().sum()

# correlation matrix to understand feature relationships
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot = True, cmap = "coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Preprocessing: Selecting features and target varaibles
X = df[["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view", "condition"]]
y = df["price"]

# SPlitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Step 2: Building the Linear Regression Model
#building the linear regression model
model = LinearRegression()

# fitting the model on the training data
model.fit(X_train, y_train)

## Step 3: Model Evaluation: 
#Model Evaluation
y_pred = model.predict(X_test)

#Mean Squared Error and R-squared for model evaluation
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print("Mean Squared Error:", mse)
print("R-squared:" , r2)

## Step 4: Predictions and Visualization
# Predictions and Visualization
# To visuaize the prediction against actual prices, we'll use a scatter plot
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices VS Predicted Prices")
plt.show()

# We can also create a residual plot to check the model's performance
residuals = y_test - y_pred
plt.scatter(y_test, residuals)
plt.axhline(y=0, color = "red", linestyle = "--")
plt.xlabel("Actual Prices")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# #Finally we can use the trained model to make predictions on new data and visualize the results
new_data = [[3, 2, 1500, 4000, 1, 0, 0, 3]]
predicted_price = model.predict(new_data)

print("Predicted Price:", predicted_price[0])


