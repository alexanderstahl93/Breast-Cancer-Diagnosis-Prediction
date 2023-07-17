## Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

## Step 2: Load & Explore the Dataset
# Load the dataset using pandas
dataset = pd.read_csv('data/breast_cancer_dataset.csv')

# Explore the structure and features of the dataset
print(dataset.head())
print(dataset.shape)
print(dataset.columns)

# Check for missing values
print(dataset.isnull().sum())

## Step 3: Data Preprocessing
# Drop the 'Unnamed: 32' column as it only contains NaN values
dataset = dataset.drop('Unnamed: 32', axis=1)

## Step 4: Perform Label Encoding
# Encode the 'diagnosis' column (M=1, B=0)
label_encoder = LabelEncoder()
dataset['diagnosis'] = label_encoder.fit_transform(dataset['diagnosis'])

# Plot the data
for feature in ['radius_mean', 'texture_mean', 'perimeter_mean']:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='diagnosis', y=feature, data=dataset)
    plt.title('Box plot of {} by Diagnosis'.format(feature))
    plt.show()

## Step 5: Split the Data
# Split the dataset into independent features (X) and the target variable (y)
X = dataset.drop(['radius_mean', 'diagnosis'], axis=1)
y = dataset['diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

## Step 6: Feature Scaling
# Perform feature scaling on the independent variables
scaler = StandardScaler()
# Apply scaling to training and testing data separately
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

## Step 7: Building Logistic Regression Classifier
# Create an instance of Logistic Regression classifier
classifier = LogisticRegression()
# Fit the model on the training data
classifier.fit(X_train_scaled, y_train)
# Make predictions on the testing data
y_pred = classifier.predict(X_test_scaled)

## Step 8: Evaluate Model Performance
# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion Matrix:\n", confusion_mat)