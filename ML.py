import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.table import table
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier

# Set the seed for reproducibility
# Load the dataset
# dataset_url = "https://www.dropbox.com/scl/fi/8et6xuwh9luvfg03hhji3/Data.csv?rlkey=10s6tu2sgw5z3ft43qk3wey79&dl=0"
col_names = ['Age', 'Gender', 'BMI', 'Region', 'No.Children', 'Insurance_Charges', 'Smoker']
feature_columns = ['Age', 'Gender', 'BMI', 'Region', 'No.Children', 'Insurance_Charges']
# dataset = pd.read_csv(dataset_url, skiprows=1, names=col_names)
dataset = pd.read_csv(r"C:\Users\Asus ZenBook\Downloads\Data.csv",  skiprows=1, names=col_names)
print(dataset.head())
# Task 1: Show the distribution of the class label (Smoker)
sns.countplot(x='Smoker', data=dataset)
plt.title('Distribution of Smoker')
plt.show()

# Task 2: Show the density plot for Age
sns.kdeplot(dataset['Age'], fill=True)
plt.title('Density Plot for Age')
plt.show()

# Task 3: Show the density plot for BMI
sns.kdeplot(dataset['BMI'], fill=True)
plt.title('Density Plot for BMI')
plt.show()

# Task 4: Visualize the scatterplot of data and split based on Region attribute
sns.scatterplot(x='Age', y='BMI', hue='Region', data=dataset)
plt.title('Scatterplot of Age vs. BMI')
plt.show()

# Task 5: Split the dataset into training (80%) and test (20%)
X = dataset[feature_columns]
y = dataset['Smoker']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
