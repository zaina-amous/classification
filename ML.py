import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.table import table
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
# Set the seed for reproducibility
np.random.seed(42)
col_names = ['Age', 'Gender', 'BMI', 'Region', 'No. Childred', 'Insurance Charges', 'DPF', 'Smoker']
feature_colomns = ['Age', 'Gender', 'BMI', 'Region', 'No. Childred', 'Insurance Charges', 'DPF']
dataset = pd.read_csv(r"C:\Users\Asus ZenBook\Desktop\Data.csv", header=1, names=col_names)





