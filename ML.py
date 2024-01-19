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

# Load the dataset
# dataset_url = "https://www.dropbox.com/scl/fi/8et6xuwh9luvfg03hhji3/Data.csv?rlkey=10s6tu2sgw5z3ft43qk3wey79&dl=0"
col_names = ['Age', 'Gender', 'BMI', 'Region', 'No.Children', 'Insurance_Charges', 'Smoker']
feature_columns = ['Age', 'Gender', 'BMI', 'Region', 'No.Children', 'Insurance_Charges']
dataset = pd.read_csv(r"C:\Users\Asus ZenBook\Downloads\Data.csv",  skiprows=1, names=col_names)

def exploreData() :
    print(dataset.head())

# Task 1: Show the distribution of the class label (Smoker)
def plotDistribution() :
    sns.countplot(x='Smoker', data=dataset)
    plt.title('Distribution of Smoker')
    plt.show()


def plotDensity(x):
    sns.kdeplot(dataset[x], fill=True)
    plt.title(f'Density Plot for {x}')
    plt.show()

def Visualize() :
    sns.scatterplot(x='Age', y='BMI', hue='Region', data=dataset)
    plt.title('Scatterplot of Age vs. BMI')
    plt.show()

def knn () :
    k_values = [3, 7, 40]

    for k in k_values:
        # Initialize KNN Classifier
        knn_classifier = KNeighborsClassifier(n_neighbors=k)

        # Fit the model on the training data
        knn_classifier.fit(X_train, y_train)

        # Predict on the test data
        y_pred = knn_classifier.predict(X_test)

        # Model Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for k={k}: {accuracy}")

        # Additional metrics
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))

def DecisionTree() :
    # Decision Tree Model Training
    # Initialize Decision Tree Classifier (you can adjust hyperparameters as needed)
    dt_classifier = DecisionTreeClassifier(random_state=42)

    # Fit the model on the training data
    dt_classifier.fit(X_train, y_train)

    # Predict on the test data
    y_pred = dt_classifier.predict(X_test)

    # Model Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for Decision Tree: {accuracy}")

    # Additional metrics
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

def NaiveBayes() :
    nb_classifier = GaussianNB()

    # Fit the model on the training data
    nb_classifier.fit(X_train, y_train)

    # Predict on the test data
    y_pred = nb_classifier.predict(X_test)

    # Model Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for Naive Bayes: {accuracy}")

    # Additional metrics
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

def ANN() :
    ann_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, activation='logistic', random_state=42)

    # Fit the model on the training data
    ann_classifier.fit(X_train, y_train)

    # Predict on the test data
    y_pred_ann = ann_classifier.predict(X_test)

    # Model Evaluation
    accuracy_ann = accuracy_score(y_test, y_pred_ann)
    print(f"Accuracy for ANN: {accuracy_ann}")

    # Additional metrics for ANN
    print(classification_report(y_test, y_pred_ann, zero_division=1))
    print(confusion_matrix(y_test, y_pred_ann))

#Task 0 : Explore Data
exploreData()
#Task 1 : show class Distribution for Smoker
plotDistribution()
# Task 2: Show the density plot for Age
plotDensity('Age')
# Task 3: Show the density plot for BMI
plotDensity('BMI')
# Task 4 : Visualize the scatterplot of data and split based on Region attribute
Visualize();
# Handle categorical variables (one-hot encoding)
X = pd.get_dummies(dataset[feature_columns], columns=['Gender', 'Region'], drop_first=True)
y = dataset['Smoker']
#Task 5 :  Split the dataset into training (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Task 6 : KNN (using 3 different	values	of	K)
knn();
# Task 7 :	Decision	Trees (C4.5)
DecisionTree();
#Task 8  : Naive Bayes Classifier
NaiveBayes();
#Task 9 : Artificial Neural Network (ANN) with a Single Hidden Layer, Number of Epochs = 500, Sigmoid Activation Function
ANN();
