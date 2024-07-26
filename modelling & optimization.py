import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import classification_report, accuracy_score
from sklearn.dummy import DummyClassifier

sns.set(style='whitegrid', font_scale=0.8)
%matplotlib inline

# Load the dataset
df_load = pd.read_csv("C:\\Users\\prana\\OneDrive\\Documents\\Dataset\\heart_cleveland_upload.csv")
print(df_load.head())

# Examine the dataset
df_load.info()
display(df_load.describe())
print(df_load['condition'].value_counts())

# Correlation Matrix
def correlation_matrix(df):
    corr_mat = df.corr().round(2)
    f, axis = plt.subplots(figsize=(10, 5))
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    mask = mask[1:, :-1]
    corr = corr_mat.iloc[1:, :-1].copy()
    sns.heatmap(corr, mask=mask, vmin=-0.3, vmax=0.3, center=0, 
                cmap='RdPu_r', square=False, lw=2, annot=True, cbar=False)
    axis.set_title('Correlation Matrix')
    plt.show()

correlation_matrix(df_load)

# Define Plots Functions
plt4 = ['#480b1b','#d42955']

def plot1count(x, xlabel, palt):
    plt.figure(figsize=(20, 2))
    sns.countplot(x=x, hue='condition', data=df_load, palette=palt)
    plt.legend(["<50% diameter narrowing", ">50% diameter narrowing"], loc='upper right')
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.show()
    
def plot1count_ordered(x, xlabel, order, palt):
    plt.figure(figsize=(20, 2))
    sns.countplot(x=x, hue='condition', data=df_load, order=order, palette=palt)
    plt.legend(["<50% diameter narrowing", ">50% diameter narrowing"], loc='upper right')
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.show()

def plot2count(x1, x2, xlabel1, xlabel2, colour, rat, ind1=None, ind2=None):
    fig, ax = plt.subplots(1, 2, figsize=(20, 3), gridspec_kw={'width_ratios':rat})
    sns.countplot(x=x1, hue='condition', data=df_load, order=ind1, palette=colour, ax=ax[0])
    ax[0].legend(["<50% diameter narrowing", ">50% diameter narrowing"], loc='upper right')
    ax[0].set_xlabel(xlabel1)
    ax[0].set_ylabel('Frequency')

    sns.countplot(x=x2, hue='condition', data=df_load, order=ind2, palette=colour, ax=ax[1])
    ax[1].legend(["<50% diameter narrowing", ">50% diameter narrowing"], loc='best')
    ax[1].set_xlabel(xlabel2)
    ax[1].set_ylabel('Frequency')
    plt.show()

def nplot2count(lst_name, lst_label, colour, n_plots):
    fig, ax = plt.subplots(1, n_plots, figsize=(20, 3))
    for i in range(n_plots):
        sns.countplot(x=lst_name[i], hue='condition', data=df_load, palette=colour, ax=ax[i])
        ax[i].legend(["<50% diameter narrowing", ">50% diameter narrowing"], loc='upper right')
        ax[i].set_xlabel(lst_label[i])
        ax[i].set_ylabel('Frequency')

# General Feature & Pain Related Features
plot2count('age', 'sex', 'Age of Patient', 'Gender of Patient', plt4, [2, 1])
lst1 = ['cp', 'exang', 'thal', 'ca']
lst2 = ['Chest Pain Type', 'Excersised Induced Angina', 'Thalium Stress Result', 'Fluorosopy Vessels']
nplot2count(lst1, lst2, plt4, 4)

# ECG Related features
lst_ecg = ['oldpeak', 'restecg', 'slope', 'condition']
plot1count('oldpeak', 'oldpeak: ST Depression Relative to Rest', plt4)
plot2count('restecg', 'slope', 'restecg: Resting electrocardiography (ECG)', 'slope: []', plt4, [1, 1])

# Blood Related Features
lst_blood = ['trestbps', 'thalach', 'fbs', 'chol', 'condition']
plot1count('trestbps', 'trestbps: Resting Blood Pressure (mmHg)', plt4)
plot1count_ordered('thalach', 'thalach: Maximum Heart Rate', df_load['thalach'].value_counts().iloc[:30].index, plt4)
plot2count('fbs', 'chol', 'Fasting Blood Sugar', 'Serum Cholestoral', plt4, [2, 10], None, df_load['chol'].value_counts().iloc[:40].index)

# Split the data into training and test sets (70:30 ratio)
X = df_load.drop('condition', axis=1)
y = df_load['condition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=41)

# Gaussian Process Classifier
kernel = 1.0 * RBF(length_scale=1.0)
gpc = GaussianProcessClassifier(kernel=kernel, max_iter_predict=100)
gpc.fit(X_train, y_train)

# Predictions
y_pred = gpc.predict(X_test)

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Dummy Classifier for comparison
model = DummyClassifier(strategy="most_frequent")
model.fit(X_train, y_train)
print(f'DC(): {model.score(X_test, y_test)}')

# Grid search for hyperparameters
lst_theta = [0.01, 0.1, 1, 10, 100, 1000, 5000]
lst_sigma = [0.01, 0.1, 1, 10, 100, 1000, 5000]

def heatmap1(scores, xlabel, xticklabels, ylabel, yticklabels):
    plt.figure(figsize=(10, 8))
    sns.heatmap(scores, annot=True, fmt=".3f", xticklabels=xticklabels, yticklabels=yticklabels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def modelEval(ldf, lst_theta, lst_sigma, feature='condition'):
    X = ldf.copy()
    y = ldf[feature].copy()
    del X[feature]

    param_grid = {
        'kernel__k1__constant_value': lst_theta,
        'kernel__k2__length_scale': lst_sigma
    }

    kernel = 1.0 * RBF(length_scale=1.0)
    model = GaussianProcessClassifier(kernel=kernel)

    gscv = GridSearchCV(model, param_grid, cv=5)
    gscv.fit(X.values, y.values)
    results = pd.DataFrame(gscv.cv_results_)
    scores = np.array(results.mean_test_score).reshape(len(lst_theta), len(lst_sigma))

    heatmap1(scores, xlabel='theta', xticklabels=lst_theta, ylabel='sigma', yticklabels=lst_sigma)

# Model evaluation on different subsets
ldf1 = df_load[lst_ecg]
modelEval(ldf1, lst_theta, lst_sigma)

ldf2 = df_load[lst_blood]
modelEval(ldf2, lst_theta, lst_sigma)

ldf3 = df_load[['age', 'sex'] + lst_ecg]
modelEval(ldf3, lst_theta, lst_sigma)

modelEval(df_load, lst_theta, lst_sigma)
