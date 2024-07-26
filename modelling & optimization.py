import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', font_scale=0.8)

from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset into the variable df_load
df_load = pd.read_csv("C:\\Users\\prana\\OneDrive\\Documents\\Dataset\\heart_cleveland_upload.csv")

import os,warnings;warnings.filterwarnings("ignore")
import numpy as np;import pandas as pd;import matplotlib.pyplot as plt
import seaborn as sns;sns.set(style='whitegrid',font_scale=0.8)
# from mlmodels.gpr_bclassifier import GPRC
%matplotlib inline

#Let's load the dataset into the variable df_load.
df_load = pd.read_csv("C:\\Users\\prana\\OneDrive\\Documents\\Dataset\\heart_cleveland_upload.csv")
print(df_load)

#examine some of the items that are frequently examined when sifting through the dataset 
df_load.info()

# Check the sample
df_load.head()

#Let's check the statistics of our numerical data.
display(df_load.describe())

#Let's take a look at the distribution for our target variable
df_load.condition.value_counts()

import os,warnings;warnings.filterwarnings("ignore")
import numpy as np;import pandas as pd;import matplotlib.pyplot as plt
import seaborn as sns;sns.set(style='whitegrid',font_scale=0.8)
# from mlmodels.gpr_bclassifier import GPRC
%matplotlib inline

#Let's load the dataset into the variable df_load.
df_load = pd.read_csv("C:\\Users\\prana\\OneDrive\\Documents\\Dataset\\heart_cleveland_upload.csv")
print(df_load)

#examine some of the items that are frequently examined when sifting through the dataset 
df_load.info()

# Check the sample
df_load.head()

#Let's check the statistics of our numerical data.
display(df_load.describe())

#Let's take a look at the distribution for our target variable
df_load.condition.value_counts()

#CORRELATION MATRIX
def correlation_matrix(df_load,id=False):
    
    corr_mat = df_load.corr().round(2)
    f, axis = plt.subplots(figsize=(10,5))
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    mask = mask[1:,:-1]
    corr = corr_mat.iloc[1:,:-1].copy()
    sns.heatmap(corr,mask=mask,vmin=-0.3,vmax=0.3,center=0, 
                cmap='RdPu_r',square=False,lw=2,annot=True,cbar=False)
#     bottom, top = axis.get_ylim() 
#     axis.set_ylim(bottom + 0.5, top - 0.5) 
    axis.set_title('Correlation Matrix')
   

correlation_matrix(df_load)

# Define Plots Functions

plt4 = ['#480b1b','#d42955']
def plot1count(x,xlabel,palt):
    
    plt.figure(figsize=(20,2))
    sns.countplot(x=x,hue='condition', data=df_load, palette=palt)
    plt.legend(["<50% diameter narrowing", ">50% diameter narrowing"],loc='upper right')
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.show()
    
def plot1count_ordered(x,xlabel,order,palt):
    
    plt.figure(figsize=(20,2))
    sns.countplot(x=x,hue='condition',data=df_load,order=order,palette=palt)
    plt.legend(["<50% diameter narrowing", ">50% diameter narrowing"],loc='upper right')
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.show()

def plot2count(x1,x2,xlabel1,xlabel2,colour,rat,ind1=None,ind2=None):
    
    # colour, ratio, index_sort

    fig,ax = plt.subplots(1,2,figsize=(20,3),gridspec_kw={'width_ratios':rat})
    # Number of major vessels (0-3) colored by flourosopy
    sns.countplot(x=x1,hue='condition',data=df_load,order=ind1,palette=colour,ax=ax[0])
    ax[0].legend(["<50% diameter narrowing", ">50% diameter narrowing"],loc='upper right')
    ax[0].set_xlabel(xlabel1)
    ax[0].set_ylabel('Frequency')

    # Defect Information (0 = normal; 1 = fixed defect; 2 = reversable defect )
    sns.countplot(x=x2,hue='condition', data=df_load,order=ind2,palette=colour,ax=ax[1])
    ax[1].legend(["<50% diameter narrowing", ">50% diameter narrowing"],loc='best')
    ax[1].set_xlabel(xlabel2)
    ax[1].set_ylabel('Frequency')
    plt.show()
    
''' Plot n Countplots side by side '''
def nplot2count(lst_name,lst_label,colour,n_plots):
    
    ii=-1;fig,ax = plt.subplots(1,n_plots,figsize=(20,3))
    for i in range(0,n_plots):
        ii+=1;id1=lst_name[ii];id2=lst_label[ii]
        sns.countplot(x=id1,hue='condition',data=df_load,palette=colour,ax=ax[ii])
        ax[ii].legend(["<50% diameter narrowing", ">50% diameter narrowing"],loc='upper right')
        ax[ii].set_xlabel(id2)
        ax[ii].set_ylabel('Frequency')


#General Feature & Pain Related Features
plot2count('age','sex','Age of Patient','Gender of Patient',plt4,[2,1])
lst1 = ['cp','exang','thal','ca']
lst2 = ['Chest Pain Type','Excersised Induced Angina','Thalium Stress Result','Fluorosopy Vessels']
nplot2count(lst1,lst2,plt4,4)

#ECG Related features
lst_ecg = ['oldpeak','restecg','slope','condition']
plot1count('oldpeak','oldpeak: ST Depression Relative to Rest',plt4)
plot2count('restecg','slope','restecg: Resting electrocardiography (ECG)','slope: []',plt4,[1,1])

#Blood Related Features
lst_blood = ['trestbps','thalach','fbs','chol','condition']
plot1count('trestbps','trestbps: Resting Blood Pressure (mmHg)',plt4)
plot1count_ordered('thalach','thalach: Maximum Heart Rate',df_load['thalach'].value_counts().iloc[:30].index,plt4)
plot2count('fbs','chol','Fasting Blood Sugar','Serum Cholestoral',plt4,[2,10],None,df_load['chol'].value_counts().iloc[:40].index)


import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier as DC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import GridSearchCV, cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target 

# Dummy Classifier
model = DC(strategy="most_frequent")
model.fit(X, y)
print(f'DC(): {model.score(X, y)}')

# Gaussian Process Classifier
kernel = 1.0 * RBF(length_scale=1.0)
gpc = GaussianProcessClassifier(kernel=kernel)
gpc.fit(X, y)
print(f'GPC(): {gpc.score(X, y)}')

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
    # Given a dataframe, split feature/target variable
    X = ldf.copy()
    y = ldf[feature].copy()
    del X[feature]

    # Define parameters for gridsearch (theta, sigma)
    param_grid = {
        'kernel__k1__constant_value': lst_theta,
        'kernel__k2__length_scale': lst_sigma
    }

    # Define the model with initial parameters
    kernel = 1.0 * RBF(length_scale=1.0)
    model = GaussianProcessClassifier(kernel=kernel)

    # Grid search with 5-fold cross-validation
    gscv = GridSearchCV(model, param_grid, cv=5)
    gscv.fit(X.values, y.values)
    results = pd.DataFrame(gscv.cv_results_) 
    scores = np.array(results.mean_test_score).reshape(len(lst_theta), len(lst_sigma))

    # Plot the cross-validation mean scores of the 5-fold CV
    heatmap1(scores, xlabel='theta', xticklabels=lst_theta, ylabel='sigma', yticklabels=lst_sigma)

# Example usage with a sample dataframe
# Assuming df_load and lst_ecg are defined appropriately
ldf1 = df_load[lst_ecg]  # Subset of ECG features
modelEval(ldf1, lst_theta, lst_sigma)

''' Cross Validation '''
lst_theta = [10,100, 500, 1000, 1500, 2000, 2500]
lst_sig = [0.01,0.1,1.0,10,50,100, 500]

ldf2 = df_load[lst_blood]
modelEval(ldf2,lst_theta,lst_sig)

lst = ['age','sex'] 
ldf3 = df_load[lst+lst_ecg]
modelEval(ldf3,lst_theta,lst_sig)

''' Categorical Feature Model '''
lst_theta = [10,100, 500, 1000, 1500, 2000, 2500]
lst_sig = [0.01,0.1,1.0,10,50,100, 500]
modelEval(df_load,lst_theta,lst_sig)

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
from sklearn.dummy import DummyClassifier as DC

model = DC(strategy="most_frequent")
model.fit(X_train, y_train)
print(f'DC(): {model.score(X_test, y_test)}')

# Grid search for hyperparameters
from sklearn.model_selection import GridSearchCV

lst_theta = [0.01, 0.1, 1, 10, 100, 1000, 5000]
lst_sigma = [0.01, 0.1, 1, 10, 100, 1000, 5000]

def heatmap1(scores, xlabel, xticklabels, ylabel, yticklabels):
    plt.figure(figsize=(10, 8))
    sns.heatmap(scores, annot=True, fmt=".3f", xticklabels=xticklabels, yticklabels=yticklabels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def modelEval(ldf, lst_theta, lst_sigma, feature='condition'):
    # Given a dataframe, split feature/target variable
    X = ldf.copy()
    y = ldf[feature].copy()
    del X[feature]

    # Define parameters for gridsearch (theta, sigma)
    param_grid = {
        'kernel__k1__constant_value': lst_theta,
        'kernel__k2__length_scale': lst_sigma
    }

    # Define the model with initial parameters
    kernel = 1.0 * RBF(length_scale=1.0)
    model = GaussianProcessClassifier(kernel=kernel)

    # Grid search with 5-fold cross-validation
    gscv = GridSearchCV(model, param_grid, cv=5)
    gscv.fit(X.values, y.values)
    results = pd.DataFrame(gscv.cv_results_) 
    scores = np.array(results.mean_test_score).reshape(len(lst_theta), len(lst_sigma))

    # Plot the cross-validation mean scores of the 5-fold CV
    heatmap1(scores, xlabel='theta', xticklabels=lst_theta, ylabel='sigma', yticklabels=lst_sigma)

# Example usage with a sample dataframe
# Assuming df_load and lst_ecg are defined appropriately
lst_ecg = ['oldpeak','restecg','slope','condition']
ldf1 = df_load[lst_ecg]  # Subset of ECG features
modelEval(ldf1, lst_theta, lst_sigma)

# Cross Validation
lst_theta = [10, 100, 500, 1000, 1500, 2000, 2500]
lst_sig = [0.01, 0.1, 1.0, 10, 50, 100, 500]

lst_blood = ['trestbps', 'thalach', 'fbs', 'chol', 'condition']
ldf2 = df_load[lst_blood]
modelEval(ldf2, lst_theta, lst_sig)

lst = ['age', 'sex']
ldf3 = df_load[lst + lst_ecg]
modelEval(ldf3, lst_theta, lst_sig)

# Categorical Feature Model
modelEval(df_load, lst_theta, lst_sig)
