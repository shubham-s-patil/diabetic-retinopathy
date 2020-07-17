# diabetic-retinopathy
Data Science Mini Project

AIM:
Automated Detection of Diabetic Retinopathy

*[Note : Refer DS Report file for better understanding]*

ABSTRACT:
 Diabetic Retinopathy is one of the leading causes of blindness and eye disease in working age population of developed world. This project is an attempt towards finding an 
 automated way to detect this disease in its early phase. In this project I am using supervised learning methods to classify a given set of images into 5 levels.

OBJECTIVE:
To provide an automated, suitable and sophisticated approach using image processing and pattern recognition so that DR can be detected at early levels easily and damage to retina can be minimized
The ability of complex computing is to perform pattern recognition by creating complex relationships based on input data and then comparing it with performance standards is a big step. 

INTRODUCTION:
Diabetic Retinopathy is a disease which is caused due to long term diabetes. It is the ocular manifestation of diabetes and around 80 percent of population having it, thus having chances of having DR in his visual system. According to ’WHO’ estimation 347 million of world population is having the disease diabetes and about 40-45% of them have some stage of the disease.
  

Research shows that progression to vision impairment can be slowed or averted if DR is detected in early stage of the disease. Testing is done manually by trained professionals in real life which is quite time taking and lengthy process and thus delayed results, treatment 
 
Keywords
Diabetic retinopathy, Deep Neural Network (dnn),
 Logistic regression, KNeighborsClassifier,
 GaussianNb, SVC, 
Linear SVC, RandomForestClassifier,
 DecisionTreeRegressor.


DATA SETS USED :
We have used STRUCTURED, multi-state (Multivariate) data set. This dataset is originally from the National Institute of Diabetes. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database.
The datasets consists of several medical predictor variables and one target variable, Outcome.
 Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.

#reference and attributes of data set..
⦁	Name		             Pima Indians Diabetes
⦁	Data types		Multivariate
⦁	Data task		Classification
⦁	Attribute types	Integer, Real
⦁	Instances		768
⦁	Attributes		8
Year		1990
Area		Life
Description	From National Institute of Diabetes and Digestive and Kidney Diseases; 
                          Includes cost data (donated by Peter Turney)
PROCESSING OF DATA:
DNN
A deep neural network (DNN) has multiple layers between the input and output layers. The DNN finds the correct mathematical manipulation to turn the input into the output. The network moves through the layers calculating the probability of each output. DNNs can model complex non-linear relationships
DNNs are typically feedforward networks in which data flows from the input layer to the output layer without looping back. At first, the DNN creates a map of virtual neurons and assigns random numerical values.

Logistic regression
The appropriate regression analysis to conduct when the dependent variable is dichotomous (binary).  Like all regression analyses, the logistic regression is a predictive analysis.  Logistic regression is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables.
Output = 0 or 1

K-Neighbors_Classifier
KNN is a non-parametric and lazy learning algorithm. Non-parametric means there is no assumption for underlying data distribution. In other words, the model structure determined from the dataset
In KNN, K is the number of nearest neighbors. The number of neighbors is the core deciding factor. K is generally an odd number if the number of classes is 2. When K=1, then the algorithm is known as the nearest neighbor algorithm

LINEAR SVC
The objective of a Linear SVC (Support Vector Classifier) is to fit to the data you provide, returning a "best fit" hyperplane that divides, or categorizes, your data. From there, after getting the hyperplane, you can then feed some features to your classifier to see what the "predicted" class is. This makes this specific algorithm rather suitable for our uses, though you can use this for many situations

Random Forest Classifier
Random forest classifier creates a set of decision trees from randomly selected subset of training set. It then aggregates the votes from different decision trees to decide the final class of the test object.
It works in four steps:
⦁	Select random samples from a given dataset.
⦁	Construct a decision tree for each sample and get a prediction result from each decision tree.
⦁	Perform a vote for each predicted result.
⦁	Select the prediction result with the most votes as the final prediction.

Decision tree regression 
observes features of an object and trains a model in the structure of a tree to predict data in the future to produce meaningful continuous output. Continuous output means that the output/result is not discrete, i.e., it is not represented just by a discrete, known set of numbers or values.


CLASSIFIERS USED:
CODE :
⦁	Diabetic Retinopathy using DNN
import tensorflow as tf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
import scikitplot as skplt
%matplotlib inline
from google.colab import files
uploaded = files.upload()

def Data_Process():
    
    columns_to_named = ["Pregnancies","Glucose","BloodPressure",
           "SkinThickness","Insulin","BMI","DiabetesPedigreeFunction",
           "Age","Class"]
    
    df = pd.read_csv("pima-indians-diabetes.csv",header=0,names=columns_to_named)
    col_norm =['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction']
       
    df1_norm = df[col_norm].apply(lambda x :( (x - x.min()) / (x.max()-x.min()) ) )
        
    X_Data = df1_norm
    Y_Data = df["Class"]
    
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_Data,Y_Data, test_size=0.3,random_state=101)
    
    return X_Train, X_Test, Y_Train, Y_Test

def create_feature_column():
    
    feat_Pregnancies = tf.feature_column.numeric_column('Pregnancies')
    feat_Glucose = tf.feature_column.numeric_column('Glucose')
    feat_BloodPressure = tf.feature_column.numeric_column('BloodPressure')
    feat_SkinThickness_tricep = tf.feature_column.numeric_column('SkinThickness')
    feat_Insulin = tf.feature_column.numeric_column('Insulin')
    feat_BMI = tf.feature_column.numeric_column('BMI')
    feat_DiabetesPedigreeFunction  = tf.feature_column.numeric_column('DiabetesPedigreeFunction')
    
  feature_column = [feat_Pregnancies, feat_Glucose, feat_BloodPressure,
                  feat_SkinThickness_tricep, feat_Insulin, 
                 feat_BMI , feat_DiabetesPedigreeFunction] 
    
    return feature_column
 
X_Train, X_Test, Y_Train, Y_Test = Data_Process()
feature_column = create_feature_column()

input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(X_Train, 
                                                 Y_Train,
                                                 batch_size=50,
                                                 num_epochs=1000,
                                                 shuffle=True)

eval_func = tf.compat.v1.estimator.inputs.pandas_input_fn(X_Test,
                                               Y_Test,
                                               batch_size=50,
                                               num_epochs=1,
                                               shuffle=False)
predict_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
      x=X_Test,
      num_epochs=1,
      shuffle=False)

dnnmodel = tf.estimator.DNNClassifier(
                                        hidden_units = [20,20],
                                        feature_columns = feature_column,
                                        n_classes=2,
                                        activation_fn=tf.nn.softmax,
                                        dropout=None,
                                        optimizer = tf.optimizers.Adam(learning_rate=0.01)
                                    )
history = dnnmodel.train(input_fn=input_func, 
               steps=500)
dnnmodel.evaluate(eval_func)

predictions = list(dnnmodel.predict(input_fn=predict_input_fn))
prediction = [p["class_ids"][0] for p in predictions]
data = classification_report(Y_Test,prediction)
conmat = confusion_matrix(Y_Test,prediction)

skplt.metrics.plot_confusion_matrix(Y_Test, 
                                    prediction,
                                   figsize=(6,6),
                                   title="Confusion Matrix")

Output :

 

⦁	Diabetic Retinopathy using different algorithms(LogisticRegression ,KNeighborsClassifier,GaussianNB,SVC,LinearSVC,RandomForestClassifier,
DecisionTreeRegressor)

# We import the libraries needed to read the dataset
import os
import pandas as pd
import numpy as np

# We placed the dataset under datasets/ sub folder
DATASET_PATH = 'datasets/'

# We read the data from the CSV file 
path = os.path.abspath(r'/home/jovyan/demo/pima-indians-diabetes.csv')
f = open(path)
print(f)
data_path = os.path.join('/home/jovyan/demo/', 'pima-indians-diabetes.csv')
dataset = pd.read_csv(data_path, header=None)

# Because thr CSV doesn't contain any header, we add column names 
# using the description from the original dataset website
dataset.columns = [
    "NumTimesPrg", "PlGlcConc", "BloodP",
    "SkinThick", "TwoHourSerIns", "BMI",
    "DiPedFunc", "Age", "HasDiabetes"]

# Check the shape of the data: we have 768 rows and 9 columns:
# the first 8 columns are features while the last one
# is the supervised label (1 = has diabetes, 0 = no diabetes)
dataset.shape

# Visualise a table with the first rows of the dataset, to
# better understand the data format
dataset.head()
corr = dataset.corr()
corr

%matplotlib inline
import seaborn as sns
sns.heatmap(corr, annot = True)

import matplotlib.pyplot as plt
dataset.hist(bins=50, figsize=(20, 15))
plt.show()

# Calculate the median value for BMI
median_bmi = dataset['BMI'].median()
# Substitute it in the BMI column of the
# dataset where values are 0
dataset['BMI'] = dataset['BMI'].replace(
    to_replace=0, value=median_bmi)

# Calculate the median value for BloodP
median_bloodp = dataset['BloodP'].median()
# Substitute it in the BloodP column of the
# dataset where values are 0
dataset['BloodP'] = dataset['BloodP'].replace(
    to_replace=0, value=median_bloodp)
# Calculate the median value for PlGlcConc
median_plglcconc = dataset['PlGlcConc'].median()
# Substitute it in the PlGlcConc column of the
# dataset where values are 0
dataset['PlGlcConc'] = dataset['PlGlcConc'].replace(
    to_replace=0, value=median_plglcconc)

# Calculate the median value for SkinThick
median_skinthick = dataset['SkinThick'].median()
# Substitute it in the SkinThick column of the
# dataset where values are 0
dataset['SkinThick'] = dataset['SkinThick'].replace(
    to_replace=0, value=median_skinthick)

# Calculate the median value for TwoHourSerIns
median_twohourserins = dataset['TwoHourSerIns'].median()
# Substitute it in the TwoHourSerIns column of the
# dataset where values are 0
dataset['TwoHourSerIns'] = dataset['TwoHourSerIns'].replace(
    to_replace=0, value=median_twohourserins)

# Split the training dataset in 80% / 20%
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(
    dataset, test_size=0.2, random_state=42)

# Separate labels from the rest of the dataset
train_set_labels = train_set["HasDiabetes"].copy()
train_set = train_set.drop("HasDiabetes", axis=1)

test_set_labels = test_set["HasDiabetes"].copy()
test_set = test_set.drop("HasDiabetes", axis=1)

# Apply a scaler
from sklearn.preprocessing import MinMaxScaler as Scaler

scaler = Scaler()
scaler.fit(train_set)
train_set_scaled = scaler.transform(train_set)
test_set_scaled = scaler.transform(test_set)

#scaled values
df = pd.DataFrame(data=train_set_scaled)
df.head()

#Select and train a model
#It's not possible to know in advance which algorithm will work better with our dataset. 
#We need to compare a few and select the one with the "best score".
#Comparing multiple algorithms

#To compare multiple algorithms with the same dataset, there is a very nice utility in sklearn called model_selection. 
#We create a list of algorithms and then we score them using the same comparison method. 
#At the end we pick the one with the best score.

# Import all the algorithms we want to test
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor

# Import the slearn utility to compare algorithms
from sklearn import model_selection

# Prepare an array with all the algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVC', SVC()))
models.append(('LSVC', LinearSVC()))
models.append(('RFC', RandomForestClassifier()))
models.append(('DTR', DecisionTreeRegressor()))

# Prepare the configuration to run the test
seed = 7
results = []
names = []
X = train_set_scaled
Y = train_set_labels

# Every algorithm is tested and results are
# collected and printed
for name, model in models:
    kfold = model_selection.KFold(
        n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(
        model, X, Y, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (
        name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#It looks like that using this comparison method, the most performant algorithm is SVC.
#Find the best parameters for SVC

#The default parameters for an algorithm are rarely the best ones for our dataset. 
#Using sklearn we can easily build a parameters grid and try all the possible combinations. 
#At the end we inspect the best_estimator_ property and get the best ones for our dataset.
from sklearn.model_selection import GridSearchCV
param_grid = {
    'C': [1.0, 10.0, 50.0],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'shrinking': [True, False],
    'gamma': ['auto', 1, 0.1],
    'coef0': [0.0, 0.1, 0.5]
}

model_svc = SVC()
grid_search = GridSearchCV(
    model_svc, param_grid, cv=10, scoring='accuracy')
grid_search.fit(train_set_scaled, train_set_labels)

# Print the bext score found
grid_search.best_score_

# Create an instance of the algorithm using parameters
# from best_estimator_ property
svc = grid_search.best_estimator_

# Use the whole dataset to train the model
X = np.append(train_set_scaled, test_set_scaled, axis=0)
Y = np.append(train_set_labels, test_set_labels, axis=0)

# Train the model
svc.fit(X, Y)

#Make a Prediction

# We create a new (fake) person having the three most correated values high
new_df = pd.DataFrame([[6, 168, 72, 35, 0, 43.6, 0.627, 65]])
# We scale those values like the others
new_df_scaled = scaler.transform(new_df)

# We predict the outcome
prediction = svc.predict(new_df_scaled)

# A value of "1" means that this person is likley to have type 2 diabetes
Prediction

Conclusion :

⦁	We finally find a score of 76% using SVC algorithm and parameters optimisation. 
⦁	Please note that there may be still space for further analysis and optimisation, 
⦁	for example trying different data transformations or trying algorithms that haven't been tested yet. 
⦁	Once again I want to repeat that training a machine learning model to solve a problem with a specific dataset is a try / fail / improve process.

Output Screenshots has been attached in document file. refer it for clear concepts.
