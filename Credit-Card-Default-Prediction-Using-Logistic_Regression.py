import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import figure

os.chdir("C:/Users/dipti/Documents/IMR/Data_Logistic_Regression/")

fullRaw = pd.read_csv('BankCreditCard.csv')


############################
# Sampling: Divide the data into Train and Testset
############################

from sklearn.model_selection import train_test_split
trainDf, testDf = train_test_split(fullRaw, train_size=0.7, random_state = 123)


# Create Source Column in both Train and Test
trainDf['Source'] = 'Train'
testDf['Source'] = 'Test'

# Combine Train and Test
fullRaw = pd.concat([trainDf, testDf], axis = 0)
fullRaw.shape

# Check for NAs
fullRaw.isnull().sum()

# % Split of 0s and 1s
fullRaw.loc[fullRaw['Source'] == 'Train', 'Default_Payment'].value_counts()/ fullRaw[fullRaw['Source'] == 'Train'].shape[0]


# Summarize the data
fullRaw_Summary = fullRaw.describe()
# fullRaw_Summary = fullRaw.describe(include = "all")

# Lets drop 'Customer ID' column from the data as it is not going to assist us in our model
fullRaw.drop(['Customer ID'], axis = 1, inplace = True) 
fullRaw.shape




############################
# Use data description excel sheet to convert numeric variables to categorical variables
############################


variableToUpdate = 'Gender'
fullRaw[variableToUpdate].value_counts() 
fullRaw[variableToUpdate].replace({1:"Male", 
                                     2:"Female"}, inplace = True)
fullRaw[variableToUpdate].value_counts()


variableToUpdate = 'Academic_Qualification'
fullRaw[variableToUpdate].value_counts()
fullRaw[variableToUpdate].replace({1:"Undergraduate",
                                     2:"Graduate",
                                     3:"Postgraduate",
                                     4:"Professional",
                                     5:"Others",
                                     6:"Unknown"}, inplace = True)
fullRaw[variableToUpdate].value_counts()


variableToUpdate = 'Marital' # 1=Married, 2=Single, 3=Do not prefer to say/ Unknown
fullRaw[variableToUpdate].value_counts()
fullRaw[variableToUpdate].replace({1:"Married",
                                     2:"Single",
                                     3:"Unknown", # Do not prefer to say
                                     0:"Unknown"}, inplace = True)
fullRaw[variableToUpdate].value_counts()

############################
# Bivariate Analysis (For Continuous Indep vars) Using Boxplot
############################

trainDf = fullRaw.loc[fullRaw['Source'] == 'Train']
continuousVars = trainDf.columns[trainDf.dtypes != object]
continuousVars

fileName = "C:/Users/dipti/Documents/IMR/Data_Logistic_Regression/Continuous_Bivariate_Analysis.pdf"
pdf = PdfPages(fileName)
for colNumber, colName in enumerate(continuousVars): # enumerate gives key, value pair
    # print(colNumber, colName)
    figure()
    sns.boxplot(y = trainDf[colName], x = trainDf["Default_Payment"])
    pdf.savefig(colNumber+1) # colNumber+1 is done to ensure page numbering starts from 1 (and NOT 0)
    
pdf.close()


############################
# Bivariate Analysis (For Categorical Indep vars) Using Histogram
############################

categoricalVars = trainDf.columns[trainDf.dtypes == object]
categoricalVars

sns.histplot(trainDf, x="Gender", hue="Default_Payment", stat="probability", multiple="fill")

fileName = "C:/Users/dipti/Documents/IMR/Data_Logistic_Regression/Categorical_Bivariate_Analysis.pdf"
pdf = PdfPages(fileName)
for colNumber, colName in enumerate(categoricalVars): # enumerate gives key, value pair
    
    figure()
    sns.histplot(trainDf, x=colName, hue="Default_Payment", stat="probability", multiple="fill")
    
    pdf.savefig(colNumber+1)
    
pdf.close()





############################
# Dummy variable creation
############################

fullRaw2 = pd.get_dummies(fullRaw, drop_first = True) #
fullRaw2.shape

fullRaw.shape
fullRaw2.dtypes

############################
# Divide the data into Train and Test
############################


# Step 1: Divide into Train and Testest
Train = fullRaw2[fullRaw2['Source_Train'] == 1].drop(['Source_Train'], axis = 1).copy()
Test = fullRaw2[fullRaw2['Source_Train'] == 0].drop(['Source_Train'], axis = 1).copy()


# Step 2: Divide into Xs (Indepenedents) and Y (Dependent)
depVar = "Default_Payment"
trainX = Train.drop([depVar], axis = 1).copy()
trainY = Train[depVar].copy()
testX = Test.drop([depVar], axis = 1).copy()
testY = Test[depVar].copy()

trainX.shape
testX.shape

############################
# Add Intercept Column
############################


from statsmodels.api import add_constant
trainX = add_constant(trainX)
testX = add_constant(testX)

trainX.shape
testX.shape

#########################
# VIF check
#########################
from statsmodels.stats.outliers_influence import variance_inflation_factor

tempMaxVIF = 10 
maxVIF = 10
trainXCopy = trainX.copy()
counter = 1
highVIFColumnNames = []

while (tempMaxVIF >= maxVIF):
    
    print(counter)
    
    # Create an empty temporary df to store VIF values
    tempVIFDf = pd.DataFrame()
    
    # Calculate VIF using list comprehension
    tempVIFDf['VIF'] = [variance_inflation_factor(trainXCopy.values, i) for i in range(trainXCopy.shape[1])]
    
    # Create a new column "Column_Name" to store the col names against the VIF values from list comprehension
    tempVIFDf['Column_Name'] = trainXCopy.columns
    
    # Drop NA rows from the df - If there is some calculation error resulting in NAs
    tempVIFDf.dropna(inplace=True)
    
    # Sort the df based on VIF values, then pick the top most column name (which has the highest VIF)
    tempColumnName = tempVIFDf.sort_values(["VIF"], ascending = False).iloc[0,1]
    
    # Store the max VIF value in tempMaxVIF
    tempMaxVIF = tempVIFDf.sort_values(["VIF"], ascending = False).iloc[0,0]
    
    if (tempMaxVIF >= maxVIF): 
       
        trainXCopy = trainXCopy.drop(tempColumnName, axis = 1)    
        highVIFColumnNames.append(tempColumnName)
        print(tempColumnName)
    
    counter = counter + 1

highVIFColumnNames


highVIFColumnNames.remove('const') 

trainX = trainX.drop(highVIFColumnNames, axis = 1)
testX = testX.drop(highVIFColumnNames, axis = 1)


trainX.shape
testX.shape



########################
# Model building
########################

# Build logistic regression model (using statsmodels package/library)
from statsmodels.api import Logit  
M1 = Logit(trainY, trainX) 
M1_Model = M1.fit()
M1_Model.summary() 


########################
# Manual model selection. Drop the most insignificant variable in model one by one and recreate the model
########################


# Drop Academic_Qualification_Postgraduate
Cols_To_Drop = ["Academic_Qualification_Postgraduate"]
M2 = Logit(trainY, trainX.drop(Cols_To_Drop, axis = 1)).fit() # (Dep_Var, Indep_Vars)
M2.summary()


# Drop Marital_Unknown
Cols_To_Drop.append('Marital_Unknown')
M3 = Logit(trainY, trainX.drop(Cols_To_Drop, axis = 1)).fit() # (Dep_Var, Indep_Vars)
M3.summary()

# Drop April_Bill_Amount
Cols_To_Drop.append('April_Bill_Amount')
M5 = Logit(trainY, trainX.drop(Cols_To_Drop, axis = 1)).fit() # (Dep_Var, Indep_Vars)
M5.summary()

# Drop June_Bill_Amount
Cols_To_Drop.append('June_Bill_Amount')
M6 = Logit(trainY, trainX.drop(Cols_To_Drop, axis = 1)).fit() # (Dep_Var, Indep_Vars)
M6.summary()

# Drop Age_Years
Cols_To_Drop.append('Age_Years')
M7 = Logit(trainY, trainX.drop(Cols_To_Drop, axis = 1)).fit() # (Dep_Var, Indep_Vars)
M7.summary()

# Drop Repayment_Status_Feb
Cols_To_Drop.append('Repayment_Status_Feb')
M8 = Logit(trainY, trainX.drop(Cols_To_Drop, axis = 1)).fit() # (Dep_Var, Indep_Vars)
M8.summary()

# Drop Academic_Qualification_Unknown
Cols_To_Drop.append('Academic_Qualification_Unknown')
M9 = Logit(trainY, trainX.drop(Cols_To_Drop, axis = 1)).fit() # (Dep_Var, Indep_Vars)
M9.summary()

# Drop Academic_Qualification_Undergraduate
Cols_To_Drop.append('Academic_Qualification_Undergraduate')
M10 = Logit(trainY, trainX.drop(Cols_To_Drop, axis = 1)).fit() # (Dep_Var, Indep_Vars)
M10.summary()

# Drop Previous_Payment_May
Cols_To_Drop.append('Previous_Payment_May')
M11 = Logit(trainY, trainX.drop(Cols_To_Drop, axis = 1)).fit() # (Dep_Var, Indep_Vars)
M11.summary()

# Drop Repayment_Status_April
Cols_To_Drop.append('Repayment_Status_April')
M11 = Logit(trainY, trainX.drop(Cols_To_Drop, axis = 1)).fit() # (Dep_Var, Indep_Vars)
M11.summary()

trainX = trainX.drop(Cols_To_Drop, axis = 1)
testX = testX.drop(Cols_To_Drop, axis = 1) 

trainX.shape
testX.shape

############################
# Prediction and Validation
############################

testX['Test_Prob'] = M11.predict(testX) 
testX.columns 
testX['Test_Prob'][0:6]
testY[:6]

# Classify 0 or 1 based on 0.5 cutoff
testX['Test_Class'] = np.where(testX['Test_Prob'] >= 0.5, 1, 0)
testX.columns 

########################
# Confusion matrix
########################

Confusion_Mat = pd.crosstab(testX['Test_Class'], testY) 
Confusion_Mat

# Check the accuracy of the model
(sum(np.diagonal(Confusion_Mat))/testX.shape[0])*100 # ~82%

########################
# Precision, Recall, F1 Score
########################

from sklearn.metrics import classification_report
print(classification_report(testY, testX['Test_Class'])) 



########################
# AUC and ROC Curve
########################

from sklearn.metrics import roc_curve, auc
# Predict on train data
Train_Prob = M11.predict(trainX)

# Calculate FPR, TPR and Cutoff Thresholds
fpr, tpr, cutoff = roc_curve(trainY, Train_Prob)

# Cutoff Table Creation
Cutoff_Table = pd.DataFrame()
Cutoff_Table['FPR'] = fpr 
Cutoff_Table['TPR'] = tpr
Cutoff_Table['Cutoff'] = cutoff

# Plot ROC Curve
import seaborn as sns
sns.lineplot(Cutoff_Table['FPR'], Cutoff_Table['TPR'])

# Area under curve (AUC)
auc(fpr, tpr) 









############################
# Improve Model Output Using New Cutoff Point
############################

import numpy as np
Cutoff_Table['DiffBetweenTPRFPR'] = Cutoff_Table['TPR'] - Cutoff_Table['FPR'] # Max Diff. Bet. TPR & FPR
Cutoff_Table['Distance'] = np.sqrt((1-Cutoff_Table['TPR'])**2 + (0-Cutoff_Table['FPR'])**2) # Euclidean Distance


# New Cutoff Point Performance (Obtained after studying ROC Curve and Cutoff Table)
cutoffPoint = 0.202342 # Max Difference between TPR & FPR

# Classify the test predictions into classes of 0s and 1s
testX['Test_Class2'] = np.where(testX['Test_Prob'] >= cutoffPoint, 1, 0)

# Confusion Matrix
Confusion_Mat2 = pd.crosstab(testX['Test_Class2'], testY) # R, C format
Confusion_Mat2

# Model Evaluation Metrics
print(classification_report(testY, testX['Test_Class2']))
# Recall of 1s has almost doubled from 0.34 to 0.59







