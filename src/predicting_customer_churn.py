import numpy as np
import pandas as pd

# For visualization
import matplotlib.pyplot as plt

import seaborn as sns
pd.options.display.max_rows = None
pd.options.display.max_columns = None
# Read the data frame
df = pd.read_csv('C:/Users/dageb/Documents/Churn_Modelling.csv', delimiter=',')
# df.shape
# print(df.shape)
# df.isnull().sum()
# print(df.isnull().sum())
# df.nunique()
# print(df.nunique())

# df = df.drop(["RowNumber", "CustomerId", "Surname"], axis = 1)
# df.head()
# print(df.head())
# df.dtypes
# print(df.dtypes)
# labels = 'Exited', 'Retained'
# sizes = [df.Exited[df['Exited']==1].count(), df.Exited[df['Exited']==0].count()]
# explode = (0, 0.1)
# fig1, ax1 = plt.subplots(figsize=(10, 8))
# ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
#         shadow=True, startangle=90)
# ax1.axis('equal')
# plt.title("Proportion of customer churned and retained", size = 20)
# plt.show()

# fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
# sns.countplot(x='Geography', hue = 'Exited',data = df, ax=axarr[0][0])
# sns.countplot(x='Gender', hue = 'Exited',data = df, ax=axarr[0][1])
# sns.countplot(x='HasCrCard', hue = 'Exited',data = df, ax=axarr[1][0])
# sns.countplot(x='IsActiveMember', hue = 'Exited',data = df, ax=axarr[1][1])
# plt.show()

# fig, axarr = plt.subplots(3, 2, figsize=(20, 12))
# sns.boxplot(y='CreditScore',x = 'Exited', hue = 'Exited',data = df, ax=axarr[0][0])
# sns.boxplot(y='Age',x = 'Exited', hue = 'Exited',data = df , ax=axarr[0][1])
# sns.boxplot(y='Tenure',x = 'Exited', hue = 'Exited',data = df, ax=axarr[1][0])
# sns.boxplot(y='Balance',x = 'Exited', hue = 'Exited',data = df, ax=axarr[1][1])
# sns.boxplot(y='NumOfProducts',x = 'Exited', hue = 'Exited',data = df, ax=axarr[2][0])
# sns.boxplot(y='EstimatedSalary',x = 'Exited', hue = 'Exited',data = df, ax=axarr[2][1])
# plt.show()

df_train = df.sample(frac=0.8,random_state=200)
df_test = df.drop(df_train.index)
# print(len(df_train))
# print(len(df_test))

df_train['BalanceSalaryRatio'] = df_train.Balance/df_train.EstimatedSalary
# sns.boxplot(y='BalanceSalaryRatio',x = 'Exited', hue = 'Exited',data = df_train)
# plt.ylim(-1, 5)
# # plt.show()

df_train['TenureByAge'] = df_train.Tenure/(df_train.Age)
# sns.boxplot(y='TenureByAge',x = 'Exited', hue = 'Exited',data = df_train)
# plt.ylim(-1, 1)
# # plt.show()

df_train['CreditScoreGivenAge'] = df_train.CreditScore/(df_train.Age)
# df_train.head()
# # print(df_train.head())

continuous_vars = ['CreditScore',  'Age', 'Tenure', 'Balance','NumOfProducts', 'EstimatedSalary', 'BalanceSalaryRatio',
                   'TenureByAge','CreditScoreGivenAge']
cat_vars = ['HasCrCard', 'IsActiveMember','Geography', 'Gender']
df_train = df_train[['Exited'] + continuous_vars + cat_vars]
# print(df_train.head())

df_train.loc[df_train.HasCrCard == 0, 'HasCrCard'] = -1
df_train.loc[df_train.IsActiveMember == 0, 'IsActiveMember'] = -1
# df_train.head()

lst = ['Geography', 'Gender']
remove = list()
for i in lst:
    if (df_train[i].dtype == np.str or df_train[i].dtype == np.object):
        for j in df_train[i].unique():
            df_train[i+'_'+j] = np.where(df_train[i] == j,1,-1)
        remove.append(i)
df_train = df_train.drop(remove, axis=1)

# # print(df_train.head())

minVec = df_train[continuous_vars].min().copy()
maxVec = df_train[continuous_vars].max().copy()
df_train[continuous_vars] = (df_train[continuous_vars]-minVec)/(maxVec-minVec)
df_train.head()
# # print(df_train.head())

def DfPrepPipeline(df_predict,df_train_Cols,minVec,maxVec):
    # Add new features
    df_predict['BalanceSalaryRatio'] = df_predict.Balance/df_predict.EstimatedSalary
    df_predict['TenureByAge'] = df_predict.Tenure/(df_predict.Age - 18)
    df_predict['CreditScoreGivenAge'] = df_predict.CreditScore/(df_predict.Age - 18)
    # Reorder the columns
    continuous_vars = ['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary','BalanceSalaryRatio',
                   'TenureByAge','CreditScoreGivenAge']
    cat_vars = ['HasCrCard','IsActiveMember',"Geography", "Gender"] 
    df_predict = df_predict[['Exited'] + continuous_vars + cat_vars]
    # Change the 0 in categorical variables to -1
    df_predict.loc[df_predict.HasCrCard == 0, 'HasCrCard'] = -1
    df_predict.loc[df_predict.IsActiveMember == 0, 'IsActiveMember'] = -1
    # One hot encode the categorical variables
    lst = ["Geography", "Gender"]
    remove = list()
    for i in lst:
        for j in df_predict[i].unique():
            df_predict[i+'_'+j] = np.where(df_predict[i] == j,1,-1)
        remove.append(i)
    df_predict = df_predict.drop(remove, axis=1)
    # Ensure that all one hot encoded variables that appear in the train data appear in the subsequent data
    L = list(set(df_train_Cols) - set(df_predict.columns))
    for l in L:
        df_predict[str(l)] = -1        
    # MinMax scaling coontinuous variables based on min and max from the train data
    df_predict[continuous_vars] = (df_predict[continuous_vars]-minVec)/(maxVec-minVec)
    # Ensure that The variables are ordered in the same way as was ordered in the train set
    df_predict = df_predict[df_train_Cols]
    return df_predict



from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform

# models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# Scoring model functions
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def best_model(model):
    print(model.best_score_)    
    print(model.best_params_)
    print(model.best_estimator_)
def get_auc_scores(y_actual, method,method2):
    auc_score = roc_auc_score(y_actual, method); 
    fpr_df, tpr_df, _ = roc_curve(y_actual, method2); 
    return (auc_score, fpr_df, tpr_df)

# # logistic regression
# param_grid = {'C': [0.1,0.5,1,10,50,100], 'max_iter': [250], 'fit_intercept':[True],'intercept_scaling':[1],
#               'penalty':['l2'], 'tol':[0.00001,0.0001,0.000001]}
# log_primal_Grid = GridSearchCV(LogisticRegression(solver='lbfgs'),param_grid, cv=10, refit=True, verbose=0)
# log_primal_Grid.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)
# best_model(log_primal_Grid)


# # logistic regression with degree 2 polynomial kernel
# param_grid = {'C': [0.1,10,50], 'max_iter': [300,500], 'fit_intercept':[True],'intercept_scaling':[1],'penalty':['l2'],
#               'tol':[0.0001,0.000001]}
# poly2 = PolynomialFeatures(degree=2)
# df_train_pol2 = poly2.fit_transform(df_train.loc[:, df_train.columns != 'Exited'])
# log_pol2_Grid = GridSearchCV(LogisticRegression(solver = 'liblinear'),param_grid, cv=5, refit=True, verbose=0)
# log_pol2_Grid.fit(df_train_pol2,df_train.Exited)
# best_model(log_pol2_Grid)


# # Fit SVM with RBF Kernel
# param_grid = {'C': [0.5,100,150], 'gamma': [0.1,0.01,0.001],'probability':[True],'kernel': ['rbf']}
# SVM_grid = GridSearchCV(SVC(), param_grid, cv=3, refit=True, verbose=0)
# SVM_grid.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)
# best_model(SVM_grid)


# # Fit SVM with pol kernel
# param_grid = {'C': [0.5,1,10,50,100], 'gamma': [0.1,0.01,0.001],'probability':[True],'kernel': ['poly'],'degree':[2,3] }
# SVM_grid = GridSearchCV(SVC(), param_grid, cv=3, refit=True, verbose=0)
# SVM_grid.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)
# best_model(SVM_grid)


# # Fit random forest classifier
# param_grid = {'max_depth': [3, 5, 6, 7, 8], 'max_features': [2,4,6,7,8,9],'n_estimators':[50,100],'min_samples_split': [3, 5, 6, 7]}
# RanFor_grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, refit=True, verbose=0)
# RanFor_grid.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)
# best_model(RanFor_grid)



# # Fit logistic regression
# log_primal = LogisticRegression(C=100, fit_intercept=True, intercept_scaling=1, max_iter=250, penalty='l2', tol=1e-05)
# log_primal.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)


# # Fit logistic regression with pol 2 kernel
# poly2 = PolynomialFeatures(degree=2)
# df_train_pol2 = poly2.fit_transform(df_train.loc[:, df_train.columns != 'Exited'])
# log_pol2 = LogisticRegression(C=50, fit_intercept=True, intercept_scaling=1, max_iter=300, penalty='l2', tol=0.0001)
# log_pol2.fit(df_train_pol2,df_train.Exited)

# # Fit svm_RBF
# SVM_RBF = SVC(C=100, gamma=0.1, kernel='rbf', probability=True)
# SVM_RBF.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)

# # Fit svm_POL
# SVM_POL = SVC(C=100, degree=2, gamma=0.1, kernel='poly', probability=True)
# SVM_POL.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)

# # Fit svm_RF
RF = RandomForestClassifier(max_depth=8, max_features=7, min_samples_split=5, n_estimators=50)
RF.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)


# print(classification_report(df_train.Exited, log_primal.predict(df_train.loc[:, df_train.columns != 'Exited'])))

# print(classification_report(df_train.Exited,  log_pol2.predict(df_train_pol2)))

# print(classification_report(df_train.Exited,  SVM_RBF.predict(df_train.loc[:, df_train.columns != 'Exited'])))

# print(classification_report(df_train.Exited,  SVM_POL.predict(df_train.loc[:, df_train.columns != 'Exited'])))

# print(classification_report(df_train.Exited,  RF.predict(df_train.loc[:, df_train.columns != 'Exited'])))


# y = df_train.Exited
# X = df_train.loc[:, df_train.columns != 'Exited']
# X_pol2 = df_train_pol2
# auc_log_primal, fpr_log_primal, tpr_log_primal = get_auc_scores(y, log_primal.predict(X),log_primal.predict_proba(X)[:,1])
# auc_log_pol2, fpr_log_pol2, tpr_log_pol2 = get_auc_scores(y, log_pol2.predict(X_pol2),log_pol2.predict_proba(X_pol2)[:,1])
# auc_SVM_RBF, fpr_SVM_RBF, tpr_SVM_RBF = get_auc_scores(y, SVM_RBF.predict(X),SVM_RBF.predict_proba(X)[:,1])
# auc_SVM_POL, fpr_SVM_POL, tpr_SVM_POL = get_auc_scores(y, SVM_POL.predict(X),SVM_POL.predict_proba(X)[:,1])
# auc_RF, fpr_RF, tpr_RF = get_auc_scores(y, RF.predict(X),RF.predict_proba(X)[:,1])


# plt.figure(figsize = (12,6), linewidth= 1)
# plt.plot(fpr_log_primal, tpr_log_primal, label = 'log primal Score: ' + str(round(auc_log_primal, 5)))
# plt.plot(fpr_log_pol2, tpr_log_pol2, label = 'log pol2 score: ' + str(round(auc_log_pol2, 5)))
# plt.plot(fpr_SVM_RBF, tpr_SVM_RBF, label = 'SVM RBF Score: ' + str(round(auc_SVM_RBF, 5)))
# plt.plot(fpr_SVM_POL, tpr_SVM_POL, label = 'SVM POL Score: ' + str(round(auc_SVM_POL, 5)))
# plt.plot(fpr_RF, tpr_RF, label = 'RF score: ' + str(round(auc_RF, 5)))
# plt.plot([0,1], [0,1], 'k--', label = 'Random: 0.5')
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC Curve')
# plt.legend(loc='best')
# plt.show()




df_test = DfPrepPipeline(df_test,df_train.columns,minVec,maxVec)
df_test = df_test.mask(np.isinf(df_test))
df_test = df_test.dropna()
# df_test.shape

auc_RF_test, fpr_RF_test, tpr_RF_test = get_auc_scores(df_test.Exited, RF.predict(df_test.loc[:, df_test.columns != 'Exited']),
                                                       RF.predict_proba(df_test.loc[:, df_test.columns != 'Exited'])[:,1])
plt.figure(figsize = (12,6), linewidth= 1)
plt.plot(fpr_RF_test, tpr_RF_test, label = 'RF score: ' + str(round(auc_RF_test, 5)))
plt.plot([0,1], [0,1], 'k--', label = 'Random: 0.5')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()