#Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures,LabelEncoder,OneHotEncoder
from sklearn.pipeline import Pipeline

#Reading Excel file
df = pd.read_excel('Watson_data.xlsx',sheetname='Data')
df.columns

#Dropping unnecessary columns
df = df.drop(['EmployeeCount','EmployeeNumber','Over18','StandardHours'],axis=1)
df.columns

#Encoding the bilateral categorical features
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})
df['Gender'] = df['Gender'].map({'Male':1,'Female':0})

#Modelling Linear Regression
regr = linear_model.LinearRegression()

def fit_linear_model_and_return_r_2_square(X_name,Y_name):
    X = df[[X_name]]
    Y = df[[Y_name]]
    regr.fit(X,Y)
    return regr.score(X,Y)

UniFeatures = df.columns
UniFeatures = list(UniFeatures)

#Removing unnecessary features
removefeatures = ['Department','BusinessTravel','EducationField','EnvironmentSatisfaction','JobRole','MaritalStatus','MonthlyIncome']
for i in removefeatures:
    UniFeatures.remove(i)

#Computing R-Square values
R_sqaure_of_UniFeatures = []
for i in UniFeatures:
    R_sqaure_of_UniFeatures.append([i,fit_linear_model_and_return_r_2_square(i,'MonthlyIncome')])
sorted(R_sqaure_of_UniFeatures,key=lambda x:x[1],reverse=True)

def fit_poly_model_and_return_r_2_square(X_name,Y_name,degree):
    poly = PolynomialFeatures(degree)
    X = df[[X_name]]
    Y = df[[Y_name]]    
    X = poly.fit_transform(X)
    regr.fit(X,Y)
    return regr.score(X,Y)

#R-Square
R_sqaure_of_UniFeatures = []
for i in UniFeatures:
    R_sqaure_of_UniFeatures.append([i,fit_poly_model_and_return_r_2_square(i,'MonthlyIncome',2)])
sorted(R_sqaure_of_UniFeatures,key=lambda x:x[1],reverse=True)

categorical_features = ['Department','BusinessTravel','EducationField','EnvironmentSatisfaction','JobRole','MaritalStatus']
for i in categorical_features:
    X = pd.get_dummies(df[[i]])
    regr.fit(X,df[['MonthlyIncome']])
    r2 = regr.score(X,df[['MonthlyIncome']])
    print (i,'R square Score',r2)

#Regression with jobrole categorical values
multi_features = df.ix[:,['JobLevel']]#,'TotalWorkingYears','YearsAtCompany','Age','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']]
multi_features = multi_features.join(pd.get_dummies(df[['JobRole']]))
print (multi_features.columns)
Y = df['MonthlyIncome']
multi_regr = linear_model.LinearRegression()
multi_regr.fit(multi_features,Y)
print(multi_regr.coef_,multi_regr.intercept_)
print ("R sqaure score of multivariant",multi_regr.score(multi_features,Y))

#Generating a heatmap of correlation
import seaborn as sns
All_high_r2_features = df.ix[:, ['TotalWorkingYears','JobLevel','YearsAtCompany','Age','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']]
correlation_matrix = All_high_r2_features.corr('pearson')
sns.heatmap(correlation_matrix,annot=True,)

#Plotting
import matplotlib.pyplot as plt
def plot_regr(X_name,Y_name):
    X = df[[X_name]]
    Y = df[[Y_name]]
    regr.fit(X,Y)
    Y_pred = regr.predict(X)
    plt.plot(X,Y_pred,'r')
    plt.plot(X,Y,'bo',alpha=0.3)
    plt.xlabel(X_name)
    plt.ylabel(Y_name)

#matplotlib inline
#plot_regr('TotalWorkingYears','MonthlyIncome')
#plot_regr('YearsAtCompany','MonthlyIncome')
plot_regr('NumCompaniesWorked','MonthlyIncome')

#Tweaking parameters, plotting again
import matplotlib.pyplot as plt
def plot_ploy_regr(X_name,Y_name,degree):
    poly = PolynomialFeatures(degree)
    X_o = df[[X_name]]
    Y = df[[Y_name]]    
    X = poly.fit_transform(X_o)
    regr.fit(X,Y)
    Y_pred = regr.predict(X)
    aa = sns.lmplot(X_name,Y_name,df,order=2,line_kws={'color': 'red'})
    
plot_ploy_regr('YearsSinceLastPromotion','MonthlyIncome',2)

#Generating covariance figure
#sns.set(rc={'figure.figsize':(10,9)})
ax = sns.stripplot(x='JobRole',y='MonthlyIncome',data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.figure.savefig('temp.png')

