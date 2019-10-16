import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import metrics
   
predict_WL = pd.read_csv("C:/DS/Machine Learning/tcd ml 2019-20 income prediction training (with labels)/tcd ml 2019-20 income prediction training (with labels).csv")
predict_WIL = pd.read_csv("C:/DS/Machine Learning/tcd ml 2019-20 income prediction test (without labels).csv")

#IMPUTATION OF CATEGORICAL VALUES
predict1 = pd.concat([predict_WL,predict_WIL],axis=0)   
predict1 = predict1.drop('Hair Color',axis=1)
predict1 = predict1.drop('Wears Glasses',axis=1)
predict1["Age"] = predict1["Age"].fillna(method='ffill')
predict1["Year of Record"] = predict1["Year of Record"].fillna(predict1["Year of Record"].mode().iloc[0])
predict1['Gender'] = predict1['Gender'].replace('0','other')
predict1['Gender'] = predict1['Gender'].replace('unknown','other')
predict1['Gender'] = predict1['Gender'].fillna(method='ffill')
predict1['Country'].isnull().any()
predict1['Size of City'].isnull().any()
predict1['University Degree'] = predict1['University Degree'].replace('0','No')
predict1['University Degree'] = predict1['University Degree'].fillna(method='ffill')
predict1['Profession'] = predict1['Profession'].replace('0','NA')
predict1['Profession'] = predict1['Profession'].fillna(method='ffill')
predict1.isnull().any()
    
#ONE HOT ENCODING
predict2 = pd.get_dummies(predict1, columns=['Year of Record','Gender','Country','Profession','University Degree'])
predict2 = predict2.drop('Profession_Armourer',axis=1)
predict2 = predict2.drop('University Degree_No',axis=1)
predict2 = predict2.drop('Country_Algeria',axis=1)
predict2.isnull().any()

#SCALING 
predict2['Size of City'] = np.log(predict2['Size of City'])
#predict2['Income(EUR)'] = predict2['Income(EUR)'].abs()
predict2['Income(EUR)'] = np.log(predict2['Income(EUR)'])
    

predict3 = predict2.copy()

#REMOVING OUTLIERS
predict2 = predict2.sort_values(by='Size of City', ascending=False)
predict2 = predict2.iloc[100:]
predict2 = predict2.sort_values(by='Income(EUR)', ascending=False)
predict2 = predict2.iloc[100:]

predict2.dropna(subset=['Income(EUR)'],inplace = True)

#SPLITTING USING K-FOLD
X = np.array(predict2.drop('Income(EUR)',axis=1))
y = np.array(predict2['Income(EUR)'])

from sklearn.model_selection import KFold
kfold = KFold(n_splits=100, shuffle=True, random_state=42)
scores = []
#
for train_index, test_index in kfold.split(predict2):   
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

#APPLYING LINEAR REGRESSION MODEL
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

#CALCULATING ROOT MEAN SQUARE
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(np.exp(y_test),np.exp(y_pred))))   

####WORK on VALIDATION SHEET
predict4 = predict3.copy()
predict4=predict4.sort_values('Instance')
predict4.drop('Income(EUR)',axis=1,inplace=True)
predict4 = predict4.iloc[111993:]
y_pred_read = regressor.predict(predict4)
y_pred_read = np.exp(y_pred_read)
predict_tests = pd.read_csv("C:/DS/Machine Learning/tcd ml 2019-20 income prediction submission file.csv")
predict_tests['Income'] = pd.DataFrame(y_pred_read).iloc[:,-1]
predict_tests.to_csv("C:/DS/Machine Learning/Result/tcd ml 2019-20 income prediction submission file.csv", index = False)