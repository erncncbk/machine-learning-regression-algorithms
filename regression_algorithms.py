# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:45:06 2018

@author: Erencan
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv('day.csv')

season      =   df.season.values.reshape(-1,1)
year        =   df.yr.values.reshape(-1,1)
month       =   df.mnth.values.reshape(-1,1)
holiday     =   df.holiday.values.reshape(-1,1)
weekday     =   df.weekday.values.reshape(-1,1)
workingday  =   df.workingday.values.reshape(-1,1)
weathersit  =   df.weathersit.values.reshape(-1,1)

temp        =   df.temp.values.reshape(-1,1)
atemp       =   df.atemp.values.reshape(-1,1)
hum         =   df.hum.values.reshape(-1,1)
windspeed   =   df.windspeed.values.reshape(-1,1)
casual      =   df.casual.values.reshape(-1,1)
registered  =   df.registered.values.reshape(-1,1)
cnt         =   df.cnt.values.reshape(-1,1)




#%%
# Label encoder: Text datalarını 0 1 2 diye adlandırır ama 2>1>0 gibi problem yaratır
# bu yüzden bir one hot encoder yaparız. Dolayısıyla 0 1 2 varsa datada no need for label encoder
# Onehot yapıyoruz boylece elimizdeki sıralanmıs datayı içindeki sıraya gore kolum yapıyor

from sklearn.preprocessing import OneHotEncoder
onehotencoder =     OneHotEncoder(categorical_features=[0])
season        =     onehotencoder.fit_transform(season).toarray()
year          =     onehotencoder.fit_transform(year).toarray()
month         =     onehotencoder.fit_transform(month).toarray()
holiday       =     onehotencoder.fit_transform(holiday).toarray()
weekday       =     onehotencoder.fit_transform(weekday).toarray()
workingday    =     onehotencoder.fit_transform(workingday).toarray()
weathersit    =     onehotencoder.fit_transform(weathersit).toarray()

#%% Concatenate
# Birleştirme işlemleri yapılıyor. (10,11,12,13) Tüm featurelar. (dateday dahil degil)
df  = np.concatenate((season,year,month,holiday,weekday,workingday,weathersit,temp,atemp,hum,windspeed), axis = 1)

#%% MinMaxScaler

from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler(feature_range=(0,1))
df     = minmax.fit_transform(df)

#%% Test Train

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df,registered,test_size=0.2,random_state=0)

#%%  Test datalarımızı predict etme
#         Multiple linear Regression
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(x_train,y_train)
y_pred = linear_regression.predict(x_test)
#print(y_pred)
accuracy = linear_regression.score(x_test, y_test)
print("accuracy: ",accuracy)
plt.scatter(y_test,y_pred,color='g',label="linear")
plt.legend()
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


#%% R square

from sklearn.metrics import r2_score

print("r_score",r2_score(y_test,y_pred))
#%%
          #Polynomial Linear Regression
          
from sklearn.preprocessing import PolynomialFeatures
polynomial_regression =  PolynomialFeatures(degree= 2)
X_poly = polynomial_regression.fit_transform(x_train)
X_test = polynomial_regression.fit_transform(x_test)
polynomial_regression.fit(X_poly,y_train)
linear_regression_2=LinearRegression()
linear_regression_2.fit(X_poly,y_train)
y_pred2=linear_regression_2.predict(polynomial_regression.fit_transform(x_test))
accuracy = linear_regression_2.score(X_test,y_test)
print("accuracy: ",accuracy)
plt.scatter(y_test,y_pred2,color="r",label="polynomial")
plt.legend()
plt.xlabel("y_test")
plt.ylabel("y_pred2")
plt.show()
print("r_score",r2_score(y_test,y_pred2))

#%%
##           Decision Tree
          
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit (x_train,y_train)
y_pred3 = dt.predict(x_test)
accuracy = dt.score(x_test,y_test)
print("accuracy: ",accuracy)
print("r_score",r2_score(y_test,y_pred3))
plt.scatter(y_test,y_pred3,label="decision three")
plt.legend()
plt.xlabel("y_test")
plt.ylabel("y_pred3")
plt.show()

#%% 
##          Random Forest Regression
#          
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=130,random_state=0)
rf.fit(x_train,y_train)
y_pred4 = rf.predict(x_test)
accuracy = rf.score(x_test,y_test)
print("accuracy: ",accuracy)
print("r_score",r2_score(y_test,y_pred4))
plt.scatter(y_test,y_pred4,color='r',label="random forest")
plt.legend()
plt.xlabel("y_test")
plt.ylabel("y_pred4")
plt.show()
