import pandas as pd

a='abcdefghijklmn'
n=[]
for i in a:
    n.append(i)
data=pd.read_csv(r"/boston.csv",names=n)
# print(data.shape)
# print(data.size)
# print(data.head())
# print(data.columns)
# print(data['a'].unique())
# print(data['a'].value_counts())
# print(data.describe())
# print(data.corr())

#print(data['b'].isna().value_counts())
#data=data.fillna(5)

x=data.iloc[ : , :-1].values
y=data.iloc[ : ,-1].values

# print(x)
# print(y)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=5)

# print(xtrain.shape)
# print(xtest.shape)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

from sklearn.metrics import mean_squared_error
import math

print(math.sqrt(mean_squared_error(ytest,ypred)))

print(model.predict([[1.1069e-01, 0.0000e+00, 1.3890e+01, 1.0000e+00, 5.5000e-01,
       5.9510e+00, 9.3800e+01, 2.8893e+00, 5.0000e+00, 2.7600e+02,
       1.6400e+01, 3.9690e+02, 1.7920e+01]]))
