# random forest Regression

# sklearn random forest Regressor code
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression 
X, y = make_regression(n_features=4, n_informative=2,random_state=0, shuffle=False)
regr = RandomForestRegressor(max_depth = 2, random_state=0,) #  # initilize random forest regressor # it' is the method in which backend defind the method
regr.fit(X, y) # fiting the data stored in variable x,y
print(regr.predict([[0, 0, 0, 0]]))

# Random Forest classifier
