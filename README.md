# random forest Regression

# sklearn random forest Regressor code
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression 
X, y = make_regression(n_features=4, n_informative=2,random_state=0, shuffle=False)
regr = RandomForestRegressor(max_depth = 2, random_state=0,) #  # initilize random forest regressor # it' is the method in which backend defind the method
regr.fit(X, y) # fiting the data stored in variable x,y
print(regr.predict([[0, 0, 0, 0]]))

# Random Forest classifier
# sklearn random forest Classifier code
!pip install sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0,random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)
print(clf.predict([[0, 0, 0, 0]]))
