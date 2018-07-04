#!/usr/bin/python

"""
    Starter code for the regression mini-project.

    Loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).

    Draws a little scatterplot of the training/testing data

    You fill in the regression code where indicated:
"""

import sys
import pickle
import numpy
from sklearn.metrics import accuracy_score
sys.path.append("../outliers/")
from outlier_cleaner import outlierCleaner
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

dictionary = pickle.load(open("../final_project/final_project_dataset_modified.pkl", "r"))

### list the features you want to look at--first item in the
### list will be the "target" feature
features_list = ["bonus", "salary"]
data = featureFormat(dictionary, features_list, remove_any_zeroes=True)
target, features = targetFeatureSplit(data)
features = numpy.reshape( numpy.array(features), (len(features), 1))
target = numpy.reshape( numpy.array(target), (len(target), 1))

### training-testing split needed in regression, just like classification
from sklearn.cross_validation import train_test_split

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.1,
                                                                          random_state=42)
train_color = "b"
test_color = "r"


### Your regression goes here!
### Please name it reg, so that the plotting code below picks it up and
### plots it correctly. Don't forget to change the test_color above from "b" to
### "r" to differentiate training points from test points.
from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit(feature_train, target_train)
reg.predict(feature_train)

print reg.coef_
print reg.intercept_
print reg.score(feature_train, target_train)

print reg.score(feature_test, target_test)
feature_test = numpy.reshape( numpy.array(feature_test), (len(feature_test), 1))
target_test = numpy.reshape( numpy.array(target_test), (len(target_test), 1))
print reg.score(feature_test, target_test)
print reg.predict(feature_test)
print accuracy_score(reg.predict(feature_test), target_test)
"""
### draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt

for feature, target in zip(feature_test, feature_test):
    plt.scatter(feature, target, color=test_color)
for feature, target in zip(feature_train, target_train):
    plt.scatter(feature, target, color=train_color)

### labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")

### draw the regression line, once it's coded

try:
    plt.plot(feature_test, reg.predict(feature_test))
except NameError:
    pasa
#reg.fit(feature_test, target_test)

#print reg.coef_
plt.plot(feature_train, reg.predict(feature_train), color="b")
plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()

### identify and remove the most outlier-y points
cleaned_data = []
try:
    predictions = reg.predict(feature_train)
    cleaned_data = outlierCleaner(predictions, feature_train, target_train)
except NameError:
    print "your regression object doesn't exist, or isn't name reg"
    print "can't make predictions to use in identifying outliers"


### only run this code if cleaned_data is returning data
if len(cleaned_data) > 0:
    feature_train, target_train, errors = zip(*cleaned_data)
    feature_train = numpy.reshape( numpy.array(feature_train), (len(feature_train), 1))
    target_train = numpy.reshape( numpy.array(target_train), (len(target_train), 1))

    ### refit your cleaned data!
    try:
        reg.fit(feature_train, target_train)
        print reg.coef_
        print reg.score(feature_test, feature_test)
        plt.plot(feature_train, reg.predict(feature_train), color="blue")
    except NameError:
        print "you don't seem to have regression imported/created,"
        print "   or else your regression object isn't named reg"
        print "   either way, only draw the scatter plot of the cleaned data"
    plt.scatter(feature_train, target_train, color=train_color)
    plt.xlabel("ages")
    plt.ylabel("net worths")
    plt.show()

else:
    print "outlierCleaner() is returning an empty list, no refitting to be done"

"""