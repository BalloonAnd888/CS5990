#-------------------------------------------------------------------------
# AUTHOR: Andrew Lau
# FILENAME: naive_bayes.py
# SPECIFICATION: Use Naive Bayes to classify weather data
# FOR: CS 5990- Assignment #4
# TIME SPENT: 2 hrs
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd

#11 classes after discretization
classes = [i for i in range(-22, 39, 6)]

s_values = [0.1, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001]

#reading the training data
#--> add your Python code here
trainingData = pd.read_csv("weather_training.csv")
xTrain = np.array(trainingData.values)[:,1:-1].astype('f')
yTrain = np.array(trainingData.values)[:,-1].astype('f')

#update the training class values according to the discretization (11 values only)
#--> add your Python code here
yTrain = np.digitize(yTrain, classes) - 1

#reading the test data
#--> add your Python code here
testData = pd.read_csv("weather_test.csv")
xTest = np.array(testData.values)[:,1:-1].astype('f')
yTest = np.array(testData.values)[:,-1].astype('f')

#update the test class values according to the discretization (11 values only)
#--> add your Python code here
yTest = np.digitize(yTest, classes) - 1

#loop over the hyperparameter value (s)
#--> add your Python code here
highestAccuracy = 0

for s in s_values:

    #fitting the naive_bayes to the data
    clf = GaussianNB(var_smoothing=s)
    clf = clf.fit(xTrain, yTrain)

    correct = 0

    #make the naive_bayes prediction for each test sample and start computing its accuracy
    #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values
    #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
    #--> add your Python code here
    for x_testSample, y_testSample in zip(xTest, yTest):
        prediction = clf.predict([x_testSample])

        if y_testSample != 0 and 100 * abs(prediction - y_testSample) / y_testSample <= 15:
            correct += 1

    accuracy = correct / len(yTest)

    # check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
    # with the KNN hyperparameters. Example: "Highest Naive Bayes accuracy so far: 0.32, Parameters: s=0.1
    # --> add your Python code here
    if accuracy > highestAccuracy:
        highestAccuracy = accuracy
        print(f"Highest Naive Bayes accuracy so far: {highestAccuracy}")
        print(f"Parameters: s={s}")
