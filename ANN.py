import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn. model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential

df = pd.read_excel(r"Folds5x2_pp.xlsx")
#excludes only the last row, so x is the independent variables of AT, V, AP, and RH
x = df.iloc[:,:-1].values
#is only the dependent variable, the output, and we do values to not get the column name
y = df.iloc[:, -1].values
#Splitting the shit into training and testing
X_train, X_test, y_Train, y_Test = train_test_split(x, y, test_size=0.2, random_state=0)
#create/initialize a neural network, we need this step to INITIALIZE IT before we add any layers
ann = tf.keras.models.Sequential()
#add the input layers, units for hidden layers, as the model collects the input layers, the 4 ones
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
#adding another hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
#creating the ouput layer, which is the predicted shit. Sigmoid/SoftMax is used for classification, but for Regression, we will use none. SIgmoid is for simple and binary classification, whereas SoftMax is
# for multivariate data analysis
#If you are trying to predict a continuous number, a real number for regression, just choose no activation function
ann.add(tf.keras.layers.Dense(units=1))
#compiling our neural network for optimizers and loss functions, optimizers for Stochastic Gradient Descent (updating the weights), adam optimizer for Stochastic Gradient Descent (highly recommended, regresssion or classification)
ann.compile(optimizer="adam", loss="mean_squared_error")
tf.keras.metrics.Accuracy(
    name='accuracy', dtype=None
)
#fitting the model, to train it, you can add batchsize, epochs, and what to fit
ann.fit(X_train, y_Train, batch_size=32, epochs=100)
#predicting the results of the test set
Y_pred = ann.predict(X_test)
np.set_printoptions(precision=4)
#Since this is horizaontal, and we want it vertically, we do reshape
predicted_values = Y_pred.reshape(len(Y_pred), 1)
test_values = y_Test.reshape(len(y_Test), 1)
sdaf = abs(predicted_values - test_values)


#print(np.concatenate((predicted_values, test_values), 1))
#sdaf.to_excel("skrt.xlsx")
print(sdaf)

