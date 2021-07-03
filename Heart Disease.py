import pandas as pd
import tensorflow as tf
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn. model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.compose import make_column_transformer
import matplotlib.pyplot as plt
import numpy as np

heart_data = pd.read_csv(r'heart.csv')
o2_data = pd.read_csv(r'o2Saturation.csv')
df = pd.concat([heart_data, o2_data], axis=1, join='inner')
df.rename(columns={'98.6':'Oxygen_Saturation', 'caa':'ca', 'thalachh':'thalach'}, inplace=True)
print(df.columns)
x = df.iloc[:, [0,1,2,3,4,5,6,7,8,9,11,14]].values #X will be our independent variables set
y = df.iloc[:, 13].values #Y will signifiy our dependent variable
#We have successfuly split our dataset into the independent and dependent variables; now we must feature scale them, starting off by Label Encoding the stuff that has order
Label_Encode = LabelEncoder()
x[:, 1] = Label_Encode.fit_transform(x[:, 1])
x[:, 5] = Label_Encode.fit_transform(x[:, 5])
x[:, 6] = Label_Encode.fit_transform(x[:, 6])
x[:, 11] = Label_Encode.fit_transform(x[:, 11])
#Now we shall apply One-Hot-Encoding
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2,11])], remainder="passthrough")
x = np.array(ct.fit_transform(x))
#Now we can finally split it into the testing and training batches
X_train, X_test, y_Train, y_Test = train_test_split(x, y, test_size=0.2, random_state=0)
#Now we will have to do feature scaling to everything, regardless of whether or not it has numerical digits or not, scaler only fitted to training set to prevent information leakage
sc = StandardScaler()
#fit transform gets mean and stdev
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#fit_transform() is used on the training data so that we can scale the training data and also learn the scaling parameters of that data. Here, the model built by
# us will learn the mean and variance of the features of the training set. These learned parameters are then used to scale our test data.

#Using the transform method we can use the same mean and variance as it is calculated from our training data to transform our test data.
# Thus, the parameters learned by our model using the training data will help us to transform our test data.
plt.hist(df['thalach'])
#Now we split the data into training and testing data
X_train, X_test, y_Train, y_Test = train_test_split(x, y, test_size=0.2, random_state=0)
#Now we standardize with feature scaling to prevent information leakage, as well as fit:
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
ANN = tf.keras.models.Sequential()
ANN.add(tf.keras.layers.Dropout(rate=.3)) #How many neurons ignored
ANN.add(tf.keras.layers.Dense(units=121 , activation='relu'))
ANN.add(tf.keras.layers.Dense(units=17, activation='relu'))
ANN.add(tf.keras.layers.Dense(units=12, activation='relu'))
ANN.add(tf.keras.layers.Dense(units=11 , activation='relu'))
ANN.add(tf.keras.layers.Dense(units=171, activation='relu'))
ANN.add(tf.keras.layers.Dense(units=12, activation='relu'))
ANN.add(tf.keras.layers.Dense(units=121 , activation='relu'))
ANN.add(tf.keras.layers.Dense(units=17, activation='relu'))
ANN.add(tf.keras.layers.Dense(units=12, activation='relu'))
#Output layer. Not all output layers have units equal to 1, sigmoid gives us the probability that the binary outcome is 1 in addition to the prediction, predicting more than 2 categories would be SoftMax
ANN.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
#Now for the training, and compiling. We will be using the accuracy metrics to evaluate the Neural Network. We use loss=binary_crossentropy as we are predicting a binary value (0 or 1) if they leave or not, and for nonbinary classification we use categorical_crossentropy
ANN.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = ANN.fit(X_train, y_Train, batch_size=32, epochs=75)
#Now we can make our predictions via y_pred
y_pred = tf.round(ANN.predict(X_test))
print(X_test.shape)
print(ANN.evaluate(X_test,y_Test))
plt.figure(figsize = (8, 8))
Confusion_Matrix = confusion_matrix(y_Test,y_pred)
sns.heatmap(Confusion_Matrix, cmap = 'Blues', annot = True,
           yticklabels = ['No Heart Attack', 'Heart Attack'], xticklabels = ['Predicted No Heart Attack', 'Predicted Heart Attack'])
plt.show()