import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('mail_data.csv')
data = df.where((pd.notnull(df)),'')
data.head(10) #prints the top 5 rows of the particular dataset

data.info()
data.shape #shows the rows and the columns
data.loc[data['Category']=='spam','Category',]=0
data.loc[data['Category']=='ham','Category',]=1
X = data['Message']
Y = data['Category']

print(X)
print(Y)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=3)#80% training and 20% testing
print(X.shape)
print(X_train.shape)
print(X_test.shape)
print(Y.shape)
print(Y_train.shape)
print(Y_test.shape)
feature_extraction = TfidfVectorizer(min_df=1,stop_words='english',lowercase=True) # Change 'True' to True
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
# It seems like you might have a typo here. Did you mean astype?
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')
# Change Y_train_astype to Y_train.astype
print(X_train)
print(X_train_features)
#now train the logistic regression model
model = LogisticRegression()
model.fit(X_train_features,Y_train)

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train,prediction_on_training_data)
print("Accuracy on training data: ", accuracy_on_training_data)

print("Accuracy on training data: ", accuracy_on_training_data)

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test,prediction_on_test_data)
print("accuracy on test data:", accuracy_on_test_data)

input_your_mail = ["THIS IS THE 2ND TIME WE HAVE TRIED REACHING YOU"]
input_data_feature = feature_extraction.transform(input_your_mail)
prediction = model.predict(input_data_feature)
print(prediction)
if(prediction[0]==1):
    print("Ham mail")
else:
    print("Spam mail")


