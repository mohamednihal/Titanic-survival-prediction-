import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

passengers = pd.read_csv('passengers.csv')
pd.set_option('display.max_columns', None)

#data cleaning
'''
changing male to 0 and female 1
'''
passengers['Sex'] = passengers['Sex'].map({'male':0, 'female':1})
# print(passengers.head())

#fill the nan values of age with mean of that column
passengers['Age'].fillna(value = round(passengers.Age.mean()), inplace=True)

print(passengers.isnull().sum())


# create first class
passengers['FirstClass'] = passengers.apply(lambda row: 1 if row.Pclass ==1 else 0, axis=1)
print(passengers.head())

#create second class
passengers['SecondClass'] = passengers['Pclass'].apply(lambda x : 1 if x == 2 else 0)
print(passengers.head())

#select training and testing features
features = passengers[['Sex','Age','FirstClass','SecondClass']]
survival = passengers['Survived']

#perform train test and split
train_features, test_features , train_labels, test_labels = train_test_split(features, survival, train_size=0.8)

#Standard scalar
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)


#creating model
model = LogisticRegression()
model.fit(train_features,train_labels)

# Score the model on the train data
print(model.score(train_features, train_labels))

# Score the model on the test data
print(model.score(test_features, test_labels))
print(model.coef_)
