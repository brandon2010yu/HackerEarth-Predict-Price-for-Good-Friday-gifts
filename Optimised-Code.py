#!/usr/bin/python3

# Getting all the Required Libraries
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import psycopg2 as pg

# Getting the Dataset into the Program
engine = pg.connect("dbname='Research' user='postgres' host='localhost' port='5432' password='React123123'")
df_original = pd.read_sql('select * from t_posts', con=engine)
df_train = df_original.copy()
# df_train = df_train[df_train['forums_id']==34]

# # Using Lable Encoder to Create Categorical Values out of Numerical Ones
# forums_encoder = preprocessing.LabelEncoder()
# forums_encoder.fit(list(df_train['forums_id']))
# users_encoder = preprocessing.LabelEncoder()
# users_encoder.fit(list(df_train['users_id']))
# topics_encoder = preprocessing.LabelEncoder()
# topics_encoder.fit(list(df_train['topics_id']))

# # Replacing Encoded Values with Original Values in the DataFrame
# df_train['forums_id'].replace(list(forums_encoder.transform(list(df_train['forums_id']))) , inplace = True)
# df_train['users_id'].replace(list(users_encoder.transform(list(df_train['users_id']))) , inplace = True)
# df_train['topics_id'].replace(list(topics_encoder.transform(list(df_train['topics_id']))) , inplace = True)

# # Identifing Features and Target Variable
# features = ['topics_id' , 'users_id']
# target = 'forums_id'

# # Slicing the Data Frame to create Model
# m = df_train[['forums_id' , 'users_id' , 'topics_id']]
# d = m.dropna()

# # Data to Train the Model
# X = d.drop(columns = [target], axis = 1)
# Y = d[target]

# # Splitting the Data
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

# # Creating the Model (Optimised)
# model = RandomForestClassifier(max_depth=2, random_state=0)
# model.fit(X_train,Y_train)
# Y_pred = model.predict(X_test)
# conf = round((r2_score(Y_test,Y_pred))*100,3)

# # Printing Confidence of Our Model
# print('Model Confidence : ' , conf)
# print('Confusion Matrix : \n' , confusion_matrix(Y_test, Y_pred))

# # # Model exported ad a Pickle File 
# # joblib.dump(model, 'Model.pkl')

df_original.to_csv('Dataset.csv')