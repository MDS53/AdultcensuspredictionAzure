from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score,precision_score
from sklearn.metrics import recall_score
import logging
import pickle


df=pd.read_csv("C:\\Users\\new\\OneDrive\\Desktop\\Ds\\EDA\\Adult EDA\\Perfect_Adult_cleaned_dataset.csv")
logging.basicConfig(filename='Training_pipeline.log',level=logging.DEBUG,format='%(asctime)s %(levelname)s %(message)s')
df.drop(df.columns[0],axis=1,inplace=True)

class split:
    def spliting(self,df):
        """This function takes Dataframe and splits df into train,test"""
        try:
            self.df=df
            self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.df.drop(['Income'],axis=1),self.df['Income'],test_size=0.25,random_state=1)
            logging.info("train test Splitting Done")
            return self.X_train,self.X_test,self.y_train,self.y_test
        except Exception as e:
            logging.info("train test Splitting NOT done")
            logging.error(e)
            print(e)

class Transformer:
    def __init__(self):
        """This function creates all Transformers tnf1,tnf2,tnf3,tnf4"""
        try:
            self.tnf1=ColumnTransformer(transformers=[
            ("Handling Int-Missing values",SimpleImputer(),[0,2,3,8,9,10]),
            ("Handling Cat-Missing values",SimpleImputer(strategy='most_frequent'),[1,4,5,6,7,11])
        ],remainder='passthrough')
            logging.info("TNF1 done")


            self.tnf2=ColumnTransformer(transformers=[
                ('Encoding',OneHotEncoder(sparse=False,drop='first',dtype='int',handle_unknown='ignore'),slice(6,13))
            ],remainder='passthrough')
            logging.info("TNF1 done")

            self.tnf3=ColumnTransformer(transformers=[
                ('Feature scaling',StandardScaler(),slice(0,76))
            ],remainder='passthrough')
            logging.info("TNF1 done")
            
            self.tnf4=LogisticRegression()
        except Exception as e:
            logging.error("Transformers NOT done")
            logging.error(e)
            print(e)

class Pipeline:
    def __init__(self):
        """This function creates a Pipeline using tnf1, tnf2 ,tnf3 and tnf4."""
        try:
            self.tnf=Transformer()
            self.Pipe=make_pipeline(self.tnf.tnf1,self.tnf.tnf2,self.tnf.tnf3,self.tnf.tnf4)
            logging.info("Pipeline making done")
        except Exception as e:
            logging.error("Pipeline making not done")
            logging.error(e)
            print(e)
        
class fit_pred:
    def __init__(self,df):
        """This function do predictions"""
        try:
            self.df=df
            self.spli=split()
            self.X_train,self.X_test,self.y_train,self.y_test=self.spli.spliting(self.df)
            self.pip=Pipeline()
            self.pkle=pickling(self.pip)
            self.pip.Pipe.fit(self.X_train,self.y_train)
            self.ypred=self.pip.Pipe.predict(self.X_test)
            self.accscr=performance_metrics()
            self.acc=self.accscr.acc(self.y_test,self.ypred)
            self.recall=self.accscr.recall(self.y_test,self.ypred)
            self.precision=self.accscr.Precision(self.y_test,self.ypred)
            logging.info("Predictions DOne")
            print(f"Accuracy :{self.acc}")
            print(f"Recall :{self.recall}")
            print(f"Precision :{self.precision}")
        except Exception as e:
            logging.error("Prediction NOT Done")
            logging.error(e)
            print(e)
class pickling:
    def __init__(self,Pipe):
        """This function creates a pkl file containing training Pipeline"""
        self.Pipe=Pipe
        pickle.dump(self.Pipe,open('pipe2.pkl','wb'))
class performance_metrics:
    def acc(self,y_test,ypred):
        """This function takes y_test,y_predicted and returns accuracy score"""
        try:
            self.y_test=y_test
            self.ypred=ypred
            logging.info('Performance metrics Accuracy DOne')
            return accuracy_score(self.y_test,self.ypred)
        except Exception as e:
            logging.error("Performance metrics Accuracy NOT DOne ")
            logging.error(e)
            print(e)
    def recall(self,y_test,ypred):
        """This function takes y_test,y_predicted and returns recall score"""
        try:
            self.y_test=y_test
            self.ypred=ypred
            logging.info('Performance metrics recall DOne')
            return recall_score(self.y_test, self.ypred,pos_label=' >50K')
        except Exception as e:
            logging.error("Performance metrics recall NOT DOne ")
            logging.error(e)
            print(e)
    def Precision(self,y_test,ypred):
        """This function takes y_test,y_predicted and returns Precision score"""
        try:
            self.y_test=y_test
            self.ypred=ypred
            logging.info('Performance metrics Precision DOne')
            return precision_score(self.y_test, self.ypred,pos_label=' >50K')
        except Exception as e:
            logging.error("Performance metrics Precision NOT DOne ")
            logging.error(e)
            print(e)




"""tnf1=ColumnTransformer(transformers=[
    ("Handling Int-Missing values",SimpleImputer(),[0,2,3,8,9,10]),
    ("Handling Cat-Missing values",SimpleImputer(strategy='most_frequent'),[1,4,5,6,7,11])
],remainder='passthrough')


tnf2=ColumnTransformer(transformers=[
    ('Encoding',OneHotEncoder(sparse=False,drop='first',dtype='int',handle_unknown='ignore'),slice(6,13))
],remainder='passthrough')

tnf3=ColumnTransformer(transformers=[
    ('Feature scaling',StandardScaler(),slice(0,76))
],remainder='passthrough')



tnf4=LogisticRegression()

Pipe=make_pipeline(tnf1,tnf2,tnf3,tnf4)


Pipe.fit(X_train,y_train)
ypred=Pipe.predict(X_test)


print(accuracy_score(y_test,ypred))"""
p=fit_pred(df)