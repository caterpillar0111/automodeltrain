from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression  
from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split
import pickle
import gzip
import os


def load_train(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv) # skip header
        # for line in csv:
        #     text, label = line[:-3], int(line[-2])
        #     yield text, label
    print(next(load_train(path=os.path.abspath(os.getcwd()+'/train/190629iris.csv')))  )

