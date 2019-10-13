from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression  
from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime
import os
from pathlib import Path
data = load_iris()
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data.data, data.target, test_size=0.3, random_state=5) 
today=datetime.date.today() 
formatted_today=today.strftime('%y%m%d')
dest_train = os.path.join('model')
joblib_file = os.path.join(dest_train,formatted_today+"joblib_model.pmml")
joblib_model = joblib.load(joblib_file)
score = joblib_model.score(Xtest, Ytest)  
print("Test score: {0:.2f} %".format(100 * score))  
        #    Ypredict = joblib_model.predict(Xtest) 
