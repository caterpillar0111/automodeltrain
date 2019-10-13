from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression  
from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime
import os
from pathlib import Path
data = load_iris()
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data.data, data.target, test_size=0.3, random_state=4) 
def load_train_data():
    today=datetime.date.today()
    formatted_today=today.strftime('%y%m%d')
    dest_train = os.path.join('train')
    try:
        if os.path.exists(os.path.abspath(os.getcwd()+'/train'))!=True:
            print("train資料夾不存在，新增train資料夾")
            os.makedirs(os.path.abspath(os.getcwd()+'/train'))
    except FileExistsError:
        print("train資料夾已存在")
    finally: 
        df = pd.DataFrame(Xtrain, columns=data.feature_names)
        df['target'] = Ytrain
        df.to_csv(os.path.join(dest_train,formatted_today+'iris.csv'))

def train_save_model():
    model = LogisticRegression(C=0.1,  
                           max_iter=20, 
                           fit_intercept=True, 
                           n_jobs=3, 
                           solver='liblinear')
    model.fit(Xtrain, Ytrain)
    try:
        if os.path.exists(os.path.abspath(os.getcwd()+'/model'))!=True:
            print("model資料夾不存在，新增model資料夾")
            os.makedirs(os.path.abspath(os.getcwd()+'/model'))
    except FileExistsError:
        print("train資料夾已存在")
    finally: 
           today=datetime.date.today()
           formatted_today=today.strftime('%y%m%d')
           dest_train = os.path.join('model')
           joblib_file = os.path.join(dest_train,formatted_today+"joblib_model.pmml")
           joblib.dump(model, joblib_file)
def load_pridect_test():
    try:
        if os.path.exists(os.path.abspath(os.getcwd()+'/test'))!=True:
            print("test資料夾不存在，新增test資料夾")
            os.makedirs(os.path.abspath(os.getcwd()+'/test'))
    except FileExistsError:
        print("test資料夾已存在")
    finally: 
           today=datetime.date.today() 
           formatted_today=today.strftime('%y%m%d')
           dest_train = os.path.join('model')
           joblib_file = os.path.join(dest_train,formatted_today+"joblib_model.pmml")
           joblib_model = joblib.load(joblib_file)
           score = joblib_model.score(Xtest, Ytest)  
           print("Test score: {0:.2f} %".format(100 * score))  
        #    Ypredict = joblib_model.predict(Xtest) 
if os.path.exists(os.path.abspath(os.getcwd()+'/train'))!=True:
    load_train_data()
if os.path.exists(os.path.abspath(os.getcwd()+'/model'))!=True:
    train_save_model()
load_pridect_test()



# data = load_iris()
# Xtrain, Xtest, Ytrain, Ytest = train_test_split(data.data, data.target, test_size=0.3, random_state=4) 
# # pickle.dump(stop, open(os.path.join(dest, formatted_today+'stopwords.pkl'), 'wb'), protocol=4)   
# df = pd.DataFrame(data.data, columns=data.feature_names)
# df['target'] = data.target
# print(df.head)
# df.to_csv(os.path.join(dest_train,formatted_today+'iris.csv'))
# # print(data.target_names)
