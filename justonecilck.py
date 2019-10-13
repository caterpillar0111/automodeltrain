
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression  
from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split
import os.path
import gzip
import os



data = load_iris()  
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data.data, data.target, test_size=0.3, random_state=4) 
# Save to file in the current working directory
# model = LogisticRegression(C=0.1,  
#                            max_iter=20, 
#                            fit_intercept=True, 
#                            n_jobs=3, 
#                            solver='liblinear')
# model.fit(Xtrain, Ytrain)

joblib_file = "joblib_model.pkl"  
# joblib.dump(model, joblib_file)
# Load from file
joblib_model = joblib.load(joblib_file)
# Calculate the accuracy and predictions
score = joblib_model.score(Xtest, Ytest)  
print("Test score: {0:.2f} %".format(100 * score))  
Ypredict = joblib_model.predict(Xtest)  