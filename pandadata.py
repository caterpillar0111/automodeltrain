import pandas as pd 
import os
# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 
data = pd.read_csv(os.path.abspath(os.getcwd()+'/train/190701iris.csv'))
# Preview the first 5 lines of the loaded data 
# print(data.head())

# df = data.DataFrame()
print(data['sepal length (cm)'])
print(data.loc[0])