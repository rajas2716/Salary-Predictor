import pandas as pd
import pickle
#Read CSV file
dataset = pd.read_csv('hiring.csv')
dataset['experience'].fillna(0, inplace = True)
dataset['test_score(out of 10)'].fillna(dataset['test_score(out of 10)'].mean(),inplace = True)

def convert_to_int(num):
    dict = {"one":1 , "two" :2 ,"three":3 , "four":4 ,"five": 5 , "six":6, "seven":7 , "eight": 8, "nine":9 ,"ten": 10 ,"eleven":11 ,'zero':0,0:0}
    return dict[num]

dataset['experience'] = dataset['experience'].apply(lambda x: convert_to_int(x))
X = dataset.iloc[:,:3]
y = dataset.iloc[:,3]
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X,y)
pickle.dump(regressor,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
