import pandas as pd
import pickle

df_ads = pd.read_csv('易速鲜花微信软文.csv')

df_ads['转发数'].fillna(df_ads['点赞数'], inplace=True)

# dataset['转发数'].fillna(dataset['转发数'].mean(), inplace=True)

X = df_ads.drop(['浏览量'],axis=1) # 特征集，Drop掉标签相关字段
y = df_ads.浏览量 # 标签集

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X, y)

pickle.dump(regressor, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

print(model.predict([[300, 800]]))