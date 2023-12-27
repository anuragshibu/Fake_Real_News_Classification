import numpy as np
import pandas as pd
import tokenize
df = pd.read_csv('news_articles.csv')
df
df.shape
df['text'][0]
df = df[['text','label']]
df
df = df.dropna()
df
df.shape
df.label.value_counts()
df.info()
df['label num'] = df['label'].map({'Fake':0,'Real':1})
df.sample(5)
!python -m spacy download en_core_web_lg
import spacy
nlp = spacy.load("en_core_web_lg")
df.head()
df['vector'] = df['text'].apply(lambda x:nlp(x).vector)
df.head()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(df.vector.values,df['label num'])
x_train.shape
x_test.shape
y_train.shape
y_test.shape
x_train_2d = np.stack(x_train)
x_test_2d = np.stack(x_test)
x_train_2d
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
nv = MultinomialNB()
sc = MinMaxScaler()
sc_train = sc.fit_transform(x_train_2d)
sc_test  = sc.fit_transform(x_test_2d)
nv.fit(sc_train,y_train)
sc_train
y_pred = nv.predict(sc_test)
y_pred
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
print(classification_report(y_test,y_pred))
