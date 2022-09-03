from matplotlib.pyplot import grid
import pandas as pd
import numpy as np

df=pd.read_csv(r"D:\Dekstop\Dekstop folders\Data Science\ML\ML1\final_raw_data",index_col=[0])

df["sum"]=df.sum(axis=1)
df.drop(index=df[df["sum"]==0].index,inplace=True)
#print(df[df["sum"]==0])
df.drop("sum",axis=1,inplace=True)

print(df.shape)
Y=df["target_y"]
X=df.drop("target_y",axis=1)
print(Y.shape,X.shape)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,stratify=Y)


from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB(alpha=0.000001)
clf.fit(x_train,y_train)
acc=clf.score(x_train,y_train)
print(acc)
acc=clf.score(x_test,y_test)
print(acc)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
y_pred=clf.predict(x_train)
score=accuracy_score(y_train,y_pred)
matrix=confusion_matrix(y_train,y_pred)
report=classification_report(y_train,y_pred)

#print(score,matrix,report)
########
y_pred=clf.predict(x_test)
score=accuracy_score(y_test,y_pred)
matrix=confusion_matrix(y_test,y_pred)
report=classification_report(y_test,y_pred)

#print(score,matrix,report)

from sklearn.model_selection import GridSearchCV
param_grid={"alpha":(1,0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001),"fit_prior":(True,False)}
grid=GridSearchCV(MultinomialNB(),param_grid=param_grid,scoring="accuracy")
grid.fit(x_train,y_train)
print(grid.score(x_train,y_train))
print(grid.score(x_test,y_test))
