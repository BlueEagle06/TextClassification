
#steps:-
# 1. Create a sum column containing sum for each row excluding target_y column
# 2. Group by target_y column
# 3. For 1st group, sum each column and find product of all sums for that group
# 4. divide by (sum of all (sum of each column))**(number of columns)
# 5. multiply by (length of group/total length of dataframe)
# 6. This is the probability for that particular class(group)
# 7. repeat for each group
# 8. predict group with highest probability as the answer

import pandas as pd
import numpy as np
 
df=pd.read_csv(r"D:\Dekstop\Dekstop folders\Data Science\ML\ML1\final_raw_data",index_col=[0])
df["sum"]=df.sum(axis=1)
df.drop(index=df[df["sum"]==0].index,inplace=True)
df.drop("sum",axis=1,inplace=True)
print(df.shape)
X=df.drop("target_y",axis=1)
print(X.shape)
X["sum"]=X.sum(axis=1)

def train_and_predict(df_train,x_test):
    
    df_train["sum"]=X["sum"]
    print(df_train["sum"])
    print(df_train.shape)
    groupby_target=df_train.groupby("target_y")
    dict1={}
    l=0
    total_rows=df_train.shape[0]
    y_pred=pd.Series([0.0 for j in range(x_test.shape[0])])
    
    for i in range(x_test.shape[0]):
        current_row=pd.Series(x_test.iloc[i,:])
        current_row_true_points=current_row.loc[current_row>0]
        print(x_test.shape[0]-i)
        for name, group in groupby_target:
            dict1[str(name)]=1
            total_sum=group["sum"].sum()
            group_length=group.shape[1]
            
            for true_col in current_row_true_points.index:
                #sum_rows=0
                #for k in range(group.shape[0]):
                
                    #t=(group.iloc[k,:].loc[true_col]+1)/(sum(group.iloc[k,:])+group_length)
                    #sum_rows=sum_rows+t
                t=(group[true_col].sum()+1)/(total_sum+group_length)
                dict1[str(name)]=dict1[str(name)]*t


            o=group.shape[0]/total_rows
            dict1[str(name)]=dict1[str(name)]*(o)
        
        prediction=max(zip(dict1.values(),dict1.keys()))[1]
        #print(prediction)
        y_pred[i]=float(prediction)
    
    return y_pred


from sklearn.model_selection import train_test_split
df_train,df_test=train_test_split(df)

x_test=df_test.drop("target_y",axis=1)
y_test=df_test["target_y"]

x_train=df_train.drop("target_y",axis=1)
y_train=df_train["target_y"]



print("Here we go!")

from sklearn.metrics import accuracy_score

#y_pred_train= train_and_predict(df_train,x_train)

#acc_train=accuracy_score(y_train,y_pred_train)

y_pred_test= train_and_predict(df_train,x_test)
acc_test=accuracy_score(y_test,y_pred_test)

print("Final Results:!")
#print(acc_train)
print(acc_test)

from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB(alpha=0.000001)
clf.fit(x_train,y_train)
acc=clf.score(x_train,y_train)
print(acc)
acc=clf.score(x_test,y_test)
print(acc)

#.7758
#0.8056
#0.8569i