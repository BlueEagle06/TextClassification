import pandas as pd
import numpy as np
import os
pd.set_option("display.max_rows",2000)
features=pd.read_csv(r"C:\Users\Gurmehar\Desktop\PC stuff\ML\thousandfeatures",index_col=[0])
features=features.dropna(axis=0).reset_index().drop("index",axis=1)
#print(features)

df=pd.DataFrame([[0.00 for i in range(999)] for j in range(20000)],columns=features["word"])
print(df.shape)
#print(df)
df1=df.copy()
df["target_y"]=pd.Series([0.00 for i in range(20000)])

def read(file):
    with open(file) as f:
        return f.read()

i=0
j=0
for dir in os.listdir(r"C:\Users\Gurmehar\Desktop\PC stuff\ML\20_newsgroups"):
    j=j+1
    print(str((j-1)*5)+r"% completed")
    for file in os.listdir(r"C:\Users\Gurmehar\Desktop\PC stuff\ML\20_newsgroups"+r"\\"+dir):
        
        data=read(r"C:\Users\Gurmehar\Desktop\PC stuff\ML\20_newsgroups"+r"\\"+dir+r"\\"+file)
        list_words_single_file=data.split()
        series_words_single_file=pd.Series(list_words_single_file)
        check=series_words_single_file.isin(features["word"])
        series_words_single_file_true=series_words_single_file.loc[check]
        for word in series_words_single_file_true:
            for col in df1.columns:
                if word ==col:
                    df.loc[i,col]=df.loc[i,col]+1
                    continue
        df.loc[i,"target_y"]=j            
                    
        i=i+1

print(str(100)+ r"% completed")
print(df)  

df.to_csv("final_raw_data")