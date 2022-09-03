import pandas as pd
import numpy as np
import os
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
pd.set_option("display.max_rows",2000)

stop_words=stopwords.words("english")
list_more_stop_words=["writes","one","subject","would","lines","message-id","article","organization","references","people","gmt","distribution","newsgroups","subject","reply-to","sender","path","date","-","--"
,"i'm","could","xref","also","=","b","i've","etc","that's","i.e","we're","_","$"]

print(stop_words)


def read(file):
    with open(file) as f:
        return(f.read())

list_classes=os.listdir(r"C:\Users\Gurmehar\Desktop\PC stuff\ML\20_newsgroups")
series_classes=pd.Series(list_classes)
series_all_words=pd.Series([],dtype=str)

for dir in list_classes:
    i=0
    
    #list_target=[i for j in range(1000)]
    for file in os.listdir(r"C:\Users\Gurmehar\Desktop\PC stuff\ML\20_newsgroups\\"+dir):
        i=i+1
        if i>200:
            continue
        single_file= read(r"C:\Users\Gurmehar\Desktop\PC stuff\ML\20_newsgroups\\"+dir+"\\"+file)
        list_words=single_file.split()
        series_words=pd.Series(list_words)
        series_all_words=pd.concat([series_all_words,series_words])
    print(series_all_words)






series_all_words=series_all_words.str.strip()
series_all_words=series_all_words.str.strip("")
series_all_words=series_all_words.str.strip(":")
series_all_words=series_all_words.str.strip(">")
series_all_words=series_all_words.str.strip(">>")
series_all_words=series_all_words.str.strip("?")
series_all_words=series_all_words.str.strip('"')
series_all_words=series_all_words.str.strip(')')
series_all_words=series_all_words.str.strip('(')
series_all_words=series_all_words.str.strip('*')
series_all_words=series_all_words.str.strip(',')
series_all_words=series_all_words.str.strip('!')
series_all_words=series_all_words.str.strip("'")
series_all_words=series_all_words.str.strip("|")
series_all_words=series_all_words.str.strip(".")
series_all_words=series_all_words.str.strip("#")
series_all_words=series_all_words.str.strip("/")
series_all_words=series_all_words.str.strip("&")
series_all_words=series_all_words.str.strip("+")
series_all_words=series_all_words.str.strip("\\")
series_all_words=series_all_words.str.strip(";")
series_all_words=series_all_words.str.strip("{")
series_all_words=series_all_words.str.strip("@")
series_all_words=series_all_words.str.strip("}")

series_all_words=series_all_words.str.lower()

counts=pd.DataFrame(series_all_words.value_counts())
counts=counts.reset_index().rename(columns={"index":"word",0:"count"})

counts.drop(index=counts.loc[np.logical_or(counts["word"].isin(stop_words),counts["word"].isin(list_more_stop_words)),:].index,inplace=True)
counts.drop(index=counts.loc[counts["count"]<4].index,inplace=True)
print(counts.head(1000))
#print(counts.tail(100))
print(counts.shape)

counts.sort_values(by="count",ascending=False,inplace=True)
counts=counts.head(1000)
counts.to_csv("thousandfeatures")


