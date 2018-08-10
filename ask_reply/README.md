### The participants’ ask/reply behaviour in their every 12 months in the community
__first_appear__: date of his first appearance in this community (It is from the first time he appear (either start a thread or reply in a thread)

__last_appear__: date of his last appearance in this community

__ask_y1__: how many threads he intiated in his first 12 months in the community

__reply_y1__: how many threads he replied to others in his first 12 months in the community

__received_y1__: how many people replied him to his questions in his first 12 months in the community- nonredundant count

__Total_points__: add all his points gained from answering questions

_Note_:
Total_points refer to the individual’s total sum of points gained as a sender and a receiver. The Karma is a score for the individual’s behaviour in the whole reddit community and may be obtained from other sub forums other than the entrepreneur one.

The points for sender is a sum of the points in receiver in his thread. The sum of the repliers’ points can be seen as the ‘value’ of the thread. A thread with 5 points from replier is not as valuable as another with 50 points. So a sender who generated a thread with 50 points can be seen ‘contributing’ to the community a more valuable thread. 

Non-redundant count means in every thread, each replier can only count once. Sometimes a replier may reply several times under one thread. But for counting ‘received_y1’, or ‘received_y2’ etc, we only count them once so as to know accurately how many people replied the inquirer.

The redundant information has been added for easy data manipulation using the following code in R

```
#original data prepare for python

library(tidyverse)
library(zoo)
library(dplyr)

myDatafull = read_delim("/home/tatyana/Downloads/data_full.csv", delim = ',' )
myData = myDatafull[!is.na(strptime(myDatafull$Date,format="%Y-%m-%d %H:%M:%S")),]

# There are title that duplicates another one. Titles are not unique
myData$Title <- make.unique(as.character(myData$Title), sep = "___-___")

# make.uniqui makes also NA - unique by adding number, need to transform them back to NA
myData$Title <- gsub("NA__+", NA, myData$Title)

# change NA by previous Title
myData['Title2'] = data.frame(col1 = myData$Title, col2 = myData$Conversation) %>% 
  do(na.locf(.))
myData['Sender2'] = data.frame(col1 = myData$Sender, col2 = myData$Title2) %>% 
  do(na.locf(.))
myData$Replier[is.na(myData$Replier)] <- as.character(myData$Sender[is.na(myData$Replier)])


write_csv(myData, "/home/tatyana/nlp/full_for_python.csv")
```

Main data manipulation occurs in the Python code:

```
# python -m pip install <name of packages/library>
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models.ldamodel import LdaModel as Lda
from gensim import corpora
import string
import os
import codecs
import pandas as pd
import pyLDAvis.gensim
from operator import itemgetter
import csv
import textblob
from textblob import TextBlob

import sys

df = pd.read_csv("full_for_python.csv")

#Date of they last appearance in this community, Note: Replier also includes Senders 
df1 = df.sort_values('Date').groupby('Replier').tail(1)
df_last = df1[['Date','Replier']]
df_last.rename(columns={'Date': 'last_appear'}, inplace=True)

#Date of they first appearance in this community
df2 = df.sort_values('Date').groupby('Replier').head(1)
df_first = df2[['Date','Replier']]
df_first.rename(columns={'Date': 'first_appear'}, inplace=True)

pk = df[['Post Karma', 'Replier']].drop_duplicates(subset=['Replier'])
ck = df[['Comment Karma', 'Replier']].drop_duplicates(subset=['Replier'])

df31 = pd.merge(df_first,df_last, on = 'Replier', how = 'inner')
df32 = pd.merge(df31, pk, on = 'Replier', how = 'left')
df3 = pd.merge(df32, ck, on = 'Replier', how = 'left')
 
# Points per a thread
df_points = df.groupby('Title2')['Points from this question'].sum().reset_index()
df_points.rename(columns={'Points from this question': 'points_per_a_thread'}, inplace=True)

df4 = pd.merge(df,df_points, on='Title2', how = 'left')

# Total sum of points gained as a sender
points_s = df4.groupby('Sender')['points_per_a_thread'].sum().reset_index()
points_s.rename(columns={'points_per_a_thread': 'total_points_per_a_sender', 'Sender':'Replier'}, inplace=True)


# Total sum of points gained as a replier
points_r = df4.groupby('Replier')['Points from this question'].sum().reset_index()
points_r.rename(columns={'Points from this question': 'total_points_per_a_replier'}, inplace=True)

df41 = pd.merge(df4,points_s, on = 'Replier', how = 'left')
df5 = pd.merge(df41,points_r,on = 'Replier', how = 'left')

df7 = pd.merge(df5,df_first, on ='Replier',how = 'left')

# The fisrt appearance of sender for the current thread
df_first.rename(columns={'first_appear':'first_appear_as_sender2','Replier': 'Sender2'}, inplace=True)
df8 = pd.merge(df7,df_first, on ='Sender2',how = 'left')

df9 = pd.merge(df8,df_last, on = 'Replier', how = 'left')
final = df9

final['Date'] = pd.to_datetime(final['Date'])
final['last_appear'] = pd.to_datetime(final['last_appear'])
final['first_appear'] = pd.to_datetime(final['first_appear'])
final['first_appear_as_sender2'] = pd.to_datetime(final['first_appear_as_sender2'])

# Non-redundant count means in every thread, each replier can only count once. The below line we need for that purpose
dt_drop_rep = final.drop_duplicates(subset=['Sender','Replier', 'Title2'])

tp = pd.merge(df3,points_s,on='Replier', how = 'left')
tp2 = pd.merge(tp,points_r,on='Replier', how = 'left')

for x in range(9):
	#How many threads the user initiated in his x 12 months
	ask_y = final[(final['Date'] <= (final['first_appear'] + pd.DateOffset(years=(x+1)))) & (final['Date'] >= (final['first_appear'] + pd.DateOffset(years=x))) ].groupby('Sender')['Title'].count().reset_index()	
	ask_y.rename(columns={'Title': 'ask_y' + str(x+1), 'Sender' : 'Replier'}, inplace=True)


	#How many threads the user replied to others in his x 12 months
	reply_y = dt_drop_rep[(dt_drop_rep['Date'] <= (dt_drop_rep['first_appear'] + pd.DateOffset(years=(x+1)))) & (dt_drop_rep['Date'] >= (dt_drop_rep['first_appear'] + pd.DateOffset(years=x))) & (dt_drop_rep['Sender'].isna()) & (dt_drop_rep['Sender2'] != dt_drop_rep['Replier'])].groupby('Replier')['Title2'].count().reset_index()
	reply_y.rename(columns={'Title2': 'reply_y' + str(x+1)}, inplace=True)


	#How many people replied to the user's question in user's x 12 months, nonredundant count
	received_y = dt_drop_rep[(dt_drop_rep['Date'] >= (dt_drop_rep['first_appear_as_sender2'] + pd.DateOffset(years=x))) & (dt_drop_rep['Date'] <= (dt_drop_rep['first_appear_as_sender2'] + pd.DateOffset(years=(x+1)))) & (dt_drop_rep['Sender'].isna()) & (dt_drop_rep['Sender2'] != dt_drop_rep['Replier'])].groupby('Sender2')['Replier'].count().reset_index()
	received_y.rename(columns={'Replier': 'received_y' + str(x+1), 'Sender2' : 'Replier'}, inplace=True)


	temp1 = pd.merge(tp2, ask_y, on='Replier', how = 'left')
	temp2 = pd.merge(temp1, reply_y, on='Replier', how = 'left')
	temp3 = pd.merge(temp2, received_y, on='Replier', how = 'left')
	tp2 = temp3


tp2.rename(columns={'Replier':'name'}, inplace=True)
print(tp2)
tp2.to_csv("ask_reply_behav.csv")
```
