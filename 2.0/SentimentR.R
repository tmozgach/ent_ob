# 1 step
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

newData = data.frame(col1 = myData$Title, col2 = myData$Conversation) %>% 
  do(na.locf(.))

newData = newData[!duplicated(newData$col1),]

write_csv(newData['col1'], "/home/tatyana/nlp/2.0/Titles2.0.csv")
write_csv(newData['col2'],  "/home/tatyana/nlp/2.0/Bodies2.0.csv")


#1.5 step
#manually delete column names
#2 step
#scp Titles2.0.csv tmozgach@cedar.computecanada.ca:/home/tmozgach/scratch/TM2.0

#scp Bodies2.0.csv tmozgach@cedar.computecanada.ca:/home/tmozgach/scratch/TM2.0

#3 step
# sbatch TM_job2.0.sh 30 Titles2.0.csv

#3.5 return back
#scp tmozgach@cedar.computecanada.ca:/home/tmozgach/scratch/TM2.0/LabeledTopic* .
#scp tmozgach@cedar.computecanada.ca:/home/tmozgach/scratch/TM2.0/TM2.0_lda* .
#scp tmozgach@cedar.computecanada.ca:/home/tmozgach/scratch/TM2.0/TM2.0_viz* .

# NOTE!!!!! for bodies delete ^M search for: tanning salon they

# 4step
myTM30 = read_delim("/home/tatyana/nlp/2.0/LabeledTopic2.0_30_Bodies2.csv", delim = ',' )
myTM40 = read_delim("/home/tatyana/nlp/2.0/LabeledTopic2.0_40_Bodies2.csv", delim = ',' )
myTM50 = read_delim("/home/tatyana/nlp/2.0/LabeledTopic2.0_50_Bodies2.csv", delim = ',' )
sentiment = read_delim("/home/tatyana/nlp/2.0/SentimentBodies.csv", delim = ',' )
temp = merge(myTM30,myTM40, by.x = 'X1', by.y = 'X1')
temp1 = merge(temp,myTM50, by.x = 'X1', by.y = 'X1')
fin = select(temp1,'text','Topic/Probability.x', 'Main Topic.x', 'Main Probability.x','Topic/Probability.y', 'Main Topic.y', 'Main Probability.y','Topic/Probability', 'Main Topic', 'Main Probability')
names(fin)[1]<- "Body"
names(fin)[2]<- "Topic/Probability(30 topics)"
names(fin)[3]<- "Main Topic(30 topics)"
names(fin)[4]<- "Main Probability(30 topics)"
names(fin)[5]<- "Topic/Probability(40 topics)"
names(fin)[6]<- "Main Topic(40 topics)"
names(fin)[7]<- "Main Probability(40 topics)"
names(fin)[8]<- "Topic/Probability(50 topics)"
names(fin)[9]<- "Main Topic(50 topics)"
names(fin)[10]<- "Main Probability(50 topics)"

fin = cbind(fin,sentiment)
fin <-fin[-c(11)]
write_csv(fin, "/home/tatyana/nlp/2.0/final_table_bodies2.0.csv")
