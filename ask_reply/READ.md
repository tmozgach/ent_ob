#original data prepare for python
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
myData['Sender2'] = data.frame(col1 = myData$Sender, col2 = myData$Title2) %>% 
  do(na.locf(.))
myData$Replier[is.na(myData$Replier)] <- as.character(myData$Sender[is.na(myData$Replier)])


write_csv(myData, "/home/tatyana/nlp/full_for_python.csv")
