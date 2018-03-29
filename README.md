# Entrepreneur’s online behavior
1) Crawl data for 2012,2013,2014,2015 using the following script:
https://github.com/peoplma/subredditarchive/blob/master/subredditarchive.py

2) Join all files into the one file taking just Title, Text of post, text of comments from crawled data:
https://github.com/tmozgach/ent_ob/blob/master/jsonTOcsvTopicModel.py

3) Topic Modeling:
https://github.com/tmozgach/ent_ob/blob/master/TopicModeling.ipynb



# Instructions:

Data: https://www.dropbox.com/s/50vkf5makcojd5w/data_full.csv?dl=0

Prepare data for merging all conversations/comments and a title:

R: 
```
library(tidyverse)
library(zoo)
library(dplyr)

myDataf = read_delim("/home/tatyana/Downloads/data_full.csv", delim = ',' )
myDataff = myDataf[!is.na(strptime(myDataf$Date,format="%Y-%m-%d %H:%M:%S")),]

# There are title that duplicates another one. Titles are not unique
myDataff$Title <- make.unique(as.character(myDataff$Title), sep = "___-___")

# make.uniqui makes also NA - unique by adding number, need to transform them back to NA
myDataff$Title <- gsub("NA__+", NA, myDataff$Title)

# change NA by previous Title
myDataff['Title2'] = data.frame(col1 = myDataff$Title, col2 = myDataff$Conversation) %>% 
  do(na.locf(.))
write_csv(myDataff, "data_full_title2.csv")

newDff = data.frame(col1 = myDataff$Title, col2 = myDataff$Conversation) %>% 
  do(na.locf(.))
write_csv(newDff, "dataForPyth.csv")
```
Merge all conversations/comments and a title:

Python 3:
```
import csv
import pandas as pd
import numpy as np

newDF = pd.DataFrame()
tit = ""
com = ""
rows_list = []
title_list = []
with open("/home/tatyana/dataForPyth.csv", "rt") as f:
    reader = csv.reader(f)
    for i, line in enumerate(reader):
        # print ('line[{}] = {}'.format(i, line))
        if i == 0:
            continue
        if i == 1:
            title_list.append(line[0])
            tit = line[0]
            com = line[0] + " " + line[1]
            continue
            
        if line[0] == tit:
            com = com + " " + line[1]
        else:
            rows_list.append(com)
            tit = line[0]
            title_list.append(line[0])
            com = line[0] + " " + line[1]

rows_list.append(com)

df = pd.DataFrame(rows_list)
se = pd.Series(title_list)
df['Topic'] = se.values

# print(title_list[84627])
# print(rows_list[84627])


df.to_csv("newRawAllData.csv",index=False, header=False) 

```
Login to the Cedar and prepare the enviroment and install python packages/module based on instructions in (`Main set up' section):

https://github.com/tmozgach/ent_ob/issues/8

Trasfer a `newRawAllData.csv` file to Cedar, for example:
```
scp newRawAllData.csv tmozgach@cedar.computecanada.ca:/home/tmozgach/scratch/TM
```
In Cedar, create a job filr for example, TM_job.sh:
```
#!/bin/bash
#SBATCH --time=11:59:00
#SBATCH --account=def-emodata
#SBATCH --mem=50000M
if [[ $1 -eq 0 ]] ; then
   echo 'You did not specify number of topics. Example: sbatch TM_job.sh 30'
   exit 1
fi
source ~/virtualenvironment/bin/activate
echo 'Number of topics:' $1
python ./TopicModeling.py $1
echo 'I finished'
```
In Cedar, create a python file TopicModeling.py:
```
# python -m pip install <name of packages/library>
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
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

import textblob
from textblob import TextBlob

import sys

n_topics = sys.argv[1]


# Function to remove stop words from sentences, punctuation & lemmatize words. 
def clean(doc):
    exclude = set(string.punctuation)
    translate_table = dict((ord(char), None) for char in string.punctuation)
    no_punct = doc.lower().translate(translate_table)
    stop_free = " ".join([i for i in no_punct.split() if i not in stop])
    blob = TextBlob(stop_free)
    singles = " ".join([word.singularize() for word in blob.words])
    normalized = " ".join(lemma.lemmatize(word,'v') for word in singles.split())    
    x = normalized.split()
    y = [s for s in x if len(s) > 2]
    return y
    
rawPostall = pd.read_csv("newRawAllData.csv", names = ['text','title'], nrows = 100)
rawPost = pd.DataFrame()
rawPost['text'] = rawPostall['text']

# Cleaning 
stop = set(stopwords.words('english'))
lemma = WordNetLemmatizer()

# Delete URL, website etc.
rawPost['text'] = rawPost['text'].replace(r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))""",' ', regex=True)

cleanPost = [clean(doc) for doc in rawPost['text']]

# Find the most frequent words and exclude NEUtral them. My bias!!! May be work more on that?
import itertools
flattened_cleanPost = list(itertools.chain(*cleanPost))

from collections import Counter
word_counts = Counter(flattened_cleanPost)
top_200 = word_counts.most_common(200)
print('*************************')
print('The most frequent 200 words:')
print(top_200)

# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(cleanPost)

#After printing the most frequent words of the dictionary, I found that few words which are mostly content neutral words are also present in the dictionary. 
# These words may lead to modeling of “word distribution”(topic) which is neutral and do not capture any theme or content. 
# I made a list of such words and filtered all such words.
stoplist = set('awesome cant though theyre yeah around try enough keep way start work busines isnt theyre didnt doesnt i\'ve you\'re that\'s what\'s let\'s i\'d you\'ll aren\'t \"the i\'ll we\'re wont 009 don\'t it\'s nbsp i\'m get make like would want dont\' use one need know good take thank say also see really could much something ive well give first even great things come thats sure help youre lot someone ask best many question etc better still put might actually let love may tell every maybe always never probably anything cant\' doesnt\' ill already able anyone since another theres everything without didn\'t isn\'t youll\' per else ive get would like want hey might may without also make want put etc actually else far definitely youll\' didnt\' isnt\' theres since able maybe without may suggestedsort never isredditmediadomain userreports far appreciate next think know need look please one null take dont dont\' want\' could able ask well best someone sure lot thank also anyone really something give years use make all ago people know many call include part find become'.split())
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
dictionary.filter_tokens(stop_ids)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.

doc_term_matrix = [dictionary.doc2bow(doc) for doc in cleanPost]

print('Training...')
#Creating the object for LDA model using gensim library & Training LDA model on the document term matrix.

ldamodel = Lda(doc_term_matrix, num_topics = int(n_topics), id2word = dictionary, passes=20, iterations=1000)

print('Training finished.')

# Label a topic to each THREAD.
def label_comment(ldamodel,doc_term_matrix, dictionary):
    # Assigns the topics to the documents in corpus
    lda_corpus = ldamodel[doc_term_matrix]
    doc_topics = ldamodel.get_document_topics(doc_term_matrix)
    se = pd.Series(doc_topics)
    rawPostall['Topic/Probability'] = se.values

    main_topics = []
    main_probability = []

    for k,topics in enumerate(doc_topics):
        if topics:
            topics.sort(key = itemgetter(1), reverse=True)
            main_topics.append(topics[0][0])
            main_probability.append(topics[0][1])

    se1 = pd.Series(main_topics)
    rawPostall['Main Topic'] = se1.values

    se2 = pd.Series(main_probability)
    rawPostall['Main Probability'] = se2.values

    rawPostall.to_csv("LabeledTopic" + (n_topics) + ".csv")

def visualize(ldamodel,doc_term_matrix, dictionary):
    import pyLDAvis
    try:
        pyLDAvis.enable_notebook()
    except:
        print ('not in jupyter notebook')
        
    viz = pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary)
    
    pyLDAvis.save_html(viz, 'TM_viz' + str(n_topics) + '.html')
    
    return viz

print('Vizualization...')
visualize(ldamodel,doc_term_matrix, dictionary)
print('Vizualization finished')

print('Labeling...')
label_comment(ldamodel,doc_term_matrix, dictionary)
print('Labeling finished')

#save model for future usage
print('Saving model...')
ldamodel.save('./TM_lda' + str(n_topics) + '.model')

# How to load model back
# loading = gensim.models.ldamodel.load(path)
# ldamodel=loading

```
In Cedar, submit a job, the second parameter is the number of topics.

```
sbatch TM_job.sh 30
```
Your visualization is: `TM_viz*.html`

Labeling Threads (comments):

Transfer `LabeledTopic.csv` back to the laptop from Cedar (You can do everything in Cedar if you want!!!):
```
scp tmozgach@cedar.computecanada.ca:/home/tmozgach/scratch/TM/LabeledTopic.csv .
```
Merge Labeled comments with the initial table.

R:
```
library(tidyverse)
library(zoo)
library(dplyr)

myDataf = read_delim("/home/tatyana/Downloads/data_full.csv", delim = ',' )
myDataff = myDataf[!is.na(strptime(myDataf$Date,format="%Y-%m-%d %H:%M:%S")),]

# There are title that duplicates another one. Titles are not unique
myDataff$Title <- make.unique(as.character(myDataff$Title), sep = "___-___")

# make.uniqui makes also NA - unique by adding number, need to transform them back to NA
myDataff$Title <- gsub("NA__+", NA, myDataff$Title)

# change NA by previous Title
myDataff['Title2'] = data.frame(col1 = myDataff$Title, col2 = myDataff$Conversation) %>% 
  do(na.locf(.))

myLabDataf = read_delim("/home/tatyana/nlp/LabeledTopic.csv", delim = ',' )

# 8 thredshad some issue and weren't mereged
newm = merge(myDataff,myLabDataf, by.x = 'Title2', by.y = 'title')

fin = select(newm, Date, Sender, Title2, Replier, Conversation,`Points from this question`, `Post Karma`, `Comment Karma`, `Date joining the forum;Category Label`, `Topic/Probability`, `Main Topic`, `Main Probability`)

write_csv(fin, "final_topic.csv")

```
Final file is: `final_topic.csv`
