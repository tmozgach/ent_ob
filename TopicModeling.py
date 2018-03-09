
# coding: utf-8

# In[54]:



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

# Function to remove stop words from sentences, punctuation & lemmatize words. 
def clean(doc):
    exclude = set(string.punctuation)
    translate_table = dict((ord(char), None) for char in string.punctuation)
    no_punct = doc.lower().translate(translate_table)
    stop_free = " ".join([i for i in no_punct.split() if i not in stop])
    normalized = " ".join(lemma.lemmatize(word,'v') for word in stop_free.split())
    
    x = normalized.split()
    y = [s for s in x if len(s) > 2]
    return y


# In[55]:


rawPost=pd.read_csv("TPostRawShort.csv", names = ['text'], sep = "\t")


# In[56]:


rawPost



# In[57]:


# Cleaning 
stop = set(stopwords.words('english'))
lemma = WordNetLemmatizer()

#There are \n\n in data above, need to delete them
rawPost['text'] = rawPost['text'].replace(r'\\n\\n|\\n',' ', regex=True)

# Delete URL, website etc.
rawPost['text'] = rawPost['text'].replace(r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))""",' ', regex=True)

cleanPost = [clean(doc) for doc in rawPost['text']]


# In[58]:


print(cleanPost) 



# In[59]:


# Find the most frequent words and exclude NEUtral them. My bias!!! May be work more on that?
import itertools
flattened_cleanPost = list(itertools.chain(*cleanPost))

from collections import Counter
word_counts = Counter(flattened_cleanPost)
top_three = word_counts.most_common(200)
print(top_three)


# In[60]:


# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(cleanPost)


# In[61]:


#After printing the most frequent words of the dictionary, I found that few words which are mostly content neutral words are also present in the dictionary. 
# These words may lead to modeling of “word distribution”(topic) which is neutral and do not capture any theme or content. 
# I made a list of such words and filtered all such words.
stoplist = set('get would like want hey might may without also make want put etc actually else far definitely youll\' didnt\' isnt\' theres since able maybe without may suggestedsort never isredditmediadomain userreports far appreciate next think know need look please one null take dont dont\' want\' could able ask well best someone sure lot thank also anyone really something give years use make all ago people know many call include part find become '.split())
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
dictionary.filter_tokens(stop_ids)


# In[62]:


# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in cleanPost]


# In[63]:


#Creating the object for LDA model using gensim library & Training LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=20, iterations=1000)
#ldafile = open('lda_model_sym_wiki.pkl','wb')
#cPickle.dump(ldamodel,ldafile)
#ldafile.close()


# In[64]:


#Print all the 50 topics
for topic in ldamodel.print_topics(num_topics=50, num_words=10):
    print (topic[0]+1, " ", topic[1],"\n")


# In[65]:


def visualize(ldamodel,doc_term_matrix, dictionary):
    import pyLDAvis
    try:
        pyLDAvis.enable_notebook()
    except:
        print ('not in jupyter notebook')
        

    viz = pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary)
    
    pyLDAvis.save_html(viz, 'TM_viz.html')
    
    return viz


# In[66]:


visualize(ldamodel,doc_term_matrix, dictionary)


# In[67]:


#save model for future usage
ldamodel.save('TM_lda.model')

# How to load model back
# loading = gensim.models.ldamodel.load(path)
# ldamodel=loading

