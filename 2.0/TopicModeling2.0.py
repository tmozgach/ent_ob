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

n_file_name = sys.argv[2]

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
    
rawPostall = pd.read_csv(n_file_name, names = ['text'])
# The file name without extension
n_file_name = n_file_name.split(".")[0]
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

    rawPostall.to_csv("LabeledTopic2.0_" + (n_topics) + "_" + (n_file_name) +  ".csv")

def visualize(ldamodel,doc_term_matrix, dictionary):
    import pyLDAvis
    #try:
    #    pyLDAvis.enable_notebook()
    #except:
    #    print ('not in jupyter notebook')
        
    viz = pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary)
    
    pyLDAvis.save_html(viz, 'TM2.0_viz' + str(n_topics) + "_" + str(n_file_name) + '.html')
    
    return viz

print('Vizualization...')
visualize(ldamodel,doc_term_matrix, dictionary)
print('Vizualization finished')

print('Labeling...')
label_comment(ldamodel,doc_term_matrix, dictionary)
print('Labeling finished')

#save model for future usage
print('Saving model...')
ldamodel.save('./TM2.0_lda' + str(n_topics) + "_" + str(n_file_name) + '.model')

# How to load model back
# loading = gensim.models.ldamodel.load(path)
# ldamodel=loading

