import nltk
import csv
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from nltk.tag import StanfordNERTagger
import gensim
from gensim import corpora
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

##ps = PorterStemmer()
##train_text = state_union.raw("2005-GWBush.txt")
##custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import re 
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
regex_pattern = "#[\w]*"
def _remove_regex(input_text, regex_pattern):
    urls = re.finditer(regex_pattern, input_text) 
    for i in urls: 
        input_text = re.sub(i.group().strip(), '', input_text)
    return input_text

######## this lines for java path declearing for corenlp lib usage
######## ı use caseless entity recognizer, this includes caseless trained model 
java_path = "C:/Program Files/Java/jdk1.8.0_141/bin/java.exe"
os.environ['JAVAHOME'] = java_path
nltk.internals.config_java("C:/Program Files/Java/jdk1.8.0_141/bin/java.exe")
jar='D:/Users/user/AppData/Local/Programs/Python/Python35/Lib/site-packages/stanford-ner-2018-02-27/stanford-ner.jar'
    
model_filename='D:/Users/user/AppData/Local/Programs/Python/Python35/Lib/site-packages/stanford-ner-2018-02-27/classifiers/english.conll.4class.caseless.distsim.crf.ser.gz'
    
english_nertagger = StanfordNERTagger(path_to_jar=jar,model_filename=model_filename)

##########################################################################################################

def get_chunks(text,list_of_person_entities,list_of_location_entities,list_of_organization_entities):    
    
    for chunk in english_nertagger.tag(nltk.word_tokenize(text)):
        print(chunk)      
        
        if(chunk[1]=="PERSON"):
            found_person=0
            for i in list_of_person_entities:
                if(i[0]==chunk[0]):
                    i[1]=i[1]+1
                    found_person=1
                    
            if(found_person==0):
                list_of_person_entities.append([chunk[0],1])
        elif(chunk[1]=="LOCATION"):
            found_location=0
            for i in list_of_location_entities:
                if(i[0]==chunk[0]):
                    i[1]=i[1]+1
                    found_location=1
                    
            if(found_location==0):
                list_of_location_entities.append([chunk[0],1])
        elif(chunk[1]=="ORGANIZATION"):
            found_organization=0
            for i in list_of_organization_entities:
                if(i[0]==chunk[0]):
                    i[1]=i[1]+1
                    found_organization=1
                    
            if(found_organization==0):
                list_of_organization_entities.append([chunk[0],1])
            
      
            
     


texts=[]
reddit_ids=[]
subreddits=[]
metas=[]
times=[]
authors=[]
ups=[]
downs=[]
authorlinkkarma=[]
authorkarma=[]
authorisgold=[]
words_all=[]
words_with_arrays_of_strings=[]
st = PorterStemmer()
def clean(doc):
    stop_free = [i for i in doc.lower().split() if i not in stop] ## without stop words
    punc_free = [ch for ch in stop_free if ch not in exclude]       ## if there are punc clean them
    normalized = [lemma.lemmatize(word) for word in punc_free]## lemmatize good better best--->  good good good
    
    stemmed = [st.stem(word) for word in normalized]## lemmatize good better best--->  good good good
    return stemmed


doc_words=[]

with open('news_politics.csv') as csvfile:
    Reader = csv.reader(csvfile, delimiter=",")
    a=0
    
    list_of_person_entities=[]
    list_of_location_entities=[]
    list_of_organization_entities=[]
    for row in Reader:
        words_tokenized=[]
        if(a>0):
            if(len(row)==12):
                #print(row[1])
                if(row[1] and row[1]!= "deleted"):
                
                    #print(type(row[1]),"a")
                    
                    row[1]= _remove_regex(row[1], regex_pattern)
                    
                    if(a<1):
                        
                        get_chunks(row[1],list_of_person_entities,list_of_location_entities,list_of_organization_entities)  # for named entity  recognition list_of_names["name",value]

                    
                    doc_words.append(row[1])
                        
                    
                    words = word_tokenize(row[1])
                    
                    
                    texts.append(row[1])
                    reddit_ids.append(row[2])
                    subreddits.append(row[3])
                    metas.append(row[4])
                    times.append(row[5])
                    authors.append(row[6])
                    try:
                        ups.append(float(row[7]))
                        downs.append(float(row[8]))
                    except:
                        ups.append(float(1.0))
                        downs.append(float(1.0))
                        
                    authorlinkkarma.append(row[9])
                    authorkarma.append(row[10])
                    authorisgold.append(row[11])
        
        
        a=a+1
        if(a==0):
            break
        
     
corpus = [clean(doc) for doc in doc_words]
# cleaan words          LDA represents documents as mixtures of topics that spit out 
dictionary = corpora.Dictionary(corpus)             #                       words with certain probabilities. initialize with random topics 
doc_term_matrix = [dictionary.doc2bow(text) for text in corpus]#              and with iteration find correct classes
Lda = gensim.models.ldamodel.LdaModel
print("-")#lda model
ldamodel = Lda(doc_term_matrix, num_topics=5, id2word = dictionary, passes=1)#  passes= number of iterations
topics=ldamodel.print_topics(num_topics=5)#                                           optimal number of topics by using Kullback Leibler Divergence Score
for topic in topics:
    print(topic)

print(list_of_person_entities)#
print(list_of_location_entities)#
print(list_of_organization_entities)#

##### clustering only text documents not with ups and downs
vectorizer = TfidfVectorizer(stop_words='english')   # vectorize words         Transforms text to feature vectors that can be used as input to estimator.
X = vectorizer.fit_transform(doc_words)
                                          # without eng stop words      Tf-Idf scoring for words
num_of_clusters=4#
model = KMeans(n_clusters=num_of_clusters, init='k-means++', max_iter=1000, n_init=1)#
model.fit(X)#

print("Top terms per cluster:")#
order_centroids = model.cluster_centers_.argsort()[:, ::-1]  #
terms = vectorizer.get_feature_names()#
for i in range(num_of_clusters):#
    print("Cluster %d:" % i),#
    for ind in order_centroids[i, :10]:   #   #first 10 term in centres
        print(' %s' % terms[ind]),#
    

print("\n")
print("Prediction")

Y = vectorizer.transform([" obama is the best "])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["trump is the best "])
prediction = model.predict(Y)
print(prediction)

########################################







