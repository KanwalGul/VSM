#Author : Kanwal Gul
import sys
import pickle
import os.path
import nltk
import pandas as pd
import numpy as np
import math
from nltk.stem import WordNetLemmatizer


lemmitizer = WordNetLemmatizer()

#initialize global variables
term_freq = {}                                #Term Frequency
stop = []                                     #stopwords
directory = os.path.dirname(os.path.abspath(__file__)) + '\ShortStories'  #location of text files
Universalset = []                              # contain list of ids of all text files
tfidf = {}                                     #term frequency * inverse documentfrequncy
df = {}                                        #document frequency                

#creates a list of stop words from file
def getstopwords():
    stop = []
    try:
        f_stop = open('Stopword.txt','r',encoding='utf-8')
        
        stopwords = str(f_stop.read().lower())
        stop = stopwords.splitlines()
        
        f_stop.close()
    except:
        print("No Stop Word Found check file name or continue without stopwords")

    
    return stop


#remove puctuation and split into tokens
#returns list of tokens
def tokenize(text):
    tokens = text.translate({ord(i): "" for i in "!@#$%^&*()[]{};:,./<>?\"'|——`~-=_’+”“"})
    tokens = tokens.split()
    return tokens

#read text drom given file and return tokens
def tokenizedoc(file):
    try:
        f = open (file,'r',encoding='utf-8')
    except:
        sys.exit('ShortStories Folder is not available or files are missing')
    try:
        text = str(f.read().lower())                #extracting tokens from file
        tokens = tokenize(text)
    
    finally:
        f.close()

    return tokens 

#find tf and df value and index them
def indexer(tokens,i):
    
    #print(stop)

    for pos, word in enumerate(tokens): #go through all tokens
        new = True
                 
        if word not in stop:
            
            word = lemmitizer.lemmatize(word)               #lemmatize the word
            
            if word not in df:
                df[word] = []                        #document Frequency
            if i not in df[word]:
                df[word].append(i)
                

            if i not in term_freq:
                term_freq[i] = {}           
            if word not in term_freq[i]:
                term_freq[i][word] = 1           #term frequency
            term_freq[i][word] += 1     
        



            
#Creates vocabulary index  
def make_index():
    
    i = 1                                           #document id
    
    for i in range(1,51):                           #loop through all documents
        file = directory+'/'+str(i)+'.txt'          #file directory
        tokens = tokenizedoc(file)                  #tokenize file
        indexer(tokens,i)                      #index file or add tokens to positional index

#calculating tfidf of documents
def tf_idf():
    N = 50
    print("Length")
    print(N)
    for key in term_freq.keys():
        for word in df.keys():
            if key not in tfidf:
                tfidf[key] = []
            temp = round((math.log10(len(df[word]))/N)*(term_freq[key].get(word,0)),6)
            #print(temp)
            tfidf[key].append(temp)

#calculate cosine similarity and return its value
def cosinesimilarity(queryvc,doc):
    temp = queryvc*doc
    temps =  np.sum(temp)
    y = np.sqrt(queryvc.dot(queryvc)) * np.sqrt(doc.dot(doc))
    
    return (temps)/(y )  

#find relevance score between query and docs and filter according to value of alpha
def getresult(query,alpha = 0.005):
    querylst = []
    N = 50
    for word in df.keys():
        qidf = (math.log10(len(df[word]))/N)
        qtf = query.count(word)
        qtfidf = round(qidf*qtf,6)
        querylst.append(qtfidf)
    
    queryvc = np.array(querylst)
    similarities = []
    keys = []
    for key in tfidf:
        docvc = np.array(tfidf[key])
        temp = cosinesimilarity(queryvc,docvc)
        if temp > alpha:
            similarities.append(temp)
            keys.append(key)
    result = [x for _,x in sorted(zip(similarities,keys),reverse= True)]
    print(result)
    #print("Length: " , len(result))
    




#AT start of program get list of all stopwords
print('Welcome')
stop = getstopwords()                     



#Check for positional index
if os.path.isfile('tfidf') and os.path.isfile('df'):               #if postional index exist on drive
    fpr = open('tfidf','rb')
    fpa = open('df','rb')
    try:
        tfidf = pickle.load(fpr)            #load it to memory
        df = pickle.load(fpa)
    except:
        sys.exit('Error in loading tfidf from drive check Directory')
    finally:
        fpr.close()
        fpa.close()

else:                                                #else make positional index
    print("Creating tfidf....")
    make_index()
    tf_idf()
    #saving dictionary in pickle object
    fpw = open('tfidf','wb')
    fpb = open('df','wb')
    try:                                             #save it to drive
        pickle.dump(tfidf, fpw ,protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(df, fpb ,protocol=pickle.HIGHEST_PROTOCOL)
    except:
        print("Could not save tfidf to drive")
       

    finally:
        fpw.close()
        fpb.close() 


for i in range(1, 51):                              #populate universalset with all docids
    Universalset.append(i)




#query processing
query = ''
alpha = float(input('Enter value of alpha '))
while 'exit..' not in query:

    query = input('Enter Query or type "exit.." to exit: ')
    
    if 'exit..' in query :
        break
    else:
        tokenizequery = tokenize(query.lower())
        lemmaquery = []
        for word in tokenizequery:
            if word not in stop:
                lemmaquery.append(lemmitizer.lemmatize(word))
        getresult(lemmaquery,alpha)

    
print("Exit Successfull")