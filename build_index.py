import pandas as pd
import numpy as np
import os
import io
import re
import time 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer,PorterStemmer
class Index:
    def __init__(self,directory=os.getcwd()):
        self.directory=directory
        self.all_words=[]
        self.inverted_index={}
        self.doc_sim_score={}
        self.all_files={}
        self.dict_list={}
        self.tfidf_query_doc=[]
        self.dict_lemm_or_stem={}
        self.ultimate_sim=[]
        self.stop_words = set(stopwords.words('english'))
        
    # Function to retrieve all the documents from the directory that is given as a input when a class Input object is initialized
    def retrieve_file(self,file_name=None,encoding="utf-8"):
        
        #print("lol\n")
        #print(self.directory)
        for _,filename in enumerate(os.listdir(self.directory)):
            if filename.endswith(".txt"):
                #print(filename)
                f = io.open(filename, mode="r", encoding=encoding)
                lines = f.read()
                self.all_files[filename.replace('.txt','')]=re.sub('[^a-zA-Z0-9]', ' ', lines.rstrip())
                #self.all_files is a dict that contains all the text present in a word doc in the format{"FILE_NAME":"TEXT IN THE FILE"}
        #print("LOL\n")
            
    
    def tok_lem_stem(self,type_op=None):
        #type_op can be a list also in the case when both lemmatization and stemming is to be applied 
        # Preprocessing the document and applying Case Folding, rudimernary normalization and spliting string to words and performing lemmitization of stemming based on user input
        self.operation=type_op
        for key in self.all_files.keys():
            lemmatizer=WordNetLemmatizer()
            stemmer=PorterStemmer()
            self.dict_list[key]=[]
            [self.dict_list[key].append(x) for x in self.all_files[key].lower().split(" ") if x not in self.stop_words if x is not '']
            # self.dict_list is a processed version of self.all_files with case folding done and stop words removed
            
            #To perform lemmatization of stemming based on user input
            for key in self.dict_list.keys():
                self.dict_lemm_or_stem[key]=[]
                if 'lemmatize' in type_op:
                    [self.dict_lemm_or_stem[key].append(lemmatizer.lemmatize(x)) for x in self.dict_list[key]]
                elif 'stemming' in type_op:
                    [self.dict_lemm_or_stem[key].append(stemmer.stem(x)) for x in self.dict_list[key]]

                self.dict_lemm_or_stem[key]=[x for x in self.dict_lemm_or_stem[key] if x is not '']
           
    
    def preprocess_query_doc(self,filename,encoding="utf-8"):
        
        #Preprocess the query document in the same order as the corpus was preprocessed
        
        f = io.open(filename, mode="r", encoding=encoding)
        lines = f.read()
        process_string=re.sub('[^a-zA-Z0-9]', ' ', lines.rstrip())
        string_updated=[x for x in process_string.lower().split(" ") if x not in self.stop_words if x is not '']
        lemmatizer=WordNetLemmatizer()
        stemmer=PorterStemmer()
        if 'lemmatize' in self.operation:
            result=[lemmatizer.lemmatize(x) for x in string_updated]
        elif 'stemming' in self.operation:
            result=[stemmer.stem(x) for x in string_updated]

        result=[x for x in result if x is not '']
        return result
    
    def inverted_index_constr(self):
        # Constructing Inverted Index of the format {"WORD1":[DOC1,DOC3,...],"WORD2":[...]}
        if self.operation:
            text_dict=self.dict_lemm_or_stem
        else:
            text_dict=self.dict_list
            
        for key in text_dict.keys():
            [self.all_words.append(x) for x in text_dict[key] if x not in self.all_words]
            
        for word in self.all_words:
            self.inverted_index[word]=[]
            [self.inverted_index[word].append(key) for key in self.all_files.keys() if word in self.dict_lemm_or_stem[key]]

    
    def calculate_tf_idf(self,test_file=None,encoding_test="utf-8"):
        
        #Calculate tf-idf score
        test_string=self.preprocess_query_doc(filename=test_file,encoding=encoding_test)
        for word in test_string:
            self.doc_sim_score[word]=[]
            try:
                doc_having_word=len(self.inverted_index[word])
            except:
                doc_having_word=0
            [self.doc_sim_score[word].append([doc,self.dict_lemm_or_stem[doc].count(word)/len(self.dict_lemm_or_stem[doc])*np.log(len(self.all_files)/(doc_having_word+1))]) for doc in self.all_files.keys()]

    def tfidf_of_query(self,data):
        
        for word in self.doc_sim_score.keys():
            x=self.doc_sim_score[word]
            for y in x:
                if(y[0]==data):
                    self.tfidf_query_doc.append(y[1]) 
