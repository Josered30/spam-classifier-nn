import pandas as pd
import numpy as np
import nltk
import re
import datetime
import math
import matplotlib.pyplot as plt
import pickle 
import copy

from heapq import nlargest
from nltk.corpus import stopwords 
from collections import OrderedDict
from itertools import islice
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

class DataUtil:
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    @staticmethod
    def normalize_data(message):
        message = re.sub(r"\$[\d]+",'price',message)
        message = re.sub(r"\%[\d]+",'percentage',message)
        message = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",'url',message)
        message = re.sub(r"www.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",'url',message)
        message = re.sub(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)",'email',message)
        message = re.sub(r'[\W\s\d]',' ',message)
        return message

    @staticmethod
    def clean_data(message):
        message = message.lower() 
        message = DataUtil.normalize_data(message)
        words = nltk.word_tokenize(message)

        result = []
        for word in words:
            if word not in DataUtil.stop_words and len(word)>2:   
                #words = DataUtil.stemmer.stem(words[i])
                word = DataUtil.lemmatizer.lemmatize(word)
                result.append(word)  
        return result

    @staticmethod
    def order_and_take(data, key, n=None):
        data = OrderedDict(sorted(data.items(), key=lambda i: i[1][key], reverse=True))
        if n!=None:
            data = dict(islice(data.items(), n))
        return data


class DocumentReader:
    def __init__(self, document):
        self.document = document
        self.words_data = {}

    def get_words(self):
        df = pd.read_csv(self.document)
        words_list = dict()

        for index, row in df.iterrows():
            words = DataUtil.clean_data(row['message'])
            for word in words: 
                if word not in words_list.keys():
                    words_list[word] = 1
                else:
                    words_list[word] += 1
        
        result = { key:val for key, val in words_list.items() if val > 10}
        result = nlargest(3000, result, key=result.get)
        return result


class Data:
    @staticmethod
    def tf(sentences):    
        words_counter = {}
        for index, sentence in enumerate(sentences):
            words = DataUtil.clean_data(sentence)
            for word in words: 
                    if word not in words_counter.keys(): 
                        words_counter[word] = {}
                        words_counter[word]['sentences'] = {}
                    if index not in words_counter[word]['sentences'].keys():
                        words_counter[word]['sentences'][index] = 1/len(words)         
                    else:
                        words_counter[word]['sentences'][index] += 1/len(words)
        return words_counter

    @staticmethod
    def tf_idf(message):
        sentences = nltk.sent_tokenize(message)
        words_count = Data.tf(sentences)
        words_data = {}

        for key, element in words_count.items():
            words_data[key] = [0 for i in range(len(sentences))]
            idf = math.log(len(sentences)/len(element['sentences']))
            for index, sentence_ratio in element['sentences'].items():    
                words_data[key][index] = sentence_ratio * idf
        return words_data
          
    @staticmethod
    def get_inputs_count(message, words_list):  
        words = DataUtil.clean_data(message)  
        inputs = np.zeros(len(words_list))

        for index, word in enumerate(words_list):
            if word in words:
                inputs[index] +=1 
        return inputs

    @staticmethod
    def load_unique_words(dataframe):
        unique_words = {}
        for index,row in dataframe.iterrows():
            unique_words[row['word']] = 0
        return unique_words

