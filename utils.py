import contractions
import nltk
from string import punctuation
from autocorrect import Speller
from langdetect import detect
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, LancasterStemmer
from unidecode import unidecode
from nltk.tokenize import word_tokenize
from googletrans import Translator
import pickle
import json
from gensim.models import Word2Vec
import pandas as pd
import numpy as np

class restaurant_review():

    def __init__(self,data):
        self.data = data

    def loading_files(self):
        
        with open(r'artifacts/resau_svm_model.pkl','rb') as file:
            self.svm = pickle.load(file)


    def vectorizer(self,list_of_docs,model):
        self.feature = []
        for rew in list_of_docs :
            zero_vector = np.zeros(model.vector_size)
            vectors = []
            for word in rew :
                try :
                    word in model.wv 
                    vectors.append(model.wv[word])

                except KeyError:
                    continue
            if vectors :
                vectors = np.asarray(vectors)
                avg_vec = vectors.mean(axis = 0)
                self.feature.append(avg_vec)
            else :
                self.feature.append(zero_vector)
                
        return self.feature

    def cleaned_data(self):
            self.loading_files()
            
            user_data = self.data['html_data']

            stop_list = stopwords.words('english')
            stop_list.remove('no')
            stop_list.remove('nor')
            stop_list.remove('not')
            clean_text = user_data.replace('\\n',' ').replace('\t',' ').replace('\\',' ')
            clean_text = contractions.fix(clean_text)
            clean_text = unidecode(clean_text)
            clean_text = [word.lower() for word in word_tokenize(clean_text) if (word not in punctuation) and (word not in stop_list)
                        and (word.isalpha()) and ((len(word) > 2))]
            correct = Speller()(' '.join(clean_text))
            lemma = []
            for i in correct.split(' ') :
                l = WordNetLemmatizer().lemmatize(i)
                lemma.append(l)
            user_data = ' '.join(lemma)

            model1 = Word2Vec.load('artifacts/restau.model',user_data)
            user = pd.Series([user_data.split()])
            
            user_data1 = np.array(self.vectorizer(user.tolist(),model1))
            
            result = self.svm.predict(user_data1)

            return result

