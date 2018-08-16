import sys
from XmlToJson import xmlToJsonHighLevel
import numpy as np
import pandas as pd
import string
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.externals import joblib

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier


from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score,confusion_matrix, roc_auc_score,roc_curve,auc

from sklearn.pipeline import Pipeline



def GetJsonFromRecords(fileName):
    # xml to json to df
    jsonFileName = xmlToJsonHighLevel(fileName)
    df0 = pd.read_json(jsonFileName)
    return df0

def ColTransform(df0):
    df=df0.copy()
    df=df.T
    df=df.reset_index()
    #giving columns names
    df.columns = ['id', 'smoking_status','descrp']
    return df



def TextProcess(description):
    
    # stop words
    sw=['summaryunsigneddisreport','yregistration','amed','date','patient','mm', 'st','amdischarge','doctor', 'hospital',
    'surgery','pain','problem','discharge','admission','i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
    'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
    'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am','be',
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'of', 'at', 'by', 'for', 'about', 'against', 
    'between', 'into', 'through', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
    'under', 'further', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'more', 
    'most', 'other', 'some', 'such', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'just', 'don', "don't",
    'should', "should've", 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'ma', 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'won', "won't", 'wouldn', "wouldn't"
    'aboard','about','above','across','along','an','and','another','any','around','as','at','below','behind','below'
    ,'beneath','beside','beyond','certain','down','during','each','following','for','from','inside','into','its',
    'like','minus','my','near','next','opposite','outside','out','over','plus','round','so','some','than','through',
    'toward','underneath','unlike','yet','under']
    
    # removing punctuation
    nopunc=[char for char in description if char not in string.punctuation]
    nopunc=''.join(nopunc)
    
    #removing numbers within words
    splitnum  = re.split('(\d+)',nopunc)
    splitnum=' '.join(splitnum)
    
     #removing numbers from corpus
    new=re.sub(" \d+", " ", splitnum)
    
    #removing stop words
    swtext= [word.lower() for word in new.split() if word not in [x.upper() for x in sw]]
    text=' '.join(swtext)
    
    return text


def UnknownCol(column):
    #seperate column Unknown from known , first layer of classification
    if 'UNKNOWN' in column :
        return 1

    else: return 0
    


def Split(df,feature, label):
    X  = df[feature]
    y = df[label]
    # X_train,  X_test,  y_train, y_test  =  train_test_split(X,y,test_size=0.25, random_state=42)
    return  X, y    #X_train,  X_test,  y_train, y_test 



def ModelGB(X, y, tfidf_vect, classifier, val):
    tfidf_vect.fit(X)
    xtfidf=tfidf_vect.transform(X)


    classifier.fit(xtfidf, y)

    file_clf = "{}.GB.pkl".format(val)
    file_tfidf="{}.GB-TFIDF.pkl".format(val)
    return joblib.dump(classifier, file_clf) ,  joblib.dump(tfidf_vect, file_tfidf)

def ModelRF(X, y, tfidf_vect, classifier, val):
    tfidf_vect.fit(X)
    xtfidf=tfidf_vect.transform(X)


    classifier.fit(xtfidf, y)

    file_clf = "{}.RF.pkl".format(val)
    file_tfidf="{}.RF-TFIDF.pkl".format(val)
    return joblib.dump(classifier, file_clf) ,  joblib.dump(tfidf_vect, file_tfidf) 

def ModelLR(X, y,tfidf_vect, classifier, val):
 
    tfidf_vect.fit(X)
    xtfidf=tfidf_vect.transform(X)


    
    classifier.fit(xtfidf, y)

    file_clf = "{}.LR.pkl.".format(val)
    file_tfidf="{}.LR-TFIDF.pkl".format(val)
    return joblib.dump(classifier, file_clf) ,  joblib.dump(tfidf_vect, file_tfidf) 

def NonSmokerCol(column):
    # seperate colum non smoker from smoker(past and current) for layer 2 classification
    if 'NON-SMOKER' in column :
        return 1

    else: return 0

def SmokerCol(column):
    # seperate column for past smokers from current smokers, last layer of classification
    if 'PAST SMOKER' in column :
        return 1

    else :return 0





def main():
    # read test record from the command line``
    MedicalRecords = "smokers_surrogate_train_all_version2.xml"
    # debug: recordToTest = "name.xml"

    # get the json from sample record as json
    df = GetJsonFromRecords(MedicalRecords)
    df = ColTransform(df)
    
   

      # apply the unknown column seperatiion   
    df["smoking_unknown"] = df["smoking_status"].apply(UnknownCol)
    

   # preprocessing the description column 

    df["descrp"] = df["descrp"].apply(TextProcess)
    
    
    X1,y1= Split(df,'descrp','smoking_unknown')


    tfidf_vect1 = TfidfVectorizer(use_idf= False,sublinear_tf= True,norm= 'l1',
                             ngram_range= (1, 2),min_df= 3,max_features=7000,max_df=0.8)
    classifier1=GradientBoostingClassifier(min_samples_leaf= 1, min_samples_split= 20,max_features= 'auto',
                                    max_depth=1, learning_rate= 0.05,n_estimators=100,random_state=15325)
    val=1 # for keeping count of the pickle file
    
    
    ModelLR(X1,y1,tfidf_vect1,classifier1, val)
    # Model_LR(X1,y1) # incase wanted to use other models for this layer 
    
    # Model_RF(X1,y1)
    
    
    # second layer of classification 
    # seperate the unknowns from the rest .perform analysis on the subset of known data points
    secondf=df[df.smoking_unknown==0]
    
    
    secondf["smoking_nonsmoker"] = secondf["smoking_status"].apply(NonSmokerCol)
    X2,y2= Split(secondf,'descrp','smoking_nonsmoker')

    tfidf_vect2 = TfidfVectorizer(use_idf= False,sublinear_tf= True,norm= 'l1',
                             ngram_range= (1, 2),min_df= 4,max_features=7000,max_df=0.8)

    classifier2=GradientBoostingClassifier(min_samples_leaf= 1,min_samples_split= 20,max_features= 'auto',
                                    max_depth=1, learning_rate= 0.05,n_estimators=100,random_state=15325)
    val=2
    
    ModelGB(X2,y2,tfidf_vect2,classifier2, val)


    # final layer , subset of data which has records tagged as current of past smoker only 
    
    thirdf=secondf[secondf.smoking_nonsmoker==0]
    thirdf["smoking_past"] = thirdf["smoking_status"].apply(SmokerCol)
    
    X3,y3= Split(thirdf,'descrp','smoking_past')

    tfidf_vect3 = TfidfVectorizer(use_idf= True,norm= 'l2',ngram_range= (2, 5),
                             min_df= 2,max_features= 8000,max_df=0.9)
    classifier3= RandomForestClassifier(n_estimators= 1000,min_samples_split=3,min_samples_leaf= 1,
                                             random_state=1,max_features =0.2,max_depth= 40)
    val=3
    ModelGB(X3,y3,tfidf_vect3,classifier3, val)
    # print output based on Prediction
    
    print("End")

# Run the main function
if __name__ == "__main__":
    main()
