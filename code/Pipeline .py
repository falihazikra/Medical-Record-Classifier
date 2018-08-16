# Read xml file and product json for processing as dataframes

import sys
from XmlToJson import xmlToJsonHighLevel
import numpy as np
import pandas as pd
import string
import re
from Train import *
# Fetch data from xml and build a json for processing as
# data frame in python

def ColTransform(df0):
    df=df0.copy()
    df=df.T
    df=df.reset_index()
    #giving columns names
    df.columns = ['id', 'status','descrp']
    return df

def Predict(df):
    # if want to return list of predictions
    #predlist=[]
    
    #loading the pickled files 
    A_loaded_tfidf = joblib.load('1.LR-TFIDF.pkl')
    xtfidf=A_loaded_tfidf.transform(df.descrp)

    A_loaded_model = joblib.load('1.LR.pkl.')
    predictions1=A_loaded_model.predict(xtfidf)


   
    if predictions1 == 1:
        print( "The patient's Smoker status  is : Unknown")
        #predlist.append(0)
    B_loaded_tfidf = joblib.load('2.GB-TFIDF.pkl')
    xtfidf=B_loaded_tfidf.transform(df.descrp)

    B_loaded_model = joblib.load('2.GB.pkl')
    predictions2=B_loaded_model.predict(xtfidf)
    if predictions2 == 1:
        print( "The patient's Smoker status  is : Non-Smoker")
        #predlist.append(1)

    C_loaded_tfidf = joblib.load('3.GB-TFIDF.pkl')
    xtfidf=C_loaded_tfidf.transform(df.descrp)

    C_loaded_model = joblib.load('3.GB.pkl')
    predictions3=C_loaded_model.predict(xtfidf)
    if predictions3 == 1:
        print( "The patient's Smoker status  is : Past Smoker")
        #predlist.append(2)
    else:
        print( "The patient's Smoker status  is : Smoker")
        #predlist.append(3)
        
    # if want to return list of predictions
     #return predlist

        
        
def main():
    # read test record from the command line
    recordToTest = "unannotated_test.xml"
    # debug: recordToTest = "name.xml"
   
    # get the json from sample record as json
    df = GetJsonFromRecords(recordToTest)
    df1 = ColTransform(df)
    
    df1["descrp"] = df1["descrp"].apply(TextProcess)
    Predict(df1)

    # print output based on Prediction
    print("End")

# Run the main function
if __name__ == "__main__":
    
    main()


