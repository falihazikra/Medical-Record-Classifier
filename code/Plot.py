from Train import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score,confusion_matrix, roc_auc_score,roc_curve,auc

#Code for category counts
def CategoryCounts(df,column):
    a4_dims = (8, 5)

    fig, ax = plt.subplots(figsize=a4_dims)
    sns.set(style="darkgrid")
    mx=df[column].sort_values()
    sns.countplot(mx, data=df,order = df[column].value_counts().index)
    plt.title("Number of records per smoking category", fontsize=18)
    plt.ylabel('# of Occurrences', fontsize=14)
    plt.xlabel('Smoking Category', fontsize=14)
    plt.show()

# Code for ROC plot
def ROC_curve(classifier1,feature_vector_test,test_label):
    plt.figure(0).clf()
    fig, ax = plt.subplots(figsize=(8,6))
    plt.title('Receiver Operating Characteristic')


    prob=classifier1.predict_proba(feature_vector_test)
    pred=prob[:,1]
    fpr, tpr, thresh = roc_curve(test_label, pred)
    auc = roc_auc_score(test_label, pred)
    plt.plot(fpr,tpr,label="{}, auc=".format(classifier1)+str(auc))

    
    # prob2=classifier2.predict_proba(feature_vector_test)
    # pred2=prob[:,1]
    # fpr2, tpr2, thresh = roc_curve(test_label, pred2)
    # auc2 = roc_auc_score(test_label, pred2)
    # plt.plot(fpr,tpr,label="{}, auc=".format(classifier2)+str(auc2))

    
    # prob=classifier3.predict_proba(feature_vector_test)
    # pred=prob[:,1]
    # fpr, tpr, thresh = roc_curve(test_label, pred)
    # auc = roc_auc_score(test_label, pred)
    # plt.plot(fpr,tpr,label="{}, auc=".format(classifier3)+str(auc))

    
    # prob=classifier4.predict_proba(feature_vector_test)
    # pred=prob[:,1]
    # fpr, tpr, thresh = roc_curve(test_label, pred)
    # auc = roc_auc_score(test_label, pred)
    # plt.plot(fpr,tpr,label="{}, auc=".format(classifier4)+str(auc))

    plt.legend(loc = 'lower right')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    
    plt.show()  

# plot for confusion matrix-binary
def plot_confusion_matrix(cm):
    
    

    print(cm)

    # Show confusion matrix in a separate window
    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt='d',xticklabels=['Known','Unknown'],yticklabels=['Known','Unknown']) # change label names per layer
    plt.title('Confusion matrix')
    
    
    
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()



