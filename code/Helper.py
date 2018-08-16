from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score,confusion_matrix, roc_auc_score,roc_curve,auc

def train_model(classifier, feature_vector_train, label, feature_vector_test, test_label):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_test)
    
    
    #predict class probabilites
    prob=classifier.predict_proba(feature_vector_test)
    pred=prob[:,1]
    
    #print results
    
    print ("{} ".format(classifier))
    print ('Accuracy:',accuracy_score(predictions, test_label))
    print ('Recall:',recall_score(predictions, test_label),)
    print ('Precision:',precision_score(predictions, test_label))
    print ('F1:',f1_score(predictions, test_label))
    print ('ROC:',roc_auc_score(test_label,pred))
    print ('CM:',confusion_matrix(predictions, test_label))
    
    
    return classifier,accuracy_score(predictions, test_label), recall_score(predictions, test_label),precision_score(predictions, test_label), f1_score(predictions, test_label),roc_auc_score(test_label,pred), confusion_matrix(predictions, test_label),pred


