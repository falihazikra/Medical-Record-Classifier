# WHAT TYPE OF SMOKER ARE YOU ??

Presentation : https://prezi.com/view/Bnt4IPCaFg6IosC0f6Mr/

## Index
**Notebooks** contains:

"Layer 1" : Jupyter notebook for EDA and first level of classification

"Layer 2" : Jupyter notebook for second level of classification

"Layer 3" : Jupyter notebook for third level of classification

"Final Confusion Matrix" : Jupyter notebook containing code for the final confusion matrix

**Code** contains:

Train.py - code for training and pickling your model.

Pipeline.py - code to classify a new document.

Helper.py - helper code

Plot.py - code used for plotting

XmltoJson.py - code to convert xml file to json
ALso contains pickled trained models for the 3 layers

demo.xml - Demo record 

### Introduction
Automated document classification can be a powerful technique to aid doctors and biomedical researchers by reducing the human effort needed to make repeated decisions categorizing samples of text. **My goal was to build a model that would classify hospital discharge summaries of patients based on their smoker status.** This categorization can help populate computerized electronic records, report summarization for physicians, assist those involved in targeted medical research and/or medical rehabilitation programs.

Hospital summaries contain a wealth of data about diagnoses, medications, procedures, etc., expressed primarily as narrative text. The narratives do not contain controlled vocabularies, and thus allow doctors flexibility of expression and are therefore in the form of fragmented English free text, showing the characteristics of a clinical sublanguage. Moreover, several clinical terms have multiple meaning such as 'discharge’ can signify either bodily excretion or release from a hospital; ‘cold’ can refer to a disease, a temperature sensation, or an environmental condition which makes their linguistic processing, search, and retrieval even more challenging.

### About the Data
I got the data from i2b2 (https://www.i2b2.org) 2006 Deidentification and Smoking NLP Challenge Dataset.

There are 502 de-identified patient discharge summaries containing information like past medical history, family history, medications, primary diagnosis, secondary diagnosis etc.

Human experts annotated each record with the smoking status of patients based on the explicitly stated smoking-related facts in the records and their medical intuitions on all information in the records.

There are five smoking categories: PAST SMOKER, CURRENT SMOKER, SMOKER, NON-SMOKER, and UNKNOWN.

1. A **Past Smoker** is a patient whose discharge summary asserts explicitly that the patient was a smoker one year or more ago but who has not smoked for at least one year.
2. A **Current Smoker** is a patient whose discharge summary asserts explicitly that the patient was a smoker within the past year.
3. A **Smoker** is a patient who is either a Current or a Past Smoker but whose medical record does not provide enough information to classify the patient as either.
4. A **Non-Smoker’s** discharge summary indicates that they never smoked.
5. An **Unknown** is a patient whose discharge summary does not mention anything about smoking.

**For the purpose of this project, I merged Smoker with Current Smoker.**

The data was in an XML file with containing ID, STATUS and TEXT tag. The TEXT tag contained all the information about the patient like history of present illness, medications, procedures etc.

**For this project I used the TEXT tag as a whole instead of splitting it into different columns**, because different records have different formats and column name making ti very hard to merge them.

### Modeling Approach
![Flowchart%20for%20classification.jpg](https://github.com/falihazikra/Medical-Record-Classifier/blob/capstone/images/Flowchart%20for%20classification.jpg)

Instead of approaching this problem as a multiclass classification , I decided to approach it as a multilevel one. This solves the problem of imbalanced classes and gives me more control at each level of classification. Boxes in yellow are the categories and the other depict the models.

**Model 1** classified documents based as known if they had any text that could analysed to determine the patients status. 'Known' documents contained words from the smoking corpus.

**Model 2** ,using negation, bi- or trigrams classified documents based on whether they smoke or never have smoked.

**Model 3** further classifies the document based on whether they smoke in the present or used to smoke in the past.

(See notebook 'Layer 1' for correlated unigrams and bigrams)


### Model Selection
         Layer 1                                  Layer 2                                      Layer 3

![image.png](https://github.com/falihazikra/Medical-Record-Classifier/blob/capstone/images/Model%20Comparison.png)

The metric I chose to measure my model performance was **ROC-AUC** score since I wanted to my model to be able to differentiate between the two classes.

I chose Logistic Regression for Layer 1 classification with ROC-AUC score of 0.98 , Gradient Boosting for the second (with ROC-AUC score of 0.85) and third layer (with ROC-AUC score of 0.76).

Here is the confusion matrix for the 4 classes on my test set :

<img src="https://github.com/falihazikra/Medical-Record-Classifier/blob/capstone/images/Confusion%20matrix.png"  width="500" height="500">



### Demo
![image](https://github.com/falihazikra/Medical-Record-Classifier/blob/capstone/images/Demo.png)

I used the 'demo' file in the code folder and passed it to the Pipeline.py .The model was able to tag this record as a past smoker.

### Next steps
Seperate the different fields in the TEXT tag into Principal Diagnosis, Medication, History of present illness, Family history, Operations, Hospital Course, Discharge instructions and medication, Lab reports .

Build a more comprehensive stop word list and normalize the features using the National Library of Medicine’s Lexical Variant Generation library (LVG). (http://SPECIALIST.nlm.nih.gov)
