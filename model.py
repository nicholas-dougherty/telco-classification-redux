# The usual modular suspects
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import math

# Of Mice & Machine Learning Mavericks
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay, mutual_info_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


def model_metrics(X_train, y_train, y_pred):
    '''
    get_metrics_binary takes in a classifier model and prints out metrics based on
    values in variables named X_train, y_train, and y_pred.
    
    return: a classification report as a transposed DataFrame
    '''
   # accuracy = classifier.score(X_train, y_train)
    class_report = pd.DataFrame(classification_report(y_train, y_pred, output_dict=True)).T
    conf = confusion_matrix(y_train, y_pred)
    tpr = conf[1][1] / conf[1].sum()
    fpr = conf[0][1] / conf[0].sum()
    tnr = conf[0][0] / conf[0].sum()
    fnr = conf[1][0] / conf[1].sum()
    
    tn, fp, fn, tp = conf.ravel()
    print(f'Number of true negatives  (tn) = {tn} \n The True Negative Rate (tnr) is: {tnr:.3} \n')
    print(f'Number of true positives  (tp) = {tp} \n The True Positive Rate (tpr) is:  {tpr:.3} \n')
    print(f'Number of false negatives (fn) = {fn} \n The False Negative Rate (tpr) is: {fnr:.3} \n')
    print(f'Number of false positives (fp) = {fp} \n The False Positive Rate (tpr) is: {fpr:.3} \n')
    print(f'''
    The True Positive Rate is {tpr:.3}, The False Positive Rate is {fpr:.3},
    The True Negative Rate is {tnr:.3}, and the False Negative Rate is {fnr:.3}
    ''')
    
    return class_report

def model_more(y_train, y_pred):
    pre = precision_score(y_train,y_pred)
    print(f' The precision is: {pre:.2%}')

    #recall
    rec = recall_score(y_train,y_pred)
    print(f' The recall rate is: {rec:.2%}')

    #f1-score
    f1 = f1_score(y_train,y_pred)
    print(f' The F1 score is: {f1:.2%}')

def baseline_model(X, y, strategy='most_frequent', random_state=123):
    '''
    Generates a baseline model using sklearn DummyClassifier strategy
    default set to 'most_frequent' and random_state=123
    '''

    # assign baseline model and fit to data
    baseline = DummyClassifier(strategy=strategy, random_state=random_state)
    baseline.fit(X, y)
    # assign baseline predictions
    y_baseline = baseline.predict(X)
    # print baseline accuracy score and first ten values for training data
    print(f'''
               Baseline Accuracy Score: {baseline.score(X, y):.2%}
        First Ten Baseline Predictions: {y_baseline[:10]}
        ''')

    return baseline, y_baseline

