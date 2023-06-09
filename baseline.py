import common_utils
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np 

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

prediction_class = [
    'Obesity', 'Non.Adherence', 'Developmental.Delay.Retardation',
    'Advanced.Heart.Disease', 'Advanced.Lung.Disease',
    'Schizophrenia.and.other.Psychiatric.Disorders', 'Alcohol.Abuse',
    'Other.Substance.Abuse', 'Chronic.Pain.Fibromyalgia',
    'Chronic.Neurological.Dystrophies', 'Advanced.Cancer', 'Depression',
    'Dementia'
]

# prediction_class = ['Other.Substance.Abuse']


fitted_count_vector = {}
def fit_count_vectorizer(train_data, val_data, test_data, max_n_gram):
    # This will cache the result since the train, val and test data is the same
    if fitted_count_vector.get(max_n_gram) == None:
        cv = CountVectorizer(ngram_range=(1, max_n_gram),
                             tokenizer=TreebankWordTokenizer().tokenize)
        cv.fit(pd.concat([train_data['text'], val_data['text'], test_data['text']]))
        fitted_count_vector[max_n_gram] = (
            cv.transform(train_data['text']), 
            cv.transform(val_data['text']), 
            cv.transform(test_data['text']))
    return fitted_count_vector[max_n_gram]


def compute_eval(train_data, val_data, test_data, pred_class):
    print(pred_class)
    l = []
    for max_n_gram in range(1,6):
        train_cv, val_cv, test_cv = fit_count_vectorizer(
            train_data, val_data, test_data, max_n_gram)
        classifier = LogisticRegression(n_jobs=2, random_state=0)
        classifier.fit(train_cv, train_data[pred_class])
        test_pred = classifier.predict_proba(test_cv)[:, 1]
        precision, recall, thresholds = precision_recall_curve(test_data[pred_class], test_pred)
        f1_scores = 2*recall*precision/(recall+precision+1e-5)
        threshold = thresholds[np.argmax(f1_scores)]
        precision = (precision_score(test_data[pred_class], test_pred > threshold))   
        recall = recall_score(test_data[pred_class], test_pred > threshold)
        f1 = f1_score(test_data[pred_class], test_pred > threshold)
        roc_auc = (roc_auc_score(test_data[pred_class], test_pred))
        l.append((pred_class, max_n_gram, precision, recall, f1, roc_auc))
    return l
    

def main():
    # w2v_model = read_word2vec_model('mimiciii_word2vec.wordvectors')
    
    merged_data = common_utils.join_data_with_label(data, label)
    
    train_data, val_data, test_data = common_utils.train_test_val_split(merged_data, 0.7, 0.2, 0.1)
    result = []
    for pred_class in prediction_class:
        output = compute_eval(
                train_data, val_data, test_data, pred_class)
        result.extend(output)
    with open(r'./baseline/' + 'all_class' + '.txt' , 'w') as fp:
        for res in result:
            print(res)
            fp.write(functools.reduce(lambda x, y: str(x) + "," + str(y), res, '') + "\n")
    return result

data = common_utils.read_data('MIMIC-III.csv')
label = common_utils.read_label('annotation.csv')
result = main()
