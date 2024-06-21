from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skseq.sequences.sequence_list import SequenceList
from skseq.sequences.label_dictionary import LabelDictionary


def gen_set(path):
    df = pd.read_csv(path)
    df.dropna(axis=0, inplace=True)
    X = df.groupby('sentence_id')['words'].apply(list).values
    y = df.groupby('sentence_id')['tags'].apply(list).values

    #for i, sentence in enumerate(X):
    #    for word in sentence:
    #        if pd.isna(word):
    #            X = np.delete(X, i, 0)
    #            y = np.delete(y, i, 0)
    return X, y


def dictionary(sentences, tags):
    from collections import defaultdict

    word_dict = defaultdict(lambda: len(word_dict))  # Dictionary for unique words
    tag_dict = defaultdict(lambda: len(tag_dict))  # Dictionary for unique tags

    # Word dictionary. We go through the word in each sentence and if it isn't there, we add it
    for sentence in sentences:
        for word in sentence:
            word_dict[word]

    # Tag dictionary. Same as with words but it will be shorter (there are less tags)
    for tag_list in tags:
        for tag in tag_list:
            tag_dict[tag]

    word_dict = dict(word_dict)
    tag_dict = dict(tag_dict)
    rev_dict = {v: k for k, v in tag_dict.items()}

    return word_dict, tag_dict, rev_dict

def get_seq(word_dict,tag_dict, X, y):
    seq = SequenceList(LabelDictionary(word_dict), LabelDictionary(tag_dict))

    for i in range(len(X)):
        # Add the sequence (X[i], y[i]) to the sequence list
        seq.add_sequence(X[i], y[i], LabelDictionary(word_dict), LabelDictionary(tag_dict))

    return seq

def get_predictions(sp, X):
    y_hats = []
    for s in X:
        y_hats.append(sp.predict_tags_given_words(s))
    y_hats = np.concatenate(y_hats)
    return y_hats

def accuracy_score(y, y_hat, defeq=0):
    tptn = 0
    c = 0
    for y_true, y_pred in zip(y, y_hat):
        if y_true != defeq:
            if y_true == y_pred:
                tptn += 1
            c+=1
    return tptn/c

def cfn_matrix(y, y_hat, rev_dict):
    _y = []
    _y_hat = []
    for tag, tag2 in zip(y,y_hat):
        if tag != 0:
            _y.append(rev_dict[tag])
            _y_hat.append(rev_dict[tag2])
    
    tags = [rev_dict[tag] for tag in np.unique(np.concatenate((y, y_hat)))]
    tags_no = tags.copy()
    tags_no.remove("O")
    #_, ax = plt.subplots(figsize=(10, 10))
    #ConfusionMatrixDisplay.from_predictions(_y, _y_hat, labels=tags, ax=ax)
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=tags, yticklabels=tags, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def evaluate(y, y_hat, rev_dict):
    accuracy = accuracy_score(y, y_hat)
    accuracyO = accuracy_score(y, y_hat, "O")
    f1 = f1_score(y, y_hat, average='weighted')
    
    cfn_matrix(y, y_hat, rev_dict)
    return {
        "accuracy with O" : accuracyO,
        "accuracy" : accuracy,
        "f1_score" : f1
    }

def evaluate_integers(y, y_hat, dict2id, rev_dict):
    accuracy = accuracy_score(y, y_hat)
    accuracyO = accuracy_score(y, y_hat, dict2id["O"])
    f1 = f1_score(y, y_hat, average='weighted')
    
    cfn_matrix(y, y_hat, rev_dict)
    return {
        "accuracy with O" : accuracyO,
        "accuracy" : accuracy,
        "f1_score" : f1
    }
