from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import dill

from src.config import models_dir

def init_dir(dir_name):
    """Recursively creates directory dir_name and doesn't raise exception if it exists"""
    if not dir_name.exists():
        Path(dir_name).mkdir(parents=True)

def save_model(model, filename):
    init_dir(models_dir)
    with open(models_dir / filename, "wb+") as f:
        dill.dump(model, f)

def load_model(filename):
    with open(models_dir / filename, "rb") as f:
        model = dill.load(f)
    return model

def print_dict(data):
    for key, value in data.items():
        print(f"{key}: {value:0.3f}")

def calc_metrics(y_test, pred, proba=None, labels=None, print_=True, mode="binary"):
    output = {}
    if proba is not None:
        roc_auc = metrics.roc_auc_score(y_test, proba)
        output["AUC"] = roc_auc
    output["Recall"] = metrics.recall_score(y_test, pred, average=mode)
    output["Precision"] = metrics.precision_score(y_test, pred, average=mode)
    output["F1"] = metrics.f1_score(y_test, pred, average=mode)
    output["Accuracy"] = metrics.accuracy_score(y_test, pred)
    if labels is not None:
        index = labels
        columns = ["pred_" + str(el) for el in index]
    else:
        columns = None
        index = None
    if not all(el in labels for el in y_test):
        labels = np.unique(y_test)
    conf_matrix = pd.DataFrame(metrics.confusion_matrix(y_test, pred, labels=labels),
                               columns=columns, index=index)
    report = metrics.classification_report(y_true=y_test, y_pred=pred, labels=labels)
    if print_:
        for key, value in output.items():
            print(f"{key}: {value:0.3f}")
        print("\nConfusion matrix:")
        print(conf_matrix)
        print("\nReport:")
        print(report)
    return output, report, conf_matrix

def top_tfidf_feats(row, features, top_n=25):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=25):
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs

def plot_tfidf_classfeats_h(dfs):
    fig = plt.figure(figsize=(12, 9), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=14)
        ax.set_title("label = " + str(df.label), fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center', color='#3F5D7D')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()
