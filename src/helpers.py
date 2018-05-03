from sklearn import metrics
import pandas as pd

def calc_metrics(y_test, pred, proba=None, labels=None, print_=True, mode="weighted"):
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
