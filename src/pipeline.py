import warnings
warnings.filterwarnings("ignore")

import re
from datetime import datetime
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd
import numpy as np

from src.transformers import ModelTransformer, TfIdfLen, MatchPattern, Length, Converter
from src.config import data_dir
from src.helpers import print_dict, save_model, load_model, calc_metrics

PATTERNS = [(r"[\(\d][\d\s\(\)-]{8,15}\d", {"name": "phone",
                                            "is_len": 0}),
           (r"%|taxi|скидк|цін", {"name": "custom",
                                  "is_len": 0,
                                  "flags": re.I | re.U})
           ]
NAMES = ["logit", "nb"]
DATAFILE = 'sms-uk-total.xlsx'
TF_PARAMS = {"lowercase": True,
             "analyzer": "char_wb",
             "stop_words": None,
             "ngram_range": (4, 4),
             "min_df": 0.0,
             "max_df": 1.0,
             "preprocessor": None,
             "max_features": 4000,
             "norm": "l2"*0,
             "use_idf": 1
             }

def build_ensemble(model_list, estimator=None):
    models = []
    for i, model in enumerate(model_list):
        models.append(('model_transform'+str(i), ModelTransformer(model)))

    if not estimator:
        return FeatureUnion(models)
    else:
        return Pipeline([
            ('features', FeatureUnion(models)),
            ('estimator', estimator)
            ])

def get_vec_pipe(add_len=True, tfidf_params={}):
    vectorizer = TfIdfLen(add_len, **tfidf_params)
    vec_pipe = [
        ('vec', vectorizer)]
    return Pipeline(vec_pipe)

def get_pattern_pipe(patterns):
    pipes = []
    for i, (patt, params) in enumerate(patterns):
        kwargs = params.copy()
        name = kwargs.pop("name") + "_" + str(i)
        transformer = MatchPattern(pattern=patt, **kwargs)
        pipes.append((name, transformer))
    return pipes

def get_len_pipe(use_tfidf=True, vec_pipe=None):
    len_pipe = [("length", Length(use_tfidf))]
    if use_tfidf:
        len_pipe.insert(0, ("vec", vec_pipe))
    return Pipeline(len_pipe)

def build_transform_pipe(tf_params, add_len=True, vec_mode="add", patterns=PATTERNS):
    vec_pipe = get_vec_pipe(add_len, tf_params)
    if vec_mode == "only":
        return vec_pipe
    patt_pipe = get_pattern_pipe(patterns)
    chain = [
        ('converter', Converter()),
        ('union', FeatureUnion([
            ('vec', vec_pipe),
            *patt_pipe
        ]))
    ]
    return chain

def build_classifier(name, seed=25):
    if name == "logit":
        model = LogisticRegression(C=1, class_weight="balanced", random_state=seed, penalty="l2")
        model.grid_s = {f'{name}__C' : (0.1, 0.2, 0.3, 0.4, 0.5, 1, 5, 10)}
        model.grid_b = {f'{name}__C' : [(1)]}
    elif name == "nb":
        model = MultinomialNB(alpha=0.1) #class_prior=[0.5, 0.5])
        model.grid_s = {f'{name}__alpha' : (0.1, 0.5, 1, 5, 10)}
        model.grid_b = {f'{name}__alpha' : [(1)]}
    model.name = name
    return model

def get_estimator_pipe(name, model, tf_params, vec_mode="add", patterns=PATTERNS):
    chain = build_transform_pipe(tf_params, vec_mode=vec_mode, patterns=patterns)
    chain.append((name, model))
    pipe = Pipeline(chain)
    pipe.name = name
    return pipe

def get_all_classifiers(names):
    return [build_classifier(name) for name in names]

def build_all_pipes(tf_params, vec_mode="add", names=NAMES, patterns=PATTERNS):
    clfs = get_all_classifiers(names)
    return [get_estimator_pipe(clf.name, clf, tf_params, vec_mode, patterns=patterns) for clf in clfs]

def preprocess(data):
    data = data.loc[data.text.notnull()]
    data["text"] = data["text"].str.replace(r"[\n\r]+", "")
    return data

def load_data(filename=DATAFILE):
    data = pd.read_excel(data_dir / filename)
    data = preprocess(data)
    return data

def class_report(conf_mat):
    tp, fp, fn, tn = conf_mat.values.flatten()
    measures = {}
    measures['accuracy'] = (tp + tn) / (tp + fp + fn + tn)
    measures['specificity'] = tn / (tn + fp)        # (true negative rate)
    measures['recall'] = tp / (tp + fn)        # (recall, true positive rate)
    measures['precision'] = tp / (tp + fp)
    measures['f1'] = 2*tp / (2*tp + fp + fn)
    return measures

def grid_search(tf_params, filename=DATAFILE, random_state=25, vec_mode="all",
                n_splits=5, log=True, grid="grid_s", transformer_grid={},
                scoring="f1", estimator_names=NAMES, patterns=PATTERNS,
                n_jobs=-1):

    data = load_data(filename)
    X, y = data["text"], data["label"]
    cv_splitter = StratifiedKFold(n_splits=n_splits, random_state=random_state,
                                  shuffle=True)
    # Build pipelines
    pipes = build_all_pipes(tf_params, names=estimator_names, vec_mode=vec_mode, patterns=patterns)

    best = []
    best_scores = []
    for i, pipe in enumerate(pipes):
        if log:
            print(f"Hypertuning model {i+1} out of {len(pipes)}: {pipe.name}")
            print("================================================================================")

        current_grid = getattr(pipe.steps[-1][1], grid)
        current_grid.update(transformer_grid)
        gs = GridSearchCV(pipe, current_grid, scoring=scoring, cv=cv_splitter, n_jobs=n_jobs, verbose=False)
        model = gs.fit(X, y)

        if log:
            print(f"Best score on training set (CV): {gs.best_score_:0.3f}" )
            print("Best parameters set:")
            for params, mean_score, scores in gs.grid_scores_:
                print(f"{mean_score:0.4f} (+/-{scores.std() / 2:0.4f}) for {params}: {scores}")
        best.append(gs.best_estimator_)
        temp = [el for el in gs.grid_scores_ if el.parameters==gs.best_params_][0]
        best_scores.append({"params": temp[0], "mean": temp[1], "scores": temp[-1],
                            "std": temp[-1].std()})
    return best, best_scores

def analyze_model(model=None, modelfile=None, datafile=None, n_splits=5, random_state=25,
                  labels=["ham", "spam"], mode="binary", log_fold=True, log_total=True):
    if model is None:
        model = load_model(modelfile)

    data = load_data(datafile)
    X, y = data["text"], data["label"]

    cv_splitter = StratifiedKFold(n_splits=n_splits, random_state=random_state,
                                  shuffle=True)

    conf_matrix = None
    results = []
    fps = set()
    fns = set()
    scores = []
    for i, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if log_fold:
            print(f"\nFit fold {i+1} out of {n_splits}")
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        metrics, _, fold_conf_matrix = calc_metrics(y_test, pred, proba, labels=labels, print_=False, mode=mode)

        if conf_matrix is None:
            conf_matrix = fold_conf_matrix.copy()
        else:
            conf_matrix += fold_conf_matrix

        fp_i = np.where((pred==1) & (y_test==0))[0]
        fn_i = np.where((pred==0) & (y_test==1))[0]
        record = {"fold": i+1,
                  "conf_matrix": fold_conf_matrix,
                  "fp": test_idx[fp_i],
                  "fn": test_idx[fn_i]}

        results.append(record)
        scores.append(metrics)
        fps.update(test_idx[fp_i])
        fns.update(test_idx[fn_i])

        if log_fold:
            print("\nFold scores")
            print_dict(metrics)
            print("\nFold Confusion Matrix")
            print(fold_conf_matrix)
    conf_matrix /= n_splits
    scores = pd.DataFrame(scores)
    if log_total:
        print("\nOverall results")
        for mean, std in zip(scores.mean().to_dict().items(), scores.std().to_dict().items()):
            print(f"{mean[0]}: {mean[1]:0.2f} +/- {std[1]:0.4f}")
        print("\nAveraged confusion matrix")
        print(conf_matrix)
        print("\nMean metrics")
        print_dict(class_report(conf_matrix))
    return scores, results, conf_matrix, {"fn": sorted(fns), "fp": sorted(fps)}

if __name__ == "__main__":
    grid_tf = {}
    best_estimators, best_scores = grid_search(transformer_grid=grid_tf, tf_params=TF_PARAMS)
    scores, results, conf_matrix, fnp = analyze_model(model=best_estimators[0], datafile=DATAFILE)
