import re
import json
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd
import numpy as np
from operator import itemgetter
from nltk.tokenize import word_tokenize
import sys

from src.transformers import (ModelTransformer, TfIdfLen, MatchPattern, Length, Converter,
                              TokenFeatures, Select)
from src.config import data_dir, models_dir, model_id, DATAFILE
from src.helpers import print_dict, save_model, load_model, calc_metrics, _to_int

CURRENCY_PATT = u"[$¢£¤¥֏؋৲৳৻૱௹฿៛\u20a0-\u20bd\ua838\ufdfc\ufe69\uff04\uffe0\uffe1\uffe5\uffe6]"
PATTERNS = [(r"[\(\d][\d\s\(\)-]{8,15}\d", {"name": "phone",
                                            "is_len": 0}),
            (r"%|taxi|скид(?:к|очн)|ц[іе]н|знижк", {"name": "custom",
                                                    "is_len": 0,
                                                    "flags": re.I | re.U}),
            (r"[.]", {"name": "dot", "is_len": 0}),
            (CURRENCY_PATT, {"name": "currency", "is_len": 0, "flags": re.U}),
            (r":\)|:\(|-_-|:p|:v|:\*|:o|B-\)|:’\(", {"name": "emoji", "is_len": 0, "flags": re.U}),
            (r"[0-9]{2,4}[.-/][0-9]{2,4}[.-/][0-9]{2,4}", {"name": "date", "is_len": 0})
            ]
NAMES = ["logit"]
TF_PARAMS = {"lowercase": True,
             "analyzer": "char_wb",
             "stop_words": None,
             "ngram_range": (4, 4),
             "min_df": 0.0,
             "max_df": 1.0,
             "preprocessor": None,
             "max_features": 4000,
             "norm": "l2" * 0,
             "use_idf": 1
             }
TOKEN_FEATURES = ["is_upper", "is_lower"]


def build_ensemble(model_list, estimator=None):
    models = []
    for i, model in enumerate(model_list):
        models.append(('model_transform' + str(i), ModelTransformer(model)))

    if not estimator:
        return FeatureUnion(models)
    else:
        return Pipeline([
            ('features', FeatureUnion(models)),
            ('estimator', estimator)
        ])


def get_vec_pipe(add_len=True, tfidf_params=None):
    if tfidf_params is None:
        tfidf_params = {}
    vectorizer = TfIdfLen(add_len, **tfidf_params)
    vec_pipe = [
        ('vec', vectorizer)]
    return Pipeline(vec_pipe)


def get_tokens_pipe(features=None):
    if features is None:
        features = TOKEN_FEATURES
    token_features = TokenFeatures(features=features)
    tok_pipe = [
        ("selector", Select(["tokens"], to_np=0)),
        ('tok', token_features)]
    return Pipeline(tok_pipe)


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


def build_transform_pipe(tf_params=None, add_len=True, vec_mode="add",
                         patterns=None, features=None):
    if features is None:
        features = TOKEN_FEATURES
    if patterns is None:
        patterns = PATTERNS
    if tf_params is None:
        tf_params = TF_PARAMS
    vec_pipe = get_vec_pipe(add_len, tf_params)
    if vec_mode == "only":
        return vec_pipe
    patt_pipe = get_pattern_pipe(patterns)
    chain = [
        ('selector', Select(["text"], to_np=0)),
        ('converter', Converter()),
        ('union', FeatureUnion([
            ('vec', vec_pipe),
            *patt_pipe
        ]))
    ]
    tok_pipe = get_tokens_pipe(features)
    final_chain = FeatureUnion([("chain", Pipeline(chain)),
                                ("tok", tok_pipe)])
    return [("final_chain", final_chain)]


def build_classifier(name, seed=25):
    model = None
    if name == "logit":
        model = LogisticRegression(C=1, class_weight="balanced", random_state=seed, penalty="l2")
        model.grid_s = {f'{name}__C': (0.1, 0.2, 0.3, 0.4, 0.5, 1, 5, 10)}
        model.grid_b = {f'{name}__C': [(1,)]}
    elif name == "nb":
        model = MultinomialNB(alpha=0.1)
        model.grid_s = {f'{name}__alpha': (0.1, 0.5, 1, 5, 10)}
        model.grid_b = {f'{name}__alpha': [(1,)]}
    if model is not None:
        model.name = name
    return model


def get_estimator_pipe(name, model, tf_params, vec_mode="add", patterns=None, features=None):
    if features is None:
        features = TOKEN_FEATURES
    if patterns is None:
        patterns = PATTERNS
    chain = build_transform_pipe(tf_params, vec_mode=vec_mode, patterns=patterns, features=features)
    chain.append((name, model))
    pipe = Pipeline(chain)
    pipe.name = name
    return pipe


def get_all_classifiers(names):
    return [build_classifier(name) for name in names]


def build_all_pipes(tf_params, vec_mode="add", names=None, patterns=None, features=None):
    if features is None:
        features = TOKEN_FEATURES
    if patterns is None:
        patterns = PATTERNS
    if names is None:
        names = NAMES
    clfs = get_all_classifiers(names)
    return [get_estimator_pipe(clf.name, clf, tf_params, vec_mode, patterns=patterns, features=features) for clf in
            clfs]


def preprocess(data):
    data = data.loc[data.text.notnull()]
    data["text"] = data["text"].str.replace(r"[\n\r]+", " ")
    data["tokens"] = data["text"].map(word_tokenize)
    return data


def load_data(filename=DATAFILE):
    data = pd.read_excel(data_dir / filename)
    data = preprocess(data)
    return data


def class_report(conf_mat):
    tp, fp, fn, tn = conf_mat.values.flatten()
    measures = dict()
    measures['accuracy'] = (tp + tn) / (tp + fp + fn + tn)
    measures['specificity'] = tn / (tn + fp)  # (true negative rate)
    measures['recall'] = tp / (tp + fn)  # (recall, true positive rate)
    measures['precision'] = tp / (tp + fp)
    measures['f1'] = 2 * tp / (2 * tp + fp + fn)
    return measures


def grid_search(tf_params=None, filename=DATAFILE, random_state=25, vec_mode="all",
                n_splits=5, log=True, grid="grid_s", transformer_grid=None,
                scoring="f1", estimator_names=None, patterns=None,
                n_jobs=-1, features=None):
    if transformer_grid is None:
        transformer_grid = {}
    if patterns is None:
        patterns = PATTERNS
    if features is None:
        features = TOKEN_FEATURES
    if estimator_names is None:
        estimator_names = NAMES
    if tf_params is None:
        tf_params = TF_PARAMS
    data = load_data(filename)
    X, y = data[["text", "tokens"]], data["label"]
    cv_splitter = StratifiedKFold(n_splits=n_splits, random_state=random_state,
                                  shuffle=True)
    # Build pipelines
    pipes = build_all_pipes(tf_params, names=estimator_names, vec_mode=vec_mode, patterns=patterns,
                            features=features)

    best = []
    best_scores = []
    for i, pipe in enumerate(pipes):
        if log:
            print(f"Hypertuning model {i + 1} out of {len(pipes)}: {pipe.name}")
            print("================================================================================")

        current_grid = getattr(pipe.steps[-1][1], grid)
        current_grid.update(transformer_grid)
        gs = GridSearchCV(pipe, current_grid, scoring=scoring, cv=cv_splitter, n_jobs=n_jobs, verbose=False)
        gs.fit(X, y)
        params = gs.cv_results_["params"]
        mean_scores = gs.cv_results_["mean_test_score"]
        std_scores = gs.cv_results_["std_test_score"]
        best_idx = gs.best_index_
        if log:
            print(f"Best score on training set (CV): {gs.best_score_:0.3f}")
            print("Best parameters set:")
            for param, mean_score, std in zip(params, mean_scores, std_scores):
                print(f"{mean_score:0.4f} (+/-{std / 2:0.4f}) for {params}")
        best.append(gs.best_estimator_)
        best_scores.append({"params": gs.best_params_,
                            "mean": mean_scores[best_idx],
                            "std": std_scores[best_idx],
                            "scoring": scoring})
    return best, best_scores


def analyze_model(model=None, modelfile=None, datafile=DATAFILE, n_splits=5, random_state=25,
                  labels=None, mode="binary", log_fold=True, log_total=True):
    if labels is None:
        labels = ["ham", "spam"]
    if model is None:
        model = load_model(modelfile)

    data = load_data(datafile)
    X, y = data[["text", "tokens"]], data["label"]

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
            print(f"\nFit fold {i + 1} out of {n_splits}")
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        metrics, _, fold_conf_matrix = calc_metrics(y_test, pred, proba, labels=labels, print_=False, mode=mode)

        if conf_matrix is None:
            conf_matrix = fold_conf_matrix.copy()
        else:
            conf_matrix += fold_conf_matrix

        fp_i = np.where((pred == 1) & (y_test == 0))[0]
        fn_i = np.where((pred == 0) & (y_test == 1))[0]
        record = {"fold": i + 1,
                  "conf_matrix": fold_conf_matrix.to_dict(),
                  "fp": test_idx[fp_i].tolist(),
                  "fn": test_idx[fn_i].tolist()}

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
    agg_scores = {}
    for mean, std in zip(scores.mean().to_dict().items(), scores.std().to_dict().items()):
        agg_scores[mean[0]] = {"mean": mean[1],
                               "std": std[1]}

    if log_total:
        print("\nOverall results")
        for mean, std in zip(scores.mean().to_dict().items(), scores.std().to_dict().items()):
            print(f"{mean[0]}: {mean[1]:0.2f} +/- {std[1]:0.4f}")
        print("\nAveraged confusion matrix")
        print(conf_matrix)
        print("\nMean metrics")
        print_dict(class_report(conf_matrix))
    return scores, agg_scores, results, conf_matrix, {"fn": _to_int(sorted(fns)),
                                                      "fp": _to_int(sorted(fps))}


def train(transformer_grid=None, tf_params=None, datafile=DATAFILE):
    if tf_params is None:
        tf_params = TF_PARAMS
    if transformer_grid is None:
        transformer_grid = {}
    best_estimators, best_scores = grid_search(transformer_grid=transformer_grid,
                                               tf_params=tf_params)
    idx, elem = max(enumerate(best_scores), key=lambda x: x[-1]["mean"])
    model = best_estimators[idx]
    params, mean, std, scoring = itemgetter("params", "mean", "std", "scoring")(best_scores[idx])
    scores, agg_scores, results, conf_matrix, fnp = analyze_model(model=model, datafile=datafile,
                                                                  log_fold=False)
    # Train on the whole dataset
    data = load_data(datafile)
    X, y = data[["text", "tokens"]], data["label"]
    model.fit(X, y)
    return model, params, agg_scores, scores.to_dict(orient="list"), results, conf_matrix, fnp


def construct_metadata(scores, agg_scores, params, fold_results, conf_matrix, fnp):
    output = {"agg_scores": agg_scores,
              "scores": scores,
              "params": params,
              "fold_results": fold_results,
              "mean_conf_matrix": {"matrix": conf_matrix.to_dict(),
                                   "metrics": class_report(conf_matrix)},
              "fnp": fnp}
    return output


def main(model_id=model_id):
    model, params, agg_scores, scores, fold_results, mean_conf_matrix, fnp = train()
    metadata = construct_metadata(scores, agg_scores, params, fold_results, mean_conf_matrix, fnp)
    save_model(model, model_id)
    with open(models_dir / f"{model_id}_metadata.json", "w+") as f:
        json.dump(metadata, f, indent=4)
    return 0


if __name__ == "__main__":
    code = main()
    sys.exit(code)
