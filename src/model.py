import random
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import log_loss

X = pd.read_csv("/data/march-madness/X.csv").drop(["Unnamed: 0"], axis=1)
Y = pd.read_csv("/data/march-madness/Y.csv")["0"]
X_test = pd.read_csv("/data/march-madness/X-test.csv").drop(["Unnamed: 0"], axis=1)

for i in range(int(X.shape[1] / 2)):
    X[str(i) + "_comb"] = X[str(2*i)] - X[str(2*i + 1)]
    X_test[str(i) + "_comb"] = X_test[str(2*i)] - X_test[str(2*i + 1)]

print(X.shape)
print(X_test.shape)

def run_grid_search(X, Y):
    model = AdaBoostClassifier()
    grid = dict()
    grid['n_estimators'] = [1000, 1500, 2000]
    grid['learning_rate'] = [0.00005, 0.0001, 0.00025, 0.0005]
    grid['base_estimator'] = [DecisionTreeClassifier(max_depth=3)]
    cv = RepeatedStratifiedKFold(n_splits=8, n_repeats=4, random_state=1)

    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring="neg_log_loss")
    grid_result = grid_search.fit(X, Y)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    n_scores = cross_val_score(model, X, Y, scoring="neg_log_loss", cv=cv, n_jobs=-1)

    return grid_result.best_params_

def fit_model(X, Y, params):
    model = AdaBoostClassifier(base_estimator=params['base_estimator'], learning_rate=params['learning_rate'],
            n_estimators=params['n_estimators'])
    cv = RepeatedStratifiedKFold(n_splits=8, n_repeats=4, random_state=1)

    scores = cross_val_score(model, X, Y, scoring="accuracy", cv=cv, n_jobs=-1)
    print("Accuracy: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))
    model.fit(X, Y)

    return model

def predict(X_test, model, title):
    preds = model.predict_proba(X_test)
    preds_frame = pd.read_csv("/data/march-madness/MSampleSubmissionStage2.csv")
    for i, row in preds_frame.iterrows():
        pred = preds[i][1]
        preds_frame.at[i, "Pred"] = pred

    preds_frame.to_csv("/data/march-madness/" + title + ".csv", index=False)

    return preds_frame

def bracket(preds, title):
    names = pd.read_csv("/data/march-madness/MTeams.csv")
    mapped = pd.DataFrame(np.zeros((len(preds), 2))).astype("object")
    mapped.columns = ["ID", "Pred"]
    for i, row in preds.iterrows():
        string = row["ID"]
        arr = string.split('_')
        t1 = names[names["TeamID"] == int(arr[1])]["TeamName"].to_numpy()[0]
        t2 = names[names["TeamID"] == int(arr[2])]["TeamName"].to_numpy()[0]
        mapped.at[i, "ID"] = str(t1 + " vs. " + t2)
        mapped.at[i, "Pred"] = row["Pred"]
    mapped.to_csv("/data/march-madness/" + title + ".csv", index=False)

#params = run_grid_search(X, Y)
#params = run_grid_search(X[["0", "1"]], Y)

params = {'n_estimators': 1500, 'learning_rate': 0.00025, 'base_estimator': DecisionTreeClassifier(max_depth=3)}

#model = fit_model(X, Y, params)
#seeds_model = fit_model(X[["0", "1", "0_comb"]], Y, params) 

#preds = predict(X_test, model, "MSubmissionStage2-1")
#bracket(preds, "MGamePreds-1")

for i, row in X.iterrows():
    for j in range(int(len(row) * 2 / 3)):
        X.iat[i, j] = np.log(X.iloc[i, j])

for i, row in X_test.iterrows():
    for j in range(int(len(row) * 2 / 3)):
        X_test.iat[i, j] = np.log(X_test.iloc[i, j])

for i in range(9):
    X[str(i) + "_comb"] = X[str(2*i)] - X[str(2*i + 1)]
    X_test[str(i) + "_comb"] = X_test[str(2*i)] - X_test[str(2*i + 1)]

model = fit_model(X, Y, params)

preds = predict(X_test, model, "MSubmissionStage2-2")
bracket(preds, "MGamePreds-2")
