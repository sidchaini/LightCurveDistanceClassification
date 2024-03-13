import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.feature_selection import (
    SequentialFeatureSelector,
)
from mlxtend.evaluate import feature_importance_permutation
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.utils.estimator_checks import check_estimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import matplotlib.ticker as ticker
import os
os.chdir("../../")
from pathlib import Path
import json

import sys

sys.path.append("scripts")

import utils
import distclassipy as dcpy

cd = dcpy.Distance()

with open("settings.txt") as f:
    settings_dict = json.load(f)
np.random.seed(settings_dict["seed_choice"])

classification_letter = "c"
classification_problem = settings_dict["classification_problem"][classification_letter]
classes_to_keep = settings_dict["classes_to_keep"][classification_letter]
results_subfolder = f"{classification_letter}. {classification_problem}"
sns_dict = settings_dict["sns_dict"]

sns.set_theme(**sns_dict)

# Load Data
X_df_FULL = pd.read_csv("data/X_df.csv", index_col=0)
y_df_FULL = pd.read_csv("data/y_df.csv", index_col=0)


# Remove features to be dropped from previous notebook
with open(os.path.join("results", results_subfolder, "drop_features.txt")) as f:
    bad_features = json.load(f)  # manually selected

X_df_FULL = X_df_FULL.drop(bad_features, axis=1)

# Keep only current classes
cl_keep_str = "_".join(classes_to_keep)

y_df = y_df_FULL[y_df_FULL["class"].isin(classes_to_keep)]
X_df = X_df_FULL.loc[y_df.index]
X = X_df.to_numpy()
y = y_df.to_numpy().ravel()

with open(os.path.join("results", results_subfolder, "best_common_features.txt")) as f:
    best_common_features = json.load(f)

all_metrics = settings_dict['all_metrics']


for metric in tqdm(all_metrics, desc="Metric", leave=True):

    metric_str = utils.get_metric_name(metric)
    print("*" * 20, metric_str, "*" * 20)
    
    lcdc1 = dcpy.DistanceMetricClassifier(
        metric=metric, scale=True, 
        central_stat=settings_dict["central_stat"], 
        dispersion_stat=settings_dict["dispersion_stat"],
        calculate_kde=False, calculate_1d_dist=False
    )
    
    lcdc2 = dcpy.DistanceMetricClassifier(
        metric=metric, scale=True, 
        central_stat=settings_dict["central_stat"], 
        dispersion_stat=settings_dict["dispersion_stat"],
        calculate_kde=False, calculate_1d_dist=False
    )
    
    
    X1, X2, y1, y2 = train_test_split(
            X_df, y_df, test_size=0.5, stratify=y, random_state=settings_dict["seed_choice"]
        )
    
    
    scoring = "f1_macro"
    
    # Sequential Feature Selection first classifier
    feat_selector1 = SequentialFeatureSelector(
        lcdc1,
        k_features=X1.shape[1],
        scoring=scoring,
        forward=True,
        n_jobs=-1,
        verbose=0,
    ).fit(X1, y1)
    
    # Sequential Feature Selection second classifier
    feat_selector2 = SequentialFeatureSelector(
        lcdc2,
        k_features=X2.shape[1],
        scoring=scoring,
        forward=True,
        n_jobs=-1,
        verbose=0,
    ).fit(X2, y2)
    
    
    res_df1 = pd.DataFrame.from_dict(feat_selector1.get_metric_dict()).T
    res_df1.index.name = "num_feats"
    res_df1["avg_score"] = res_df1["avg_score"].astype("float")
    res_df1 = res_df1.sort_values(by="avg_score", ascending=False)
    res_df1.to_csv(".tempres_df1.csv")
    
    res_df2 = pd.DataFrame.from_dict(feat_selector2.get_metric_dict()).T
    res_df2.index.name = "num_feats"
    res_df2["avg_score"] = res_df2["avg_score"].astype("float")
    res_df2 = res_df2.sort_values(by="avg_score", ascending=False)
    res_df2.to_csv(".tempres_df2.csv")
    
    # Reloading to
    sfs_df1 = pd.read_csv(".tempres_df1.csv", index_col=0)
    feats_idx1, feats1 = utils.load_best_features(sfs_df1)
    print(f"{metric_str} LCDC 1: Selected {len(feats1)} features: {feats1}")
    
    sfs_df2 = pd.read_csv(".tempres_df2.csv", index_col=0)
    feats_idx2, feats2 = utils.load_best_features(sfs_df2)
    print(f"{metric_str} LCDC 2: Selected {len(feats2)} features: {feats2}")
    
    
    HIDDENy_df = pd.read_csv("data/HIDDENy_df_multiclass.csv", index_col=0)
    HIDDENX_df = pd.read_csv("data/HIDDENX_df_multiclass.csv", index_col=0)
    
    
    HIDDENX_df1 = HIDDENX_df.loc[:, feats1]
    HIDDENX_df1 = HIDDENX_df1.dropna()
    HIDDENy_df1 = HIDDENy_df.loc[HIDDENX_df1.index]
    HIDDENX1 = HIDDENX_df1.to_numpy()
    HIDDENy1 = HIDDENy_df1.to_numpy().ravel()
    
    HIDDENX_df2 = HIDDENX_df.loc[:, feats2]
    HIDDENX_df2 = HIDDENX_df2.dropna()
    HIDDENy_df2 = HIDDENy_df.loc[HIDDENX_df2.index]
    HIDDENX2 = HIDDENX_df2.to_numpy()
    HIDDENy2 = HIDDENy_df2.to_numpy().ravel()
    
    # assert (HIDDENX2 == HIDDENX2).all()
    
    lcdc1.fit(X1.loc[:,feats1].to_numpy(), y1)
    lcdc2.fit(X2.loc[:,feats2].to_numpy(), y2)
    
    HIDDENypred1 = lcdc1.predict(HIDDENX1)
    HIDDENypred2 = lcdc2.predict(HIDDENX2)
    
    
    acc1 = accuracy_score(y_true=HIDDENy1, y_pred=HIDDENypred1)
    f1score1 = f1_score(y_true=HIDDENy1, y_pred=HIDDENypred1, average="macro")
    matthew_coef1 = matthews_corrcoef(y_true=HIDDENy1, y_pred=HIDDENypred1)
    
    print(f"{metric_str} LCDC 1: F1 Score = {f1score1:.3f}")
    
    acc2 = accuracy_score(y_true=HIDDENy2, y_pred=HIDDENypred2)
    f1score2 = f1_score(y_true=HIDDENy2, y_pred=HIDDENypred2, average="macro")
    matthew_coef2 = matthews_corrcoef(y_true=HIDDENy2, y_pred=HIDDENypred2)
    
    print(f"{metric_str} LCDC 2: F1 Score = {f1score2:.3f}")
    
    
    
