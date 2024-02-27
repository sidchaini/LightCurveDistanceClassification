import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
from sklearn.metrics import confusion_matrix
import seaborn as sns


def get_2digits(num):
    """
    Function to get simplified string form of from abc * 10^(xyz) -> (a,x)
    Examples
    --------
    >>> get_2digits(547.123)
    '5 \\times 10^{2}'
    """
    scinum_ls = "{:.0e}".format(num).split("e+")
    if scinum_ls[1] == "00":
        label = r"{0}".format(scinum_ls[0])
    else:
        label = r"{0} \times 10^{{{1}}}".format(
            scinum_ls[0], scinum_ls[1].replace("0", "")
        )
    return label


def plot_two_chen_lightcurves(df1, df2):
    # Plot the lightcurves for the given object
    dflist = [df1, df2]
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    for i, ax in enumerate(axs):
        df = dflist[i]
        df_g = df[df["band"] == "g"].sort_values(by="HJD", ascending=True)
        df_r = df[df["band"] == "r"].sort_values(by="HJD", ascending=True)
        objid = df["SourceID"].iloc[0]

        ax.errorbar(df_g["HJD"], df_g["mag"], yerr=df_g["e_mag"], fmt="go", label="g")
        ax.errorbar(df_r["HJD"], df_r["mag"], yerr=df_r["e_mag"], fmt="ro", label="r")
        ax.invert_yaxis()
        ax.set_xlabel("HJD")
        ax.set_ylabel("Magnitude")
        ax.set_title("Lightcurve for SourceID " + str(objid))
        ax.legend()
    plt.show()


def get_individual_lc(lcs, objid):
    return lcs[(lcs["SourceID"] == objid)]


def local_alerce_data_cleaner(alerce_features_path, chen_features_path):
    """
    Note: This is a highly specific function built only for preprocessing and combining
    the features calculatef from ALeRCE's package lc_classifier along with chen's features,
    for a objects from Chen's catalog ONLY.

    For objects from different sources (e.g ZTF DRs), a new function will be needed.
    Albeit, a lot of code here can be copied over.
    Nevertheless,
    this function is for internal purposes only.
    """

    ### 1.1 Load alerce featues calculated in step 2 and prep
    alercefeatures_df = pd.read_csv(alerce_features_path)
    alercefeatures_df = alercefeatures_df.set_index("oid")
    ### 1.2 Load chen features from Chen subset downloaded in step 1 and prep them
    chenfeatures_df = pd.read_csv(chen_features_path)
    chenfeatures_df = chenfeatures_df.rename(
        columns={
            "SourceID": "oid",
            "Type": "class",
            "213-elta_min_g": "Delta_min_g",
            "219-elta_min_r": "Delta_min_r",
        }
    )
    chenfeatures_df = chenfeatures_df.set_index("oid")
    ### 1.3 Drop redundant and irrelevant columns
    ####### (refer siddharth/archived/removing irrelevant and redundant features.ipynb for why)
    alercefeatures_df = alercefeatures_df.drop(
        [
            # Irrelevant:
            "MHPS_non_zero_g",
            "MHPS_non_zero_r",
            # Redundant:
            "iqr_r",
            "iqr_g",
        ],
        axis=1,
    )

    chenfeatures_df = chenfeatures_df.drop(
        [
            # Irrelevant:
            "ID",
            "RAdeg",
            "DEdeg",
            "T_0",
            "Num_g",
            "Num_r",
            # Redundant:
            "Per",
            "Per_g",
            "Per_r",
            "rmag",
            "gmag",
            "Amp_g",
            "Amp_r",
            "phi21",
            "R21",
        ],
        axis=1,
    )
    ### 1.4 Drop all NA from everything - important to remove objects where ALeRCE feature extraction failed
    chenfeatures_df = chenfeatures_df.dropna()
    alercefeatures_df = alercefeatures_df.dropna()
    #### 1.4.1 Remove the same NA rows objects from both the dataframes
    com_idx = np.intersect1d(
        alercefeatures_df.index.to_numpy(), chenfeatures_df.index.to_numpy()
    )
    chenfeatures_df = chenfeatures_df.loc[com_idx]
    alercefeatures_df = alercefeatures_df.loc[com_idx]
    ### 1.5 Create a dataframe with just the "oid" and "class"
    class_df = chenfeatures_df[["class"]]
    #### 1.5.1 Remove classes from chenfeatures
    chenfeatures_df = chenfeatures_df.drop(["class"], axis=1)
    ### 1.6 Create a final combined feature dataset
    chenalerce_df = pd.concat([alercefeatures_df, chenfeatures_df], axis=1)
    print(f"Total of {alercefeatures_df.shape[1]} features in ALeRCE")
    print(f"Total of {chenfeatures_df.shape[1]} features in Chen")
    print("*" * 40)
    print(f"Total of {chenalerce_df.shape[1]} features in ALL")

    print("List of features being used:")
    print(
        "https://docs.google.com/spreadsheets/d/1EDzk51Nzk6jhGbZhimN3iQiRkba_NCRN6Uoxi6eTzoU/edit?usp=sharing"
    )

    # Standardising
    # chenalerce_df_meannorm = (chenalerce_df-chenalerce_df.mean())/chenalerce_df.std()

    # Normalising:
    # chenalerce_df_minmax = (chenalerce_df-chenalerce_df.min())/(chenalerce_df.max()-chenalerce_df.min())

    return alercefeatures_df, chenfeatures_df, chenalerce_df, class_df


def get_metric_name(metric):
    if callable(metric):
        metric_str = metric.__name__
    else:
        metric_str = metric
    return metric_str.title()


def load_best_features(sfs_df):
    """

    Choose n features for this metric such that n is the
    lowest number of features whose F1 score is within 1
    standard deviation of the maximum F1 score.

    Input:
    sfs_df: Sequential feature selection dataframe with objid as 'num_feats' as index.

    Returns:
    feats_idx: Index of the best features
    feats: Name of the best features
    """

    idx_maxscore = sfs_df["avg_score"].idxmax()
    max_score = sfs_df.loc[idx_maxscore, "avg_score"]
    max_score_std = sfs_df.loc[idx_maxscore, "std_dev"]

    sfs_df = sfs_df.loc[(sfs_df["avg_score"]) >= max_score - max_score_std]
    sfs_df = sfs_df.sort_index()

    feats_idx = list(ast.literal_eval(sfs_df.iloc[0]["feature_idx"]))
    feats = list(ast.literal_eval(sfs_df.iloc[0]["feature_names"]))

    return feats_idx, feats


def plot_cm(y_true, y_pred, annot_fmt="pn", label_strings=None):
    """
    annot_fmt: n is numeric, p is percentage and np is both no.(%)
    """
    if label_strings is None:
        label_strings = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=label_strings)

    fig, ax = plt.subplots()

    if annot_fmt == "n":
        df_cm = pd.DataFrame(cm, index=label_strings, columns=label_strings)
        sns.heatmap(
            df_cm, annot=True, cmap="Blues", square=True, fmt="d", ax=ax, cbar=False
        )  # fmt='.2%'
    elif annot_fmt == "p":
        cm = confusion_matrix(y_true, y_pred, normalize="true", labels=label_strings)
        df_cm = pd.DataFrame(cm, index=label_strings, columns=label_strings)
        sns.heatmap(
            df_cm, annot=True, cmap="Blues", square=True, fmt=".1%", ax=ax, cbar=False
        )

    elif annot_fmt == "np":
        df_cm = pd.DataFrame(cm, index=label_strings, columns=label_strings)
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                annot[i, j] = f"{cm[i, j]}\n({cm_perc[i, j]:.0f}%)"
        sns.heatmap(
            df_cm, annot=annot, cmap="Blues", square=True, fmt="", ax=ax, cbar=False
        )

    elif annot_fmt == "pn":
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100
        df_cm = pd.DataFrame(cm_perc, index=label_strings, columns=label_strings)
        df_cm_rounded = df_cm.round(decimals=0).astype("int")
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                annot[i, j] = f"{cm_perc[i, j]:.0f}%\n({cm[i, j]})"
        sns.heatmap(
            df_cm, annot=annot, cmap="Blues", square=True, fmt="", ax=ax, cbar=False
        )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    # plt.xticks(rotation=45, ha='right')
    # plt.yticks(rotation=45)

    return ax
