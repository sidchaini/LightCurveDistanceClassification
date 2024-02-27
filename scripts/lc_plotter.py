import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


class LightCurvePlotter:
    def __init__(self, all_lc_path=None):
        self.all_lc_path = all_lc_path
        self.lcdf = pd.read_csv(all_lc_path)

    def plot_lc(self, oid, savepath=None):
        if oid not in self.lcdf["SourceID"].unique():
            raise IndexError(f"OID {oid} not found.")
        df = self.lcdf[self.lcdf["SourceID"] == oid]
        df = df.sort_values(by="HJD")
        df["HJD"] = df["HJD"] - df["HJD"].min()
        df_g = df[df["band"] == "g"]
        df_r = df[df["band"] == "r"]
        plt.errorbar(
            x=df_g["HJD"],
            y=df_g["mag"],
            c="#55a868",
            yerr=df_g["e_mag"],
            fmt=".",
            label="g",
        )
        plt.errorbar(
            x=df_r["HJD"],
            y=df_r["mag"],
            c="#c44e52",
            yerr=df_r["e_mag"],
            fmt=".",
            label="r",
        )
        plt.legend(loc="lower right")
        plt.title(f"OID: {oid}")
        plt.xlabel("HJD")
        plt.ylabel("Magnitude")
        plt.gca().invert_yaxis()
        if savepath is not None:
            plt.savefig(f"{savepath}.pdf")
        return plt
        # plt.show()
