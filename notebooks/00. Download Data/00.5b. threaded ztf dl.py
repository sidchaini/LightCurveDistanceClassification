import pandas as pd
import os
import shutil
import sys
import numpy as np
import wget
from tqdm.auto import tqdm
import subprocess

subcatalog = pd.read_csv("data/chenfeatures_df.csv", index_col=0)
oids = subcatalog.index.to_numpy()

### START OF OID CODE

new_directory = "data/lightcurves/"
old_directory = (
    "/Users/sidchaini/Research/LCDistanceMetrics/phoenix/ztf_download/ztfdata/"
)
dl_directory = "data/ztfdata/"

old_oids_files = os.listdir(old_directory)
old_oids = [
    filename.split(".")[0] for filename in old_oids_files if filename.endswith(".csv")
]

# Initialize lists to hold the categorized oids
oids_present = []
oids_absent = []

# Check each oid and categorize
for oid in tqdm(oids):
    if str(oid) in old_oids:
        oids_present.append(oid)
        # Copy the file to the current directory
        shutil.copy(os.path.join(old_directory, f"{oid}.csv"), new_directory)
    else:
        oids_absent.append(oid)

print(f"Copied {len(oids_present)} csvs successfully to {new_directory}")
oids = oids_absent

### END OF OID CODE

numthreads = 200

objlistarr = []
for i in range(numthreads):
    if i < numthreads - 1:
        objlistarr.append(
            oids[i * len(oids) // numthreads : (i + 1) * len(oids) // numthreads]
        )
    else:
        objlistarr.append(oids[i * len(oids) // numthreads :])


def run_thread(thread_num):
    continue_flag = False
    sidobjlist = objlistarr[thread_num]
    for i in tqdm(range(len(sidobjlist)), desc=f"Thread {thread_num}"):
        o = sidobjlist[i]
        r = subcatalog.loc[o, "RAdeg"]
        d = subcatalog.loc[o, "DEdeg"]

        if f"{o}_g.csv" not in os.listdir(dl_directory):
            url_g = f"https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?POS=CIRCLE+{r}+{d}+0.00028&BANDNAME=g&NOBS_MIN=10&BAD_CATFLAGS_MASK=32768&FORMAT=CSV"
            file_g = wget.download(url_g, out=f"{dl_directory}/{o}_g.csv")

        if f"{o}_r.csv" not in os.listdir(dl_directory):
            url_r = f"https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?POS=CIRCLE+{r}+{d}+0.00028&BANDNAME=r&NOBS_MIN=10&BAD_CATFLAGS_MASK=32768&FORMAT=CSV"
            file_r = wget.download(url_r, out=f"{dl_directory}/{o}_r.csv")


import threading

threads = []

for i in range(numthreads):
    t = threading.Thread(target=run_thread, args=(i,))
    t.daemon = True
    threads.append(t)

for i in range(numthreads):
    threads[i].start()

for i in range(numthreads):
    threads[i].join()
