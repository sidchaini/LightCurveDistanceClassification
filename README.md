## Repository Details

This repository contains all the code (in the form of Jupyter Notebooks) to reproduce the results in our paper, "[Light Curve Classification with DistClassiPy: a new distance-based classifier](https://arxiv.org/abs/2403.12120)".

The accompanying package, [```DistClassiPy```](https://github.com/sidchaini/DistClassiPy/) can be found on GitHub at [sidchaini/DistClassiPy](https://github.com/sidchaini/DistClassiPy/) and can be installed from PyPI by the command ```pip install distclassipy```.

## Notebooks
The Jupyter Notebooks are in the [```notebooks/```](https://github.com/sidchaini/LightCurveDistanceClassification/tree/main/notebooks) directory. There are a total of 10 base notebooks with subvariants for each classification depending on the notebook.

00. Download Data
01. Distance Metrics
02. Preprocess Data (a,b,c)
03. Classification (a,b,c)
04. Analysis (a,b,c)
05. RFC Comparison (a,b,c)
06. Hidden Set Results (a,b,c)
07. Computational Complexity
08. Confidence Comparison (a,b,c)
09. Robustness (a,b,c)

Note that ```a``` denotes the one-vs-rest classification problem (EA vs notEA), ```b``` denotes the binary classification problem (RSCVn vs BYDra) and ```c``` denotes the multi-class classification problem (CEP vs DSCT vs RR vs RRc).

## Citation

If you use DistClassiPy in your research or project, please consider citing the paper:
> Chaini, S., Mahabal, A., Kembhavi, A., & Bianco, F. B. (2024). Light Curve Classification with DistClassiPy: a new distance-based classifier. arXiv. https://doi.org/10.48550/arXiv.2403.12120

### Bibtex


```bibtex
@ARTICLE{chaini2024light,
       author = {{Chaini}, Siddharth and {Mahabal}, Ashish and {Kembhavi}, Ajit and {Bianco}, Federica B.},
       title = "{Light Curve Classification with DistClassiPy: a new distance-based classifier}",
       journal = {arXiv e-prints},
       keywords = {Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Solar and Stellar Astrophysics, Computer Science - Machine Learning},
       year = 2024,
       month = mar,
       eid = {arXiv:2403.12120},
       pages = {arXiv:2403.12120},
       archivePrefix = {arXiv},
       eprint = {2403.12120},
       primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv240312120C},
       adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
  

<!-- You can also find citation information in the [CITATION.cff](https://github.com/sidchaini/DistClassiPy/CITATION.cff) file. -->

<!-- You can also find citation information in the [CITATION.cff](https://github.com/sidchaini/DistClassiPy/CITATION.cff) file. -->


## Authors
Siddharth Chaini, Ashish Mahabal, Ajit Kembhavi and Federica B. Bianco.
