## Repository Details

This repository contains all the code (in the form of Jupyter Notebooks) to reproduce the results in our paper, "Light Curve Classification with DistClassiPy: a new distance-based classifier"

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
> Light Curve Classification with DistClassiPy: a new distance-based classifier (submitted to A&C)


<!--
### Bibtex


```bibtex
@ARTICLE{Chaini2024,
       author = {{Chaini}, S. and {Mahabal}, A. and {Kembhavi}, A. and {Bianco}, F.~B.},
        title = "{Light Curve Classification with DistClassiPy: a new distance-based classifier}",
      journal = {Submitted to A&C},
    %  keywords = {},
         year = 2024,
      %   month = ,
      %  volume = {},
      %     eid = {},
      %   pages = {},
      %     doi = {},
      %  adsurl = {},
      % adsnote = {}
}
```
-->
  

<!-- You can also find citation information in the [CITATION.cff](https://github.com/sidchaini/DistClassiPy/CITATION.cff) file. -->


## Authors
Siddharth Chaini, Ashish Mahabal, Ajit Kembhavi and Federica B. Bianco.
