Project Structure:
Abstract - do at the end.
Introduction
Literature Review
Theory
Methodology and Implementation
Results
Discussion
Constraints and Future Work
Conclusion
References
Appendix

## Abstract

Leave for now

## Introduction

The Literature Review defines the current state of research in your area and places your work in context with other work in your project area. It also acts as the foundation for the comparison of your results with other relevant work
in the 'Discussion’ chapter in your Individual Project. The review is a summary of relevant articles (that is, material relevant to the background to your project, overview of the subject, discussion of any essential theories, etc.) published in technical journals, conference proceedings, books, websites, etc. - avoid using very general references. The majority of references should be books or journal papers; web sites should appear only occasionally in the list. 

This needs to be applied to the MOVG work we have done so far. Do not use proper references just yet, but add in placeholders where you have referred to the papers in the project files. We will need to talk about the current state of time series preprocessing methods as well, hinting to how they struggle with interpretability and explainability. This section needs to be around 1250 words. Most of the references will go here. We will not need to worry about IMS at all. At most, 1250 words tops.

## Theory

Need to explain the point of the MOVG. 
Need to explain how the MOVG was derived from min-max trees and visibility graph equations, we need to talk about how it is visually interpretable and how we can use it to make preprocessing a clearer procedure. A small example could be used here as well.

Explain how it can be used to find long term and short term trends, explaining the point of the decomposition algorithm and how it works - include pseudocode here as well, talking about the chains. 

Show an example of the decomposition algorithm working here as well - reuse the pseudocode from the MOVG formalisation doc, highlighting the points on a time series. I will add it in for the MOVG, you don't need to worry about that.

Need to explain how it works in conjunction with signatures (aggregate and expanded signatures - with definitions for both and explaining the terms in a list), with an explanation of what time series signatures are and what makes them special and useful in this case.

As a timeline or chart/diagram (I will do this)
Start of with time series -> add the points in order, using magnitude and visibility conditions as constraints -> builds MOVG. Then explain how decomposition works - add the pseudocode for decomposition here as well. This should give an overview  

Will need to explain how the other preprocessing methods work as well (FFT, Wavelet decomposition, Catch22) fairly concisely, but we can go into some depth here, explaining how we can use them as a comparison method.

We will need to reuse a bit of the proofs and pseudocode from the MOVG formalisation pdf that is a part of the project files. 

talk about the backtrace algorithm as well here. 

## Methodology and Implementation

Need to explain the deliverables of the project as well (1 - pseudocode and formal documentation, 2 RESTful API design, 3. Model and preprocessing benchmarking). 

Will need to explain how the MOVG fits into the ML model pipeline (also need to explain how we take in data) and how it integrates and how we can achieve and read interpretable outputs from the MOVG preprocessing (The model predicted a spike because it learned from this chain in the preprocessing). We will also need to explain why we chose a windowed MOVG. I will add the diagram myself, don't worry.

I will need you to explain how the MOVG can work with XGB, LGBM, SVR, GPR, Ridge Regression etc, to explain how we can use it in a variety of models. Need to explain all of the testing that I have done - different preprocessing methods, different dataset sizes, different models, using SHAP to extract features and I will also need you to explain how the backtracing algorithm works, as well as the 32 features that we use and how we arrived on them.  

We will need to talk about the walkforward method for the time series and how it was used to prevent data leakage - can refer to some initial IMS dataset plots, 

## Results and Discussion

Need to talk about the datasets that we tested on and how we used a wide range of datasets, where certain features of time series were exhibitied, like the weather csv, which could show seasonality. We need to talk about the three project deliverables here in more detail, and in particular, take some more screenshots of the backtrace window, talk about how we get JSON output for a specific window that can be easily analysed as well. Need to talk about the API and also tests that we will run for the package, to make sure it is working. We will need to talk about the documentation as well. We need to explain how the bacltracing algo worked, and how it picked certain trends from a user selected window. 

# Constraints and Future Work

Will need to explain here how when we built the MOVG library, we used a lot of packages that were not designed to work in tandem with a GPU, specifically networkx, and how the algorithm is very slow due to the visibility checking aspect of the algorithm etc. Need to talk about how it is only a univariate version, and how we can use the theory behind multivariate visibility graphs to implement a multivariate version as well. 

