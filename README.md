# MS&E 231 - Assignment 2 - Axel Peytavin & Constantin Eulenstein

This repository contains the full code to reproduce our results for the paper submitted.

To run it, use Python 3.7 or 3.8. This code requires the following packages: `numpy, networkx, tweepy, sklearn, pandas, pickle, yaml, json pygraphviz, matplotlib`.

## How to run:

You can run the following files in order:

- `get_legislators.py` takes in the `legislators-current.yaml` to generate a legislators file.
- `tweet_stream.py` uses this file to download in a `data/` folder the tweets between today and 1 week ago, and the related conversations in `data/conversations/conversation*.json`
 - This data is available in the zip file attached with the code.
- `compute_graphs_and_metrics.py` will assemble graphs and trees from the conversation data and compute associated metrics, putting everything in a `conversation_metrics_v6.json` file.
- `conversation_classifier.py` will take this json file to process our analysis on it and output values in different plots and in the std output, depending on the booleans at the beginning of it. The options available are:
 - plot_distributions: Plot metrics CCDFs across the datasets in pyplot.
 - check_logistic_regression: Add a round of logistic regression on the selected metrics first and plot the features coefficients at the end. 
 - evaluate_models: Evalute all different models we picked across the dataset and choose the best one to perform the final analysis.
 - calculate_stats: Print some statistics of the metrics across the dataset.

## Data available

You don't need to rerun everything to reproduce our results. Our data collected for the paper is available in the attached data.zip file. 
Copy what's in the archive at the root of the code folder, and you will be able to start each part separately, and run the last steps (graph metrics computing and classifier) on our data pulled between Oct 18th and Oct 25th.
