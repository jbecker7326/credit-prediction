# Credit Prediction

This project uses models implemented with scikit-learn to predict of whether a person will default on their credit. It was created as a project for ML Zoomcamp 2023 cohort.

## Important Files
- **notebooks/notebook.ipynb**: 
-- Section 1 contains data preparation and EDA for the credit dataset. 
-- Section 2 includes cross-validation training of multiple models (Decision Trees, Neural Nets, KNN and SVM) implemented with scikit-learn and hyperparameter tuning for each model
-- Section 3 compares all tuned models from section 2 to determine the best one. Our SVM model had the best performance as well as a relatively low runtime, so we decided to move forward with this one.
- **scripts/train.py**: Train the model and saves the output to a bin file.
- **scripts/predict.py**: Reads the bin file and deploys the model to an app using flask.
- **Pipfile**: Pip dependencies for local reproducibility of the project.
- **Dockerfile**: Dockerfile for local reproducibility of the project.
- **scripts/predict-test.py**: Tests functionality of the app after deployment.

## Background
Recently, the surge in the housing market has caused many prospective homebuyers to apply for loans. By analyzing historical data and various features related to a client's financial history, the model can estimate the likelihood of defaulting on a loan. This information is crucial for making informed decisions about whether to approve or deny a loan application.

Our objective is to build a model that predict whether or not a person will default on their credit. The resulting model would likely be used by credit lenders to determine whether or not to approve an individual for a new credit line. In this case, lenders look at both cases in which individuals have good or bad credit to make an assessment. In this case, the most important factor in our analysis is reducing the risk of incorrectly classifying bad credit as good credit to prevent lending to the wrong customers. For this, we want to increase the number of true positives and reduce the chance of false negatives, so we used **recall** as our performance metric.


## Data

We observed a binary classification problem of whether or not someone has good credit. The corresponding dataset can be downloaded from [OpenML](https://www.openml.org/search?type=data&sort=runs&id=31&status=active). The scripts currently use scikit-learns built-in functionality with OpenML to load the dataset directly.

The dataset contains 1,000 records and 20 features. Of these features, there are 7 discrete numeric features of various ranges and 14 symbolic string features with up to 5 possible values. 


## Data Preparation

Section 1.2 and 1.4 of the [notebook](./notebooks/notebook.ipynb) includes our EDA for the credit dataset. It was also converted into a standalone [python file](./scripts/data_prep.py), which is automatically applied to the training data before deployment. Our primary goal was to covert all categorical columns to numeric.
To achieve this, we used binary conversion, ordinal encoding, and one-hot encoding. Our preparation pipeline was run on both the training and testing dataset.


## EDA

Section 1.3 of the [notebook](./notebooks/notebook.ipynb) includes our EDA for the credit dataset.

First, for each feature we computed the following metrics, which are also exported to [csv](./notebooks/eda/credit.csv):
- **Outliers** Outliers are a significant distance from the mean. Here they are qualified as anything with a zscore > 3.
- **Kurtosis** determine the volume of outliers.
- **Skewness** is a measure of asymmetry of a distribution.
- **Sparse** data means that many of the values are zero, but you know that they are zero. NZeros column counts number of records with value of 0. We use EDA to determine whether these are sparse or missing records.
- **Missing** data means that you don't know what some or many of the values are. NNull counts the number of missing records.
- **Unique** NUnique counts unique number of unique values.
- **Label Correlation** Pearson correlation between each column and the label
- **Imbalanced data** can be measured as the percent of the minority class to determine if the data must be balanced before analysis.
- **Feature importance** can be measured by calculating the mutual information between each column and the label column. We used sklearn.metrics.mutual_info_score.

Next, we generated [pair plots](./notebooks/eda/credit_hist) to get an overview of all features, then took a closer look at their relationships with the label using a [correlation matrix](./notebooks/eda/credit_corr) for the top 5 most highly correlated features.


## Training and Model Comparison

Section 2 of the [notebook](./notebooks/notebook.ipynb) includes training for 4 different models implemented with scikit-learn. We created validation curves, learning curves, and loss curves (where applicable) to tune the hyperparameters to their optimal performance. We looked at:
- **Decision Trees**: Section 2.1
- **K-Nearest Neighbors**: Section 2.2
- **Support Vector Machines**: Section 2.3
- **Neural Networks**: Section 2.4

Section 3 of the [notebook](./notebooks/notebook.ipynb) shows the comparison metrics for all four models. We found that our best performing models were SVM and Neutral Networks, however, we decided to move forward with SVM due to it notable reduction in runtime. 
The final performance metrics for each model can be seen in the following table:

Model | Fit time | Precision | Recall | F1
--- | --- | --- | --- | ---
DT | 0.003500 | 0.782051 | 0.865248 | 0.821549
KNN | 0.001000 | 0.738372 | 0.900709 | 0.811502
SVM | 0.012499 | 0.705000 | 1.000000 | 0.826979
NN | 0.027501 | 0.705000 | 1.000000 | 0.826979


## Local deployment of the app
To follow this deployment, you will first need to have python and pipenv installed. After doing so, navigate to the root of the repository and open the pipenv virtual environment by running the following command in your command line:
```sh
pipenv shell
```

Next, navigate to the scripts folder. If you are on mac, run the following:
```sh
gunicorn —bind 0.0.0.0:9696 predict:app
```

Alternatively, if you are on windows, run:
```sh
waitress-serve --listen=*:9696 predict:app
```

Finally, you can test the apps functionality by opening a separate command line window, navigate to the scripts directory, and run the following command:
```sh
python predict-test.py
```

## Containerized local deployment of the app
Instead of using a local conda environment, the app can be deployed locally using docker. Open a command prompt and navigate to the root of the directory. Next, run the following commands to deploy the app.

```sh
docker build -t credit-prediction .

docker run -it -p 9696:9696 credit-prediction:latest
```

Again, you can test the apps functionality by opening a separate command line window, navigate to the scripts directory, and run the following command:
```sh
python predict-test.py
```