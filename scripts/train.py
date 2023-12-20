# data storage structures
import pandas as pd
import numpy as np

# preprocessing and modeling
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import precision_score, f1_score, recall_score
from sklearn.feature_extraction import DictVectorizer

# saving model
import pickle

# timer
import time

# data preparation pipeline
import data_prep

def train_svm(data):
    credit_train, credit_test = train_test_split(data, test_size=0.2, random_state=42)

    # preprocess train set
    credit_train = data_prep.data_prep(credit_train)
    credit_train_x, credit_train_y = credit_train.drop('class', axis=1), credit_train['class']
    credit_test = data_prep.data_prep(credit_test)

    # save training data features
    dicts = credit_train_x.to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    
    # split x and y for test and train sets
    credit_train_x, credit_train_y = credit_train.drop('class', axis=1), credit_train['class']
    credit_test_x, credit_test_y = credit_test.drop('class', axis=1), credit_test['class']

    # final SVM model
    model = svm.SVC(random_state=42, kernel='poly', C=4, degree=1)

    # train and record performance
    start = time.time()
    model.fit(credit_train_x, credit_train_y)
    fit_time = time.time() - start

    # predict results
    y_pred = model.predict(credit_test_x)
    precision = precision_score(credit_test_y, y_pred)
    recall = recall_score(credit_test_y, y_pred)
    f1 = f1_score(credit_test_y, y_pred)

    # print performance metrics
    model_performance = np.array([fit_time, precision, recall, f1])
    model_performance = pd.Series(model_performance, index=['Fit Time', 'Precision', 'Recall', 'F1'])
    #model_performance_df = pd.DataFrame(model_performance, columns = ['Fit time', 'Precision', 'Recall', 'F1'], index = ['DT', 'KNN', 'SVM', 'NN'])
    print("Final SVM Model Performance: \n", model_performance)

    # save model with pickle
    output_path = f"model.bin"
    pickle.dump((dv, model), open(output_path,'wb'))
    print(f"Model saved to {output_path}")    

def main():
    # download data from OpenML
    credit_data = fetch_openml(data_id=31, target_column=None, parser='auto')['data']
    train_svm(credit_data)

if __name__ == "__main__":
    main()