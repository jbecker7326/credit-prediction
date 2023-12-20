# data storage structures
import pandas as pd
import numpy as np

# preprocessing and modeling
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn_pandas import DataFrameMapper

def data_prep(X):
    X = X.copy()    
    # converting columns to binary
    X['personal_status'] = X['personal_status'].str.split(' ').str[0]
    X['other_parties'] = X['other_parties'].replace(['guarantor', 'co applicant'], 'co-applicant/guarantor')
    X['other_payment_plans'] = X['other_payment_plans'].replace(['bank', 'stores'], 'bank/stores')

    ordinal_encoder = OrdinalEncoder()  # (handle_unknown='use_encoded_value', unknown_value=np.nan)
    binary_cols = ['own_telephone', 'foreign_worker', 'class', 'personal_status', 'other_parties',
                   'other_payment_plans']
    X[binary_cols] = ordinal_encoder.fit_transform(X[binary_cols])

    # single col ordinal encoder with specified columns
    ordinal_encoder = OrdinalEncoder(categories=[['no checking', '<0', '0<=X<200', '>=200']],
                                     handle_unknown='use_encoded_value', unknown_value=np.nan)

    # converting credit history column bc it has three 'paid' values that can be merged
    X['credit_history'] = X['credit_history'].replace(['no credits/all paid', 'all paid', 'existing paid'], 'paid')

    # multi col - first specify attributes
    column_to_cat = {
        'credit_history': ['critical/other existing credit', 'delayed previously', 'paid'],
        "checking_status": ['no checking', '<0', '0<=X<200', '>=200'],
        "savings_status": ['no known savings', '<100', '100<=X<500', '500<=X<1000', '>=1000'],
        "employment": ['unemployed', '<1', '1<=X<4', '4<=X<7', '>=7'],
        "housing": ['for free', 'rent', 'own'],
        'job': ['unemp/unskilled non res', 'unskilled resident', 'skilled', 'high qualif/self emp/mgmt']
    }

    # encode using mapper
    mapper_df = DataFrameMapper(
        [
            ([col], OrdinalEncoder(categories=[cat])) for col, cat in column_to_cat.items()
        ],
        df_out=True
    )
    X[list(column_to_cat.keys())] = mapper_df.fit_transform(X)

    # one hot encoding
    X.reset_index(inplace=True, drop=True)
    onehot = OneHotEncoder(dtype=int, sparse_output=False)
    nominals = pd.DataFrame(
        onehot.fit_transform(X[['property_magnitude']]),
        columns=onehot.get_feature_names_out()
    )
    X = pd.concat([X, nominals], axis=1)
    X.drop(['property_magnitude', 'purpose'], axis=1, inplace=True)

    # min max scaling - can be beneficial for neural nets to prevent excessive runtimes
    # scaler = MinMaxScaler(feature_range=(0,1))
    # scaler.fit_transform(X)

    # discretization - can be helpful for tree based models
    # k_disc = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
    # k_disc.fit_transform(X)

    return X