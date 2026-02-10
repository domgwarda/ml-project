import pandas as pd
import importlib
import features
from cached_features import add_cached_features
importlib.reload(features)
from features import FEATURES

TARGET = "URLs"
FEATURE_NAMES = []

def add_feature(df, feature):
    df[feature.__name__] = feature(df[TARGET])

data = pd.read_excel('PhishDataset/data_imbal - 55000.xlsx')
# data = pd.read_csv('Many-urls.csv')


for feature in FEATURES:
    add_feature(data, feature)
    FEATURE_NAMES.append(feature.__name__)

data = add_cached_features(data)
FEATURE_NAMES.append('best_ratio')

data.to_csv('data_preprocessed.csv', index=False)
    
data.to_pickle("data_with_features.pkl")