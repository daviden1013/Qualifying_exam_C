PATH = r'/home/daviden1013/David_projects/Qualifying_exam_C/'

import os
import numpy as np
import pandas as pd

# Load data 
df = pd.read_csv(os.path.join(PATH, 'data', 'breast-cancer.data.csv'))

# Define output
curated_df = pd.DataFrame({"id": df.index})

""" classes"""
curated_df['class'] = df['class'].map({'no-recurrence-events': False, 'recurrence-events': True})

""" Predictors """
# ordinal to numeric
curated_df['age'] = df['age'].map({t:(int(t.split('-')[0]) + int(t.split('-')[1]))/2 for t in df['age'].unique()})
curated_df['tumor-size'] = df['tumor-size'].map({t:(int(t.split('-')[0]) + int(t.split('-')[1]))/2 for t in df['tumor-size'].unique()})
curated_df['inv-nodes'] = df['inv-nodes'].map({t:(int(t.split('-')[0]) + int(t.split('-')[1]))/2 for t in df['inv-nodes'].unique()})
curated_df['deg-malig'] = df['deg-malig'].astype(float)

# categorical to one-hot encoding
cate = pd.get_dummies(df[['menopause', 'node-caps', 'breast', 'breast-quad', 'irradiat']], drop_first=True)

# Concat
curated_df = pd.concat([curated_df, cate], axis=1)

curated_df.dtypes
"""
>>> curated_df.dtypes
class                       bool
age                      float64
tumor-size               float64
inv-nodes                float64
deg-malig                float64
menopause_lt40              bool
menopause_premeno           bool
node-caps_no                bool
node-caps_yes               bool
breast_right                bool
breast-quad_central         bool
breast-quad_left_low        bool
breast-quad_left_up         bool
breast-quad_right_low       bool
breast-quad_right_up        bool
irradiat_yes                bool
"""
""" Train-test split """
np.random.seed(123)
test_ids = np.random.choice(curated_df['id'], size=int(curated_df.shape[0] * 0.2), replace=False)
curated_df['train_test'] = (curated_df['id'].isin(test_ids)).map({True: 'test', False: 'train'})
curated_df['train_test'].value_counts()
"""
>>> curated_df['train_test'].value_counts()
train_test
train    229
test      57
"""

""" Save """
curated_df.to_pickle(os.path.join(PATH, 'data', 'curated_df.pkl'))
