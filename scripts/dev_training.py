PATH = r'/home/daviden1013/David_projects/Qualifying_exam_C/'

import os
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import pandas as pd

df = pd.read_pickle(os.path.join(PATH, 'data', 'curated_df.pkl'))

feature_cols = ['age', 'tumor-size', 'inv-nodes', 'deg-malig',
       'menopause_lt40', 'menopause_premeno', 'node-caps_no', 'node-caps_yes',
       'breast_right', 'breast-quad_central', 'breast-quad_left_low',
       'breast-quad_left_up', 'breast-quad_right_low', 'breast-quad_right_up',
       'irradiat_yes']

train_features = df.loc[df['train_test'].isin(['train']), feature_cols]
train_labels = df.loc[df['train_test'].isin(['train']), 'class']

test_features = df.loc[df['train_test'].isin(['test']), feature_cols]
test_labels = df.loc[df['train_test'].isin(['test']), 'class']

""" Scale X """
scaler = StandardScaler()
train_features_scaled = pd.DataFrame(scaler.fit_transform(train_features), columns=train_features.columns, index=train_features.index)
test_features_scaled = pd.DataFrame(scaler.transform(test_features), columns=test_features.columns, index=test_features.index)

""" Fit model """
if config['model'] == "LogisticRegression":
    model = LogisticRegression(max_iter=500, penalty='l2', random_state=123)
elif config['model'] == "RandomForestClassifier":
    model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=123)
else:
    raise ValueError(f"model config['model'] is not supported.")

model.fit(train_features_scaled, train_labels)

y_prob = model.predict_proba(test_features_scaled)
y_pred = model.predict(test_features_scaled)
gold = test_labels

precision = precision_score(gold, y_pred)
recall = recall_score(gold, y_pred)
f1 = f1_score(gold, y_pred)
acc = accuracy_score(gold, y_pred)
auroc = roc_auc_score(gold, y_prob[:,1])
metrics.append({"Precision": precision, 
                "Recall": recall, 
                "F1": f1, 
                "Accuracy": acc,
                "AUROC": auroc})

