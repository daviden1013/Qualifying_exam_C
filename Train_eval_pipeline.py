# -*- coding: utf-8 -*-
import argparse
from easydict import EasyDict
import yaml
import os
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import logging
import numpy as np
import pandas as pd


def main():
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info('Training-evaluation pipeline started')
    """ load config """
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg("-c", "--config", help='path to config file', type=str)
    args = parser.parse_known_args()[0]
    
    with open(args.config) as yaml_file:
        config = EasyDict(yaml.safe_load(yaml_file))
  
    logging.info('Config loaded:')

    """ Load training datasets """
    logging.info('Loading datasets...')
    df = pd.read_pickle(config['data_file'])

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
    logging.info('Transforming features...')
    scaler = StandardScaler()
    train_features_scaled = pd.DataFrame(scaler.fit_transform(train_features), columns=train_features.columns, index=train_features.index)
    test_features_scaled = pd.DataFrame(scaler.transform(test_features), columns=test_features.columns, index=test_features.index)

    """ Fit model """
    logging.info('Fitting model...')
    if config['model'] == "LogisticRegression":
        model = LogisticRegression(multi_class='multinomial', max_iter=500, penalty=None)
    elif config['model'] == "RidgeRegression":
        model = LogisticRegression(multi_class='multinomial', max_iter=500, penalty='l2')
    elif config['model'] == "RandomForestClassifier":
        model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=123)
    elif config['model'] == "XGBClassifier":
        model = XGBClassifier(random_state=123)
    else:
        raise ValueError(f"model {config['model']} is not supported.")

    model.fit(train_features_scaled, train_labels)

    """ Evaluate """
    logging.info('Evaluating...')
    y_prob = model.predict_proba(test_features_scaled)
    y_pred = model.predict(test_features_scaled)
    gold = test_labels

    metrics = []
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
    metrics_df = pd.DataFrame(metrics)

    """ Save """
    # save evaluation metrics
    if not os.path.isdir(os.path.join(config['out_path'], 'evaluation')):
        os.makedirs(os.path.join(config['out_path'], 'evaluation'))

    metrics_df.to_csv(os.path.join(config['out_path'], 'evaluation', f'{config["run_name"]}.csv'))

    # save y_prob
    if not os.path.isdir(os.path.join(config['out_path'], 'y_prob')):
        os.makedirs(os.path.join(config['out_path'], 'y_prob'))

    np.save(os.path.join(config['out_path'], 'y_prob', f'{config["run_name"]}.npy'), y_prob)

    logging.info('Training-evaluation pipeline completed.')

if __name__ == '__main__':
    main()