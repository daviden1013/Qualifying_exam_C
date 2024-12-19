<div align="center"><img src=ROC.png width=500 ></div>

For better readability, view this project on the [GitHub page](https://github.com/daviden1013/Qualifying_exam_C) 

https://github.com/daviden1013/Qualifying_exam_C.

This is a quick project that use machine learning for breast cancer recurrence prediction. The [dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer) is publicly available. 

## Table of Contents
- [Overview](#overview)
- [Prerequisite](#prerequisite)
- [Methods & Results](#methods--results)
  - Data exploration & Summary table
  - Data pre-processing
  - Machine learning model training
  - Evaluation

## Overview
We processed the raw data (N=286) and split it into a training set (80%) and a test set (20%). Four classic machine learning models were evaluated:

| Model | Precision | Recall | F1 | Accuracy | AUROC|
|----------|----------|----------|----------|----------|----------|
| Logistic Regression | 0.3750 | 0.1875 | 0.2500 | 0.6842 | 0.6311 |
| Ridge Regression | 0.3750 | 0.1875 | 0.2500 | 0.6842 | 0.6372 |
| Random Forest | 0.3000 | 0.1875 | 0.2308 | 0.6491 | 0.5991 |
| XGBoost | 0.4167 | 0.3125 | 0.3571 | 0.6842 | 0.6601 |

Among them, the XGBoost shows best performance (precision, recall, F1 and AUROC). 


## Prerequisite
The required packages are listed in the [requirements.txt](requirements.txt). The important packages are 
- pandas
- numpy
- sklearn
- xgboost
- matplotlib

## Methods & Results
Our method includes several steps: data exploration, data curation, summary table, machine learning model training and evaluation. 

### Data exploration & Summary table
Data exploration is done in the Jupyter Notebook [explore_data.ipynb](scripts/explore_data.ipynb). Visual inspection were done to understand the data structure. Data types, missingness, range and distributes were accessed. 

A summary table is created with the *tableone* package.

We notice that cancer recurrent: non-recurrent is about 1:3. There are predictors (e.g., inv-nodes, node-caps, deg-malig, irradiat) show significant correlation with the recurrence. 

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="5" halign="left">Grouped by class</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>Missing</th>
      <th>Overall</th>
      <th>no-recurrence-events</th>
      <th>recurrence-events</th>
      <th>P-Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>n</th>
      <th></th>
      <td></td>
      <td>286</td>
      <td>201</td>
      <td>85</td>
      <td></td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">age, n (%)</th>
      <th>20-29</th>
      <td></td>
      <td>1 (0.3)</td>
      <td>1 (0.5)</td>
      <td></td>
      <td>0.550</td>
    </tr>
    <tr>
      <th>30-39</th>
      <td></td>
      <td>36 (12.6)</td>
      <td>21 (10.4)</td>
      <td>15 (17.6)</td>
      <td></td>
    </tr>
    <tr>
      <th>40-49</th>
      <td></td>
      <td>90 (31.5)</td>
      <td>63 (31.3)</td>
      <td>27 (31.8)</td>
      <td></td>
    </tr>
    <tr>
      <th>50-59</th>
      <td></td>
      <td>96 (33.6)</td>
      <td>71 (35.3)</td>
      <td>25 (29.4)</td>
      <td></td>
    </tr>
    <tr>
      <th>60-69</th>
      <td></td>
      <td>57 (19.9)</td>
      <td>40 (19.9)</td>
      <td>17 (20.0)</td>
      <td></td>
    </tr>
    <tr>
      <th>70-79</th>
      <td></td>
      <td>6 (2.1)</td>
      <td>5 (2.5)</td>
      <td>1 (1.2)</td>
      <td></td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">menopause, n (%)</th>
      <th>ge40</th>
      <td></td>
      <td>129 (45.1)</td>
      <td>94 (46.8)</td>
      <td>35 (41.2)</td>
      <td>0.673</td>
    </tr>
    <tr>
      <th>lt40</th>
      <td></td>
      <td>7 (2.4)</td>
      <td>5 (2.5)</td>
      <td>2 (2.4)</td>
      <td></td>
    </tr>
    <tr>
      <th>premeno</th>
      <td></td>
      <td>150 (52.4)</td>
      <td>102 (50.7)</td>
      <td>48 (56.5)</td>
      <td></td>
    </tr>
    <tr>
      <th rowspan="11" valign="top">tumor-size, n (%)</th>
      <th>0-4</th>
      <td></td>
      <td>8 (2.8)</td>
      <td>7 (3.5)</td>
      <td>1 (1.2)</td>
      <td>0.056</td>
    </tr>
    <tr>
      <th>10-14</th>
      <td></td>
      <td>28 (9.8)</td>
      <td>27 (13.4)</td>
      <td>1 (1.2)</td>
      <td></td>
    </tr>
    <tr>
      <th>15-19</th>
      <td></td>
      <td>30 (10.5)</td>
      <td>23 (11.4)</td>
      <td>7 (8.2)</td>
      <td></td>
    </tr>
    <tr>
      <th>20-24</th>
      <td></td>
      <td>50 (17.5)</td>
      <td>34 (16.9)</td>
      <td>16 (18.8)</td>
      <td></td>
    </tr>
    <tr>
      <th>25-29</th>
      <td></td>
      <td>54 (18.9)</td>
      <td>36 (17.9)</td>
      <td>18 (21.2)</td>
      <td></td>
    </tr>
    <tr>
      <th>30-34</th>
      <td></td>
      <td>60 (21.0)</td>
      <td>35 (17.4)</td>
      <td>25 (29.4)</td>
      <td></td>
    </tr>
    <tr>
      <th>35-39</th>
      <td></td>
      <td>19 (6.6)</td>
      <td>12 (6.0)</td>
      <td>7 (8.2)</td>
      <td></td>
    </tr>
    <tr>
      <th>40-44</th>
      <td></td>
      <td>22 (7.7)</td>
      <td>16 (8.0)</td>
      <td>6 (7.1)</td>
      <td></td>
    </tr>
    <tr>
      <th>45-49</th>
      <td></td>
      <td>3 (1.0)</td>
      <td>2 (1.0)</td>
      <td>1 (1.2)</td>
      <td></td>
    </tr>
    <tr>
      <th>5-9</th>
      <td></td>
      <td>4 (1.4)</td>
      <td>4 (2.0)</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>50-54</th>
      <td></td>
      <td>8 (2.8)</td>
      <td>5 (2.5)</td>
      <td>3 (3.5)</td>
      <td></td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">inv-nodes, n (%)</th>
      <th>0-2</th>
      <td></td>
      <td>213 (74.5)</td>
      <td>167 (83.1)</td>
      <td>46 (54.1)</td>
      <td>&lt;0.001</td>
    </tr>
    <tr>
      <th>12-14</th>
      <td></td>
      <td>3 (1.0)</td>
      <td>1 (0.5)</td>
      <td>2 (2.4)</td>
      <td></td>
    </tr>
    <tr>
      <th>15-17</th>
      <td></td>
      <td>6 (2.1)</td>
      <td>3 (1.5)</td>
      <td>3 (3.5)</td>
      <td></td>
    </tr>
    <tr>
      <th>3-5</th>
      <td></td>
      <td>36 (12.6)</td>
      <td>19 (9.5)</td>
      <td>17 (20.0)</td>
      <td></td>
    </tr>
    <tr>
      <th>6-8</th>
      <td></td>
      <td>17 (5.9)</td>
      <td>7 (3.5)</td>
      <td>10 (11.8)</td>
      <td></td>
    </tr>
    <tr>
      <th>9-11</th>
      <td></td>
      <td>10 (3.5)</td>
      <td>4 (2.0)</td>
      <td>6 (7.1)</td>
      <td></td>
    </tr>
    <tr>
      <th>24-26</th>
      <td></td>
      <td>1 (0.3)</td>
      <td></td>
      <td>1 (1.2)</td>
      <td></td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">node-caps, n (%)</th>
      <th>?</th>
      <td></td>
      <td>8 (2.8)</td>
      <td>5 (2.5)</td>
      <td>3 (3.5)</td>
      <td>&lt;0.001</td>
    </tr>
    <tr>
      <th>no</th>
      <td></td>
      <td>222 (77.6)</td>
      <td>171 (85.1)</td>
      <td>51 (60.0)</td>
      <td></td>
    </tr>
    <tr>
      <th>yes</th>
      <td></td>
      <td>56 (19.6)</td>
      <td>25 (12.4)</td>
      <td>31 (36.5)</td>
      <td></td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">deg-malig, n (%)</th>
      <th>1</th>
      <td></td>
      <td>71 (24.8)</td>
      <td>59 (29.4)</td>
      <td>12 (14.1)</td>
      <td>&lt;0.001</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td>130 (45.5)</td>
      <td>102 (50.7)</td>
      <td>28 (32.9)</td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td></td>
      <td>85 (29.7)</td>
      <td>40 (19.9)</td>
      <td>45 (52.9)</td>
      <td></td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">breast, n (%)</th>
      <th>left</th>
      <td></td>
      <td>152 (53.1)</td>
      <td>103 (51.2)</td>
      <td>49 (57.6)</td>
      <td>0.389</td>
    </tr>
    <tr>
      <th>right</th>
      <td></td>
      <td>134 (46.9)</td>
      <td>98 (48.8)</td>
      <td>36 (42.4)</td>
      <td></td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">breast-quad, n (%)</th>
      <th>central</th>
      <td></td>
      <td>21 (7.3)</td>
      <td>17 (8.5)</td>
      <td>4 (4.7)</td>
      <td>0.319</td>
    </tr>
    <tr>
      <th>left_low</th>
      <td></td>
      <td>110 (38.5)</td>
      <td>75 (37.3)</td>
      <td>35 (41.2)</td>
      <td></td>
    </tr>
    <tr>
      <th>left_up</th>
      <td></td>
      <td>97 (33.9)</td>
      <td>71 (35.3)</td>
      <td>26 (30.6)</td>
      <td></td>
    </tr>
    <tr>
      <th>right_low</th>
      <td></td>
      <td>24 (8.4)</td>
      <td>18 (9.0)</td>
      <td>6 (7.1)</td>
      <td></td>
    </tr>
    <tr>
      <th>right_up</th>
      <td></td>
      <td>33 (11.5)</td>
      <td>20 (10.0)</td>
      <td>13 (15.3)</td>
      <td></td>
    </tr>
    <tr>
      <th>?</th>
      <td></td>
      <td>1 (0.3)</td>
      <td></td>
      <td>1 (1.2)</td>
      <td></td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">irradiat, n (%)</th>
      <th>no</th>
      <td></td>
      <td>218 (76.2)</td>
      <td>164 (81.6)</td>
      <td>54 (63.5)</td>
      <td>0.002</td>
    </tr>
    <tr>
      <th>yes</th>
      <td></td>
      <td>68 (23.8)</td>
      <td>37 (18.4)</td>
      <td>31 (36.5)</td>
      <td></td>
    </tr>
  </tbody>
</table>


### Data pre-processing
Based on the observations and understanding from data exploration, we pre-processed the data with [this Python script](scripts/preprocess_data.py).

#### Converting ordinal data to numeric
Given the small sample size, we decide to represent the "range" data by the mid-point. This will reduce the dimention compared to encoding as one-hot vectors. 

```python
# ordinal to numeric
curated_df['age'] = df['age'].map({t:(int(t.split('-')[0]) + int(t.split('-')[1]))/2 for t in df['age'].unique()})
curated_df['tumor-size'] = df['tumor-size'].map({t:(int(t.split('-')[0]) + int(t.split('-')[1]))/2 for t in df['tumor-size'].unique()})
curated_df['inv-nodes'] = df['inv-nodes'].map({t:(int(t.split('-')[0]) + int(t.split('-')[1]))/2 for t in df['inv-nodes'].unique()})
curated_df['deg-malig'] = df['deg-malig'].astype(float)
```

#### Converting categorical data to one-hot
We convert the true categorical data into one-hot encoding, while dropping the first category to avoid multi-collinearity. Note that dropping first category is optional for most machine learning models, but will help linear models' interpretability. 

```python
cate = pd.get_dummies(df[['menopause', 'node-caps', 'breast', 'breast-quad', 'irradiat']], drop_first=True)
```

#### Split data into training set and test set
We randomly sample 20% of instances into a test set. This is not the best practise for predictive models since temporal effect is not considered. It would be nice to use more recent data as test set. In this case, since dates are not provided, we did random sampling. 

```python 
""" Train-test split """
np.random.seed(123)
test_ids = np.random.choice(curated_df['id'], size=int(curated_df.shape[0] * 0.2), replace=False)
curated_df['train_test'] = (curated_df['id'].isin(test_ids)).map({True: 'test', False: 'train'})
curated_df['train_test'].value_counts()
```

The final dataset 
```
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
```

### Machine learning model training
The [training and evaluation pipeline](Train_eval_pipeline.py) is a Python script that runs in cmd. It takes a parameter `--config` or `-c` for each run. The configs for our experiments are available in [this folder](configs/). For example:

```cmd
python Train_eval_pipeline.py -c ./configs/RandomForestClassifier.yaml
```

When the pipeline runs, the training set and test set are loaded 

```python
feature_cols = ['age', 'tumor-size', 'inv-nodes', 'deg-malig',
    'menopause_lt40', 'menopause_premeno', 'node-caps_no', 'node-caps_yes',
    'breast_right', 'breast-quad_central', 'breast-quad_left_low',
    'breast-quad_left_up', 'breast-quad_right_low', 'breast-quad_right_up',
    'irradiat_yes']

train_features = df.loc[df['train_test'].isin(['train']), feature_cols]
train_labels = df.loc[df['train_test'].isin(['train']), 'class']

test_features = df.loc[df['train_test'].isin(['test']), feature_cols]
test_labels = df.loc[df['train_test'].isin(['test']), 'class']
``` 

Scaling is performed to boost model performance.

```python
scaler = StandardScaler()
train_features_scaled = pd.DataFrame(scaler.fit_transform(train_features), columns=train_features.columns, index=train_features.index)
test_features_scaled = pd.DataFrame(scaler.transform(test_features), columns=test_features.columns, index=test_features.index)
```

We train models with Sci-kit learn. 
```python
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
```

### Evaluation
The evaluation metrics include precision, recall, F1, accuracy, and AUROC. 

```python
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
```

We also plot an overall ROC curve with [this script](scripts/plot_ROC.py). 