PATH = r'/home/daviden1013/David_projects/Qualifying_exam_C/'

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

""" Load gold labels """
df = pd.read_pickle(os.path.join(PATH, 'data', 'curated_df.pkl'))
gold = df.loc[df['train_test'].isin(['test']), 'class']

""" Load predicted probabilities """
LR_arr = np.load(os.path.join(PATH, 'y_prob', 'LogisticRegression.npy'))
RG_arr = np.load(os.path.join(PATH, 'y_prob', 'RidgeRegression.npy'))
FR_arr = np.load(os.path.join(PATH, 'y_prob', 'RandomForestClassifier.npy'))
XB_arr = np.load(os.path.join(PATH, 'y_prob', 'XGBClassifier.npy'))

""" Plot ROC """
plt.figure(figsize=(8, 6))

for arr, mode_name in zip([LR_arr, RG_arr, FR_arr, XB_arr], ["LogisticRegression", "RidgeRegression", "RandomForestClassifier", "XGBClassifier"]):
    fpr, tpr, thresholds = roc_curve(gold, arr[:,1], pos_label=1)
    auroc = roc_auc_score(gold, arr[:,1])
    plt.plot(fpr, tpr, lw=2, marker='o', label=f'{mode_name} (AUROC = {auroc:.2f})')

plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig(os.path.join(PATH, 'ROC.png'), dpi=300, bbox_inches='tight')
plt.show()
