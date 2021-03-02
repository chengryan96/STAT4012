# please run it under linux base os
# import required packages
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt

# read the dataset
csv_path = os.path.abspath(os.path.join('__file__', '..', '..', 'csv'))
jpg_path = os.path.abspath(os.path.join('__file__', '..', '..', 'jpg'))
fin_ratio = pd.read_csv(os.path.join(csv_path, 'fin-ratio.csv'))

# split the data into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(fin_ratio.drop(
    ['HSI'], axis='columns'), fin_ratio['HSI'], test_size=0.3, random_state=4012)

# standardize the data
scaler = StandardScaler()
X_train_stand = scaler.fit_transform(X_train)
X_test_stand = scaler.transform(X_test)

# fit the data into logistic regression
log_reg = LogisticRegression(random_state=4012, penalty="none")
log_reg.fit(X_train_stand, y_train)
y_pred = log_reg.predict(X_test_stand)
accuracy_score(y_test, y_pred)

# plot roc curve
y_pred_proba = log_reg.predict_proba(X_test_stand)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
fig = plt.figure()
plt.style.use('ggplot')
plt.plot(fpr, tpr, label="Logistic regression")
plt.xlabel("True positive rate (tpr)")
plt.ylabel("False positive rate (fpr)")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title('ROC curve')
plt.legend(loc="lower right")
fig.savefig(os.path.join(jpg_path, 'q2.jpg'))
