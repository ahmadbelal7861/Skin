import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
import cv2
import matplotlib.pyplot as plt
import os
import keras

_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: {:.2f}%'.format(accuracy * 100))
print('Error Rate: {:.2f}%'.format((1 - accuracy) * 100))


from sklearn.metrics import accuracy_score
y_true = np.argmax(y_test, axis=1)
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_true, y_pred)
print('Accuracy: {:.2f}%'.format(accuracy * 100))
print('Error rate: {:.2f}%'.format((1 - accuracy) * 100))


####---ROC---####
from sklearn.metrics import roc_curve, auc
y_scores = model.predict(X_test)

# ROC of each classes
n_classes = 7
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print('AUC class {}: {:.4f}'.format(i, roc_auc[i]))

# ROC of micro-average
fpr['micro'], tpr['micro'], _ = roc_curve(y_test.ravel(), y_scores.ravel())
roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
print('AUC micro-average: {:.4f}'.format(roc_auc['micro']))

# ROC of macro-average
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes
fpr['macro'] = all_fpr
tpr['macro'] = mean_tpr
roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
print('AUC macro-average: {:.4f}'.format(roc_auc['macro']))


####---AUC---####
plt.figure(figsize=(8, 6), dpi=100)
plt.plot([0, 1])
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC class {} (area = {:.4})'.format(i, roc_auc[i]), linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.title('ROC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.show()
