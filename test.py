from sklearn.metrics import f1_score
import numpy as np

y_true = np.array([0, 1, 2, 0, 1, 2])
y_pred = np.array([0, 2, 1, 0, 0, 1])

print(f1_score(y_true, y_pred, average='macro'))
