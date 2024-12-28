from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
import numpy as np

ground_truth = []
ground_truth_confidence = []
predictions = []
with open('uit_vsfc\sentiment_out.txt', 'r') as file:
    for line in file:
        elements = eval(line.strip())
        ground_truth.append(elements[0])
        ground_truth_confidence.append(elements[1])
#print (ground_truth)
# print (ground_truth_confidence)
with open('uit_vsfc\sentiments.txt', 'r') as file:
    for line in file:
        predictions.append(eval(line.strip()))
# print (predictions)
with open('uit_vsfc/xxnew.txt', 'w') as file:
    file.write(str(predictions))
#Accuracy (AC), F1, AUC ROC (AR), Expected Calibration Error (ECE), and Accuracy at C% coverage (A@C)

#AC between ground_truth and predictions
correct = 0
for i in range(len(ground_truth)):
    if ground_truth[i] == predictions[i]:
        correct += 1
AC = correct / len(ground_truth)
print(f'AC: {AC}')

#F1 between ground_truth and predictions
F1 = f1_score(ground_truth, predictions, average='weighted')
print(f'F1: {F1}')

confidence = []
binary_label = []
for k in range(len(ground_truth)):
    if ground_truth[k] != predictions[k]:
        confidence.append(1 - ground_truth_confidence[k])
        binary_label.append(0)
    else:
        confidence.append(ground_truth_confidence[k])
        binary_label.append(1)

#AR between ground_truth and predictions
# AR = roc_auc_score(ground_truth, confidence, multi_class='ovo', labels=[0, 1, 2])
# print(f'AR: {AR}')

#ECE between ground_truth and predictions
prob_true, prob_pred = calibration_curve(binary_label, confidence, n_bins=10)
ECE = np.mean(np.abs(prob_pred - prob_true))
print(f'ECE: {ECE}')

#A@C between ground_truth and predictions, with C = 0.1 (10%), which is accuracy of data in top 10% of the confidence scores from ground_truth_confidence
C = 0.1
top_10 = int(len(ground_truth) * C)
top_10_indices = np.argsort(ground_truth_confidence)[-top_10:]
correct_top_10 = 0
for i in top_10_indices:
    if ground_truth[i] == predictions[i]:
        correct_top_10 += 1
A_at_C = correct_top_10 / top_10
print(f'A@C: {A_at_C}')