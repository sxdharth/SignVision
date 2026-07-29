Data: 700 samples, 7 classes
Classes: ['goodbye', 'hello', 'how', 'please', 'sorry', 'thanks', 'yes']
Test set: 140 samples

CLASSIFICATION REPORT
              precision    recall  f1-score   support

     goodbye       1.00      1.00      1.00        25
       hello       1.00      1.00      1.00        22
         how       1.00      1.00      1.00        21
      please       1.00      1.00      1.00        20
       sorry       1.00      1.00      1.00        14
      thanks       1.00      1.00      1.00        18
         yes       1.00      1.00      1.00        20

    accuracy                           1.00       140
   macro avg       1.00      1.00      1.00       140
weighted avg       1.00      1.00      1.00       140


CONFUSION MATRIX (rows=actual, cols=predicted)
          goodbye     hello       how    please     sorry    thanks       yes
  goodbye        25         0         0         0         0         0         0
    hello         0        22         0         0         0         0         0
      how         0         0        21         0         0         0         0
   please         0         0         0        20         0         0         0
    sorry         0         0         0         0        14         0         0
   thanks         0         0         0         0         0        18         0
      yes         0         0         0         0         0         0        20

Overall Accuracy: 100.0%

Confused Pairs:
  None — model has 100% accuracy on test set