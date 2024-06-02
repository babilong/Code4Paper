Please modify the code in lines 331 and 347 in the Test.py file to:
y_pred_all.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
y_pred_all.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
Indicates that the first categorization classifies the botnet family incorrectly and is randomly set to Gafgyt or Mirai, which does not affect the evaluation metrics.
