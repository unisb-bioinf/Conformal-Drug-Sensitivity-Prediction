from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('C:\Users\User\Documents\Uni\WS2223\HiWi\Conformal-Drug-Sensitivity-Prediction\Example_Data\Irinotecan_scores.txt', sep = '\t', names=['cl','score'])
train, test = train_test_split(df, test_size=0.15)
train, cal = train_test_split(train, test_size=0.15/0.85)
train.to_csv('Training_Irinotecan_scores.txt', sep = '\t', header = False, index = False)
test.to_csv('Test_Irinotecan_scores.txt', sep = '\t', header = False, index = False)
cal.to_csv('Calibration_Irinotecan_scores.txt', sep = '\t', header = False, index = False)