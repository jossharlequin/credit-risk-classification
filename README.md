
# Import the modules
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
Split the Data into Training and Testing Sets
Step 1: Read the lending_data.csv data from the Resources folder into a Pandas DataFrame.
# Read the CSV file from the Resources folder into a Pandas DataFrame
df = pd.read_csv(
    Path('Resources/lending_data.csv')
)

# Review the DataFrame
df.head()
loan_size	interest_rate	borrower_income	debt_to_income	num_of_accounts	derogatory_marks	total_debt	loan_status
0	10700.0	7.672	52800	0.431818	5	1	22800	0
1	8400.0	6.692	43600	0.311927	3	0	13600	0
2	9000.0	6.963	46100	0.349241	3	0	16100	0
3	10700.0	7.664	52700	0.430740	5	1	22700	0
4	10800.0	7.698	53000	0.433962	5	1	23000	0
Step 2: Create the labels set (y) from the “loan_status” column, and then create the features (X) DataFrame from the remaining columns.
# Separate the data into labels and features

# Separate the y variable, the labels
y = df['loan_status']

# Separate the X variable, the features
X = df.drop(columns=['loan_status'])
# Review the y variable Series
print(y.head())
0    0
1    0
2    0
3    0
4    0
Name: loan_status, dtype: int64
# Review the X variable DataFrame
print(X.head())
   loan_size  interest_rate  borrower_income  debt_to_income  num_of_accounts  \
0    10700.0          7.672            52800        0.431818                5   
1     8400.0          6.692            43600        0.311927                3   
2     9000.0          6.963            46100        0.349241                3   
3    10700.0          7.664            52700        0.430740                5   
4    10800.0          7.698            53000        0.433962                5   

   derogatory_marks  total_debt  
0                 1       22800  
1                 0       13600  
2                 0       16100  
3                 1       22700  
4                 1       23000  
Step 3: Split the data into training and testing datasets by using train_test_split.
# Import the train_test_learn module
from sklearn.model_selection import train_test_split

# Split the data using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# Assign a random_state of 1 to the function
Create a Logistic Regression Model with the Original Data
Step 1: Fit a logistic regression model by using the training data (X_train and y_train).
# Import the LogisticRegression module from SKLearn
from sklearn.linear_model import LogisticRegression

# Instantiate the Logistic Regression model
model = LogisticRegression(random_state=1)

# Fit the model using training data
model.fit(X_train, y_train)
LogisticRegression(random_state=1)
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
Step 2: Save the predictions on the testing data labels by using the testing feature data (X_test) and the fitted model.
# Make a prediction using the testing data
y_pred = model.predict(X_test)
Step 3: Evaluate the model’s performance by doing the following:
Generate a confusion matrix.

Print the classification report.

# Generate a confusion matrix for the model
conf_matrix = confusion_matrix(y_test, y_pred)
# Print the classification report for the model
print(classification_report(y_test,y_pred))
              precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.85      0.91      0.88       619

    accuracy                           0.99     19384
   macro avg       0.92      0.95      0.94     19384
weighted avg       0.99      0.99      0.99     19384

Step 4: Answer the following question.
Question: How well does the logistic regression model predict both the 0 (healthy loan) and 1 (high-risk loan) labels?

Answer: The model is good a predicting both 0 and 1 results. It, however, is better at predicting the 0 results at a 1.00 precision, compared to a 0.85 precision score for 1.


# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
* Explain what financial information the data was on, and what you needed to predict.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
* Describe the stages of the machine learning process you went through as part of this analysis.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any other algorithms).

* The purpose of this analysis was to determine loan risk based on income, loan amount, debt, and other factors. This machine learning was done by removing the existing target variable and then split into Test and Training groups. The dataset was set through a Logistic Regression and then A Confusion Matrix to make a prediction of values, using the non-target variables, and then compared to the actual results.

## Results

Logistic Regression Model:
-Accuracy: 0.99
  This shows the 99% of plots were predicted correctly.
-Precision for class 0 (healthy loan): 1.00
  This means that of the 0's, or True Nagatives, all the ones predicted were actually 0's in actuality.
-Recall for class 0 (healthy loan): 0.99
  This shows that of all the true negatives, 99% were accurately predicted by the model.
-Precision for class 1 (high-risk loan): 0.85
  This means that of the 1's, or True Positives, 85% of the ones predicted were actually 1's in actuality.
-Recall for class 1 (high-risk loan): 0.91
  This shows that of all the true positives, 91% were accurately predicted by the model.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:

* Which one seems to perform best? How do you know it performs best?
*   Of the True Positives, and True Negatives, the True Negatives are predicted the best at a 100% and a 99% recall. This means that the model is great at predicting True Negatives, or Healthy Loans. This is great, because it means the credit agency will be able to predict accurately the borrowers who are low-risk. 
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )
*   For this case, of a creditor determining risk of possible borrowers, I believe the ability to predict high-risk borrowers is more important than the ability to predict low-risk borrowers. This is because if a low-risk borrower is denied because they are modeled as high-risk the bank loses nothing aside from a customer. However, if the bank predicts a high-risk borrower as a low-risk borrower than the bank could lose their whole loan amount, or most of it, of the customer defaults on their loan.

If you do not recommend any of the models, please justify your reasoning.


