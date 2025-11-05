# Credit default prediction - logistic regression model

1) Used the BRS-2016-HMEQ dataset (Baesens et al., 2016) containing 5960 credit applications.

- Target variable: bad (1 = default, 0 = non-default);
- Missing values: 5271;
- 10 numerical and 2 categorical features (income, debt, credit history length, job type, etc.)

2) **Performed exploratory data analysis (EDA)**

- Identified strong class imbalance - defaults represent a minority class;
- Key predictors of default: 1) delinq (number of delinquent credit lines, r = 0.354), 2) derog (derogatory reports, r = 0.276), 3) debtinc (debt-to-income ratio, r = 0.20), 4) ninq (recent credit inquiries, r = 0.175);
- Protective factors: 1) clage (age of oldest credit line, r = -0.17), 2) loan (loan amount, r = -0.075)

3) Handled missing values and categorical variables via one-hot encoding, applied feature scaling, and used SMOTE oversampling to balance the target classes

4) Built **a logistic regression model** to predict credit default probability

- Implemented using Pipeline and RandomizedSearchCV for tuning;
- Checked feature significance and multicollinearity

5) Evaluated model performance using classification_report and f1_score. After class balancing, the model achieved strong recall and F1 for default prediction

6) Applied PCA for visualization of class separation and feature informativeness

7) **Result**: delivered an interpretable credit risk scoring model with clear insight into key risk drivers, suitable for integration into credit decision workflows
