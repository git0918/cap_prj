### Problem Statement

Heart disease is the major cause of morbidity and mortality globally.  It is difficult to identify high risk patients because of the multifactorial nature of several contributory risk factors such as diabetes, high blood pressure, high cholesterol etc.

As a data scientist, I was asked to predict the disease by using modern approaches as  Machine Learning that is proved to be effective in assisting decision making and risk assessment from the large quantity of data produced by the healthcare industry on heart disease.

My goal is to explore different machine learning approaches for predicting whether a patient has 10 year risk of developing coronary heart disease (CHD).

### Executive Summary

- Explored multicollinearity of features
- Removed outliers using Mahalanobis Distance
- Binarize levels of categorical features to reduce multicollinearity and curse of dimensionality
- Identified the most important features that contribute to heart disease
- Applied SMOTE to balance inbalanced classes
- Applied gridsearch cross validation to find optimal hyperparameters
- Compared the performance of models using F1 score and ROC AUC as evaluation metrics

### Data Collection

The dataset I use provides the patients information and it includes over 4,240 records and 16 attributes. Each attribute is a potential risk factor.
There are on demographic, behavioral and medical risk factors:
- Demographic
    - Sex: male or female
    - Age: Age of the patient
- Behavioral
    - Current Smoker
    - Cigs Per Day
- Information on medical history
    - BP Meds  
    - Prevalent Stroke  
    - Prevalent Hyp  
    - Diabetes:
- Information on current medical condition:
    - Tot Chol  
    - Sys BP  
    - Dia BP  
    - BMI  
    - Heart Rate  
    - Glucose
- Target TenYearCHD:
    - 0 = Less chance of heart attach
    - 1 more chance of heart attack

### EDA

- Part 1: Understand the data
- Part 2: Check the null model
- Part 3: perform data cleanup
- Part 4: Analyze the correlation with the Heatmap
- Part 5: Analyze features in details

### Pre-Processing

- Feature Selections: Use the Gini importance (or mean decrease impurity) method, which is computed from the Random Forest structure to get my top features that impact the heart disease the most.
- SMOTE:  It’s really essential that the dataset we are working on should be approximately balanced. An extremely imbalanced dataset can render the whole model training useless and thus, will be of no use.  The class in the dataset is very imbalanced.  After applying SMOTE, the new dataset is much more balanced: the new ratio between negative and positive cases is 1:1.2 up from 1:5.57

### Model selection

Using a GridSearchCV hyperparameter optimization, we selected models to build and examine:
- Logistic Regression
- KNN
- Decision Tree
- SVM

### Model Optimization
I use the search for optimum parameters using GridSearchCV

### Conclusions and Recommendations

- The most important features in predicting the ten year risk of developing CHD were age and systolic blood pressure
- The Support vector machine with the radial kernel was the best performing model in terms of accuracy and the F1 score. Its high AUC shows that it has a high true positive rate.
- Balancing the dataset by using the SMOTE technique helped in improving the models' sensitivity, this is when compared to the performance metrics of other models on different notebooks on the same dataset
- With more data(especially that of the minority class) better models can be built
