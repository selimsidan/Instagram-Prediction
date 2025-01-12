# Instagram User Category and Like Count Prediction

This project predicts Instagram users' categories and posts' like counts using machine learning models with advanced feature engineering and classification/regression techniques.

## Overview

This repository contains the implementation and analysis of a machine learning project with two notebooks, each implementing a different model:

- **Classification**: Predicts Instagram user categories based on textual features from Instagram post captions and user-level profile attributes.
- **Regression**: Predicts the like count of a given Instagram post by leveraging temporal patterns, historical engagement, and content characteristics.

Key sections of the project include:
- Data preprocessing and feature engineering
- Model training and optimization
- Evaluation and comparison of models

### Key Files:
- **classification.ipynb**: The classification notebook containing all steps for preprocessing, feature engineering, model training, and evaluation.
- **regression.ipynb**: The regression notebook implementing time-weighted engagement predictions, feature engineering, and model optimization.

---

## Methodology

### Classification

The methodology for classification consists of four key stages:

1. **Text Preprocessing**  
   Preprocessing steps were adapted from a provided notebook, with minor customizations to better fit the data. This involved removing unnecessary characters, normalizing text, and creating a unified training corpus. Each user’s captions were combined into a single document for TF-IDF representation, ensuring alignment with the username-label mapping.  


2. **TF-IDF Vectorization**  
   Textual features were extracted using TF-IDF vectorization, which encodes the corpus into a high-dimensional sparse matrix. This process involved defining thresholds (min_df=2 and max_df=0.9) to exclude overly frequent or rare terms, ensuring the robustness of the feature space. Additionally, user profile features were preprocessed to handle missing values through imputation strategies (e.g., mode-based for categorical features, constant values for numerical features). Boolean fields were encoded as binary indicators, while categorical features were label-encoded for compatibility with machine learning models. Numerical attributes weere standardized to maintain uniform feature scaling.

3. **User-Level Feature Engineering**  
   User-level features, such as follower count and engagement metrics, were extracted using the provided notebook. The novel contributions in this step included handling missing values through imputation (e.g., replacing missing numeric values with median or mean), encoding categorical variables, and performing feature selection to identify the most informative user attributes while eliminating unnecessary ones. These processed features were then combined with the TF-IDF features into a unified dataset, capturing both textual and user-level features.  

4. **Model Training and Optimization**  
   Building upon the baseline Naive Bayes classifier, which relied solely on TF-IDF features, the enhanced approach employed a Logistic Regression model with L1-regularization. Designed to address the 10-class classification problem, the model adopted the One-vs-All (OvA) strategy, wherein a separate binary classifier was trained for each class against all others. This approach allowed efficient handling of multi-class classification while leveraging L1-regularization to mitigate overfitting and perform automatic feature selection by shrinking less informative features to zero.
Hyperparameter tuning was systematically conducted using GridSearchCV, optimizing critical parameters such as regularization strength (C) and class weighting. Robustness was further enhanced by employing cross-validated feature selection with SelectFromModel, ensuring that the model consistently identified and retained the most predictive features across different validation splits. By leveraging both textual and user-level features, the final Logistic Regression model demonstrated superior generalizability and predictive accuracy, significantly outperforming the baseline Naive Bayes classifier.


### Like Prediction (Regression)

The regression methodology consists of four key steps:

1. **Temporal Feature Engineering and Preprocessing**  
   Temporal features regarding the Instagram posts were processed to capture engagement patterns: the timestamp of the post, hour of the day, day of the week, and month. Although a number of basic temporal features could be found in the provided dataset, some extended customizations were made to highlight nuanced temporal patterns. Additionally, data cleaning included identifying missing values with respect to temporal information and standardizing datetime presentation across all rows.

2. **Historical Engagement Processing**  
   A significant part of processing historical engagement data involved implementing time-weighted metrics. This new metric uses an exponential decay function with a 7-day half-life, where higher weights are given to recent interactions, while still maintaining the relevance of older posts. Preprocessing in this context aimed to calculate engagement statistics such as the average number of likes, standard deviation of likes, and rolling average of the last 5 posts. Missing values in historical data were handled through forward-filling or zero-imputation strategies, ensuring robust feature computation even when engagement histories were incomplete.

3. **Content-Based Feature Engineering**  
   Features were extracted from both post-level and user-level data. Media types were binary encoded as video, image, and carousel. User-level features, such as verification status and number of followers, were preprocessed to handle missing values and normalized where necessary. Novel contributions in this step included creating composite features that combined multiple aspects of content characteristics, enabling the model to capture more complex patterns in user engagement.

4. **Model Architecture and Training**  
   The approach incorporated regression methods using XGBoost Regression, with hyperparameter tuning. In designing the model architecture, the nonlinear nature of social media engagement was considered, which led to the use of gradient boosting and tree-based learning. The target variable underwent log transformation (log(like_count + 1)) since social media metrics usually follow a right-skewed distribution. Hyperparameter tuning focused on the most important parameters: the number of estimators (200), learning rate (0.1), and tree depth (6), with column and row sampling (0.8) to prevent overfitting.
Systematic feature selection and importance analysis were performed, revealing that time-weighted engagement metrics and recent post performance were the strongest predictors. The final model architecture effectively combines these various features, with the time-weighted approach showing strong predictive power for engagement patterns. Cross-validation was used to ensure robust performance assessment, and Log Mean Squared Error was chosen as the metric because it can handle the wide range of values typically found in social media data.

---

## Results

### Classification

The evaluation of our methodology for classification prediction was carried out by comparing the performance of the baseline model, which utilizes only TF-IDF features with a Naive Bayes classifier, against the proposed final model that integrates additional profile-based features and employs Logistic Regression with optimized hyperparameters.

The baseline Naive Bayes model achieved a cross-validation score of 0.59, indicating moderate performance in predicting Instagram user categories based solely on text features extracted through TF-IDF vectorization. In contrast, our final Multinomial Logistic Regression model significantly enhanced predictive accuracy by incorporating user metadata and leveraging a more advanced logistic regression framework. Through feature preprocessing, selection, and scaling, alongside hyperparameter tuning, the final model demonstrated superior generalization capability. The cross-validation score for this improved model rose to 0.74, reflecting a 15 percentage point improvement over the baseline.

Detailed performance results of our final model and its confusion matrix for the 10 predicted categories could be found below.

### Key Performance Metrics

<img width="605" alt="logistic_regression" src="https://github.com/user-attachments/assets/a9ecb4e3-ba3c-48de-a178-5f186ed89401" />

<img width="838" alt="cross_validation" src="https://github.com/user-attachments/assets/22c4c3b7-e10b-4b68-8bd3-ce59a6b5bccf" />



### Confusion Matrix

<img width="635" alt="confusion_matrix" src="https://github.com/user-attachments/assets/8fffb8c8-2227-4484-b0f0-cbca2ad587de" />


### Like Prediction

We evaluated our methodology for like count prediction by comparing the performance of a baseline model, which relies on simple historical averages, with our advanced XGBoost model that incorporates temporal patterns and sophisticated feature engineering. The baseline model, which predicts the average number of likes a user receives based on their past activity, achieved a Log MSE of 1.227, reflecting moderate performance.

In comparison, our XGBoost Regression model significantly improved prediction accuracy by leveraging time-weighted features and content characteristics. It achieved a Log MSE of 0.522 on the training set and 0.584 on the test set, marking a 52.4% improvement over the baseline.

We tested the model using a large dataset, consisting of 94,824 training samples and 92,478 test samples. Feature importance analysis showed that time-weighted engagement metrics were the most influential predictors, contributing 49.62% to the model's performance. Rolling average engagement features followed with 23.67%, while user average likes accounted for 16.42%. This confirms the effectiveness of our approach, which emphasizes recent engagement patterns through time-weighted features.

### Feature Importance

<img width="585" alt="feature_importance" src="https://github.com/user-attachments/assets/f7c505ba-866f-4829-baa8-3bf32387206a" />

### Feature Correlation Heatmap

<img width="540" alt="feature_correlation_heatmap" src="https://github.com/user-attachments/assets/120629e0-ab50-4e07-b950-b8b034e69d3f" />

---

## Contributions

- **Selim Sıdan**: Responsible for the entire classification part of the project, including feature engineering, model training, and optimization. For feature engineering, enhanced the provided TF-IDF implementation by incorporating English stopwords and tuning parameters like max_df and min_df to reduce overfitting and improve feature quality. Additionally, preprocessed user-level features by imputing missing values, encoding categorical variables, and standardizing numerical attributes, ensuring a seamless integration of user-level and TF-IDF features. In terms of model development, replaced the baseline Naive Bayes model, which used only TF-IDF features, with a Logistic Regression model employing L1 regularization. This change enabled feature selection and improved handling of the multi-class classification task. Finally, conducted hyperparameter tuning using GridSearchCV to optimize parameters such as regularization strength and class weighting, leading to a significantly more robust and accurate classification model.
  
- **Saner Bilir**: Enhanced the regression model through feature engineering and hyperparameter optimization. Developed key derived features including time-weighted engagement metrics using exponential decay, temporal features from timestamps (hour, day, month), and historical engagement patterns (rolling averages, standard deviations, maximum likes). Implemented user-level features that capture engagement history while emphasizing recent activities through time-weighted calculations. For model optimization, conducted systematic hyperparameter tuning of XGBoost parameters including number of estimators (200), learning rate (0.1), maximum tree depth (6), and sampling ratios (0.8 for both column and row sampling).
  
- **Fırat Yurdakul**: Built a dataframe for the most correlated features on the given dataset and found 5 best most correlated features. After finding these features, used linear regression for the like prediction part in order to improve MSE yet got worse results than xgboost. Also tried Random forest for classification part but compared to Logistic Regression it didn’t improve the model. Wrote explanations for the code in regression notebook. 
  
- **Mert Polat**: Worked on feature engineering on the regression model. Extracted rich features for the XGBoost algorithm like recent like counts and their rolling average, standart deviation. Since an account’s recent activity reflect the current engagement of that account these features helped get lower MSE for the regression model.

---

