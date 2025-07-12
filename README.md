# CSI_Week06

Train multiple machine learning models and evaluate their performance using metrics such as accuracy, precision, recall, and F1-score. Implement hyperparameter tuning techniques like GridSearchCV and RandomizedSearchCV to optimize model parameters. Analyze the results to select the best-performing model.

## Machine Learning Assignment: Model Evaluation & Hyperparameter Tuning  
**Dataset:** Wine Dataset from `sklearn.datasets`  
**Goal:** Compare models and improve them using GridSearchCV and RandomizedSearchCV

---

## Objectives
- Train multiple models  
- Evaluate using Accuracy, Precision, Recall, F1-score  
- Tune hyperparameters using:  
  - GridSearchCV on Random Forest  
  - RandomizedSearchCV on Decision Tree  

---

## Models Used
- Logistic Regression  
- Decision Tree  
- Random Forest  

---

## Evaluation Metrics
- Confusion Matrix  
- Accuracy  
- Precision  
- Recall  
- F1-score (macro)  

---

## Hyperparameter Tuning

**GridSearchCV**  
Tuned parameters for `RandomForestClassifier`:  
- n_estimators  
- max_depth  
- min_samples_split  

**RandomizedSearchCV**  
Tuned parameters for `DecisionTreeClassifier`:  
- max_depth  
- min_samples_split  
- min_samples_leaf  

---

## Results Summary

| Model                             | Accuracy | Precision       | Recall              | F1-Score |
|----------------------------------|----------|------------------|----------------------|----------|
| Logistic Regression              | ~97%     | High             | High                 | High     |
| Decision Tree                    | ~94%     | High             | Slightly overfitting | Decent   |
| Random Forest                    | ~98%     | Excellent        | Excellent            | Excellent|
| Random Forest (GridSearchCV)     | ↑ Improved | Best overall   | Best                 | Best     |
| Decision Tree (RandomizedSearchCV) | ↑ Improved | More stable  | Less overfit         | Better   |

---

## Tools & Environment
- Google Colab  
- Python 3.x  
- sklearn, pandas, numpy, matplotlib, seaborn  

---

## Conclusion
- Hyperparameter tuning significantly improved model performance  
- Random Forest with GridSearchCV was the best model  
- EDA helped understand the distribution and shape of data before training  
