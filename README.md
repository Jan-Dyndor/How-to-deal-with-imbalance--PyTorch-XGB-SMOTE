# 🧾 Project Description

Handling imbalanced datasets is a fundamental challenge in machine learning. This project explores and compares various resampling techniques and machine learning models to address class imbalance using a real-world tabular dataset.

Both traditional (shallow) machine learning algorithms and a simple deep learning model were trained under four different scenarios:
- **a) Raw data** – only basic preprocessing was applied, without any balancing technique  
- **b) Undersampling** – majority class was reduced to match the minority class (preprocessing was applied)
- **c) Oversampling** – minority class was duplicated to balance the dataset (preprocessing was applied)
- **d) SMOTE** – synthetic data points were generated to oversample the minority class (preprocessing was applied)

The project follows a structured machine learning pipeline including:
- data preprocessing  
- resampling (balancing) techniques  
- model training 
- evaluation using appropriate metrics  

To support interpretability, **Principal Component Analysis (PCA)** was also conducted to visualize how each resampling technique affects the distribution of the data in feature space.

# 🎯 Objective

The main goals of this project are:

- **Investigating Class Imbalance Solutions**: Apply and compare resampling techniques such as SMOTE, Random Undersampling nad Oversampliing
- **Modeling**: Build custom models using deep learning and tree-based algorithms, examining their performance under imbalance conditions.
- **Evaluation Using Advanced Metrics**: Go beyond accuracy by analyzing precision, recall, F1-score, AUC PR.
- **Visual Comparison of Techniques**: Illustrate how balancing methods influence model behavior and misclassifications.

# 📁 Dataset

- **Type**: Tabular classification dataset with imbalanced target variable.
- **Source**: Public dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
- **Preparation**:
  - Standardization features.
  - Train-test split using stratified sampling to maintain class ratios.
  - Resampling using SMOTE / Random Undersampling / Oversampling.

# 🧠 Models

- **PyTorch Model**:
  - Multi-layer fully connected neural network.
- **XGBoost, Logistic Regression, Random Forest, KNN**:
  - Trained on raw data, Oversampled, Undersampled. Used RandomizedSearchCV, GridSearchCV for hyperparameter tuning.
- **XGBoost , Random Forest**
  - Trained on SMOTE
- **PCA** 


# 🛠️ Tools and Libraries

- Python
- PyTorch
- XGBoost
- Scikit-learn
- imbalanced-learn (SMOTE, RandomUnderSampler, RandomOverSampler)
- NumPy, Pandas
- Matplotlib

# 📊 Results

#### Summary of Key Experiments
In the basic models the best one was Random Forest and the most promising technique (Over,UnderSample and raw data) was Oversampling:

| Metric      | RF Oversampled Final | RF No Manipulation Final | RF Undersampled Final |
|-------------|----------------------|---------------------------|------------------------|
| F1 Macro    | 0.950627             | 0.912293                  | 0.563900              |
| F1 Micro    | 0.999693             | 0.999456                  | 0.979724              |
| Recall      | 0.820513             | 0.744898                  | 0.948718              |
| Precision   | 1.000000             | 0.924051                  | 0.074447              |
| Accuracy    | 0.999693             | 0.999456                  | 0.979724              |
| AUC PR      | 0.929177             | 0.824086                  | 0.707043              |

XGBoost model trained on Over and Undersample also turned out to have better results on OverSampling.

#### 📊 XGBoost - Summary of Results

| Metric      | XGB Final Over       | XGB Final Undersample     |
|-------------|----------------------|----------------------------|
| F1 Macro    | 0.941460             | 0.536920                  |
| F1 Micro    | 0.999605             | 0.967347                  |
| Recall      | 0.871795             | 0.948718                  |
| Precision   | 0.894737             | 0.047497                  |
| Accuracy    | 0.999605             | 0.967347                  |
| AUC PR      | 0.917100             | 0.794623                  |

#### 📊 SMOTE - Random Forest vs XGBoost
Also SMOTE was applied - only on Ranfom Forest and XGBoost

| Metric      | RF SMOTE              | XGB SMOTE              |
|-------------|------------------------|------------------------|
| F1 Macro    | 0.934101               | 0.909474              |
| F1 Micro    | 0.999561               | 0.999342              |
| Recall      | 0.846154               | 0.871795              |
| Precision   | 0.891892               | 0.772727              |
| Accuracy    | 0.999561               | 0.999342              |
| AUC PR      | 0.910549               | 0.906436              |

Deep Learning models did not perform well. 

### 🔍 Observations
- Random Forest + Oversampling achieved the highest F1 Macro (0.9506) and perfect Precision (1.0), indicating exceptional performance in distinguishing between both classes with minimal false positives. This result suggests the model is extremely cautious, possibly at the cost of missing some true positives (Recall = 0.82).
-XGBoost + Oversampling was the second-best overall — or the best, depending on which metrics matter most for the business — with high F1 Macro (0.9414), strong Recall (0.8718), and solid AUC PR (0.9171).
- SMOTE also performed well for both models, especially boosting Recall, but slightly below the performance of pure oversampling.

# 📌 Conclusions

This project demonstrates the effectiveness of combining resampling techniques with different modeling approaches to handle class imbalance. In particular:
- Oversampling proved to be the most effective resampling technique across models.
- Undersampling led to the weakest results.
- Random Forest showed better stability and robustness across different balancing methods.
- While XGBoost performs well, its precision drops more significantly under undersampling scenarios.
- SMOTE offers a balanced alternative, improving recall while keeping decent precision and AUC PR (Note: The models trained on SMOTE data were not tuned with hyperparameter optimization. Therefore, their performance might improve with further tuning).

# 🎓 Lessons Learned & Future Directions

- Resampling techniques like SMOTE, Over and UncderSampling are powerful tools but should be used cautiously (risk of overfitting).
- PyTorch models may require more advanced architectures or regularization strategies to match tree-based performance.
- Future improvements:
  - Cross-validation for better generalization estimates (higher cv like 5).
  - Hyperparameter tuning with Optuna or HalvingRandomSearchCV .
  - Exploration of other imbalance techniques: ADASYN.


# 🙋‍♂️ Author

**Jan Dyndor**  
ML Engineer & Pharmacist  
📧 dyndorjan@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/jan-dyndor/)  
📊 [Kaggle](https://www.kaggle.com/jandyndor)  
💻 [GitHub](https://github.com/Jan-Dyndor)

# 🧠 Keywords

imbalanced dataset, SMOTE, XGBoost, PyTorch, binary classification, machine learning, oversampling, undersampling deep learning, precision-recall, F1-score, tabular data

