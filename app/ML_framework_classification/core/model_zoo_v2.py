# app/ML_framework_classification/core/model_zoo_v2.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scipy.stats import uniform, randint, loguniform

def get_model_zoo_with_search_spaces():
    return [
        {
            "name": "Logistic Regression",
            "model": LogisticRegression(max_iter=1000),
            "param_distributions": {
                "classifier__C": uniform(0.01, 100),
                "classifier__penalty": ["l1", "l2"],
                "classifier__solver": ["liblinear", "saga"],
                "classifier__class_weight": [None, "balanced"]
            }
        },
        {
            "name": "Random Forest",
            "model": RandomForestClassifier(random_state=42),
            "param_distributions": {
                "classifier__n_estimators": randint(100, 500),
                "classifier__max_depth": [None, 10, 20, 30],
                "classifier__min_samples_split": randint(2, 10),
                "classifier__class_weight": [None, "balanced"]
            }
        },
        {
            "name": "XGBoost",
            "model": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss"),
            "param_distributions": {
                "classifier__n_estimators": randint(100, 500),
                "classifier__max_depth": randint(3, 10),
                "classifier__learning_rate": uniform(0.01, 0.3),
                "classifier__subsample": uniform(0.6, 0.4)
            }
        },
        # {
        #     "name": "LightGBM",
        #     "model": LGBMClassifier(random_state=42),
        #     "param_distributions": {
        #         "classifier__n_estimators": randint(100, 500),
        #         "classifier__num_leaves": randint(20, 100),
        #         "classifier__learning_rate": uniform(0.01, 0.3),
        #         "classifier__class_weight": [None, "balanced"]
        #     }
        # },
        {
            "name": "CatBoost",
            "model": CatBoostClassifier(random_state=42, verbose=0),
            "param_distributions": {
                "classifier__iterations": randint(100, 500),
                "classifier__depth": randint(4, 10),
                "classifier__learning_rate": uniform(0.01, 0.3),
                "classifier__l2_leaf_reg": randint(1, 10)
            }
        },
        {
            "name": "SVM",
            "model": SVC(probability=True),
            "param_distributions": {
                "classifier__C": uniform(0.1, 100),
                "classifier__kernel": ["linear", "rbf"],
                "classifier__class_weight": [None, "balanced"]
            }
        },
        {
            "name": "KNN",
            "model": KNeighborsClassifier(),
            "param_distributions": {
                "classifier__n_neighbors": randint(3, 15),
                "classifier__weights": ["uniform", "distance"],
                "classifier__p": [1, 2]
            }
        },
        {
            "name": "Naive Bayes",
            "model": GaussianNB(),
            "param_distributions": {
                "classifier__var_smoothing": loguniform(1e-10, 1e-8)
            }
        }
    ]