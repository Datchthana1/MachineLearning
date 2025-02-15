import time
from itertools import combinations

import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sampling_strategy = "not majority"
nsplits = [5, 7, 10]

params = {
    "ADASYN": {
        "EDA_Phasic_EmotiBit+EDA_Tonic_EmotiBit": {
            "Random Forest": RandomForestClassifier(warm_start= False, random_state= 42, n_estimators= 800, min_samples_split= 3, min_samples_leaf= 4, max_samples= 0.5, max_features= 0.5, max_depth= 20, criterion= "entropy", class_weight= "balanced_subsample"),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators= 700, max_depth= 9, learning_rate= 0.6),
            "XGBoost": XGBClassifier(subsample= 0.7, n_estimators= 600, max_depth= 11, learning_rate= 0.05, colsample_bytree= 1.0, alpha= 1.0),
            "AdaBoosting": AdaBoostClassifier(algorithm= "SAMME", estimator= DecisionTreeClassifier(max_depth=5), learning_rate= 0.7, n_estimators= 50, random_state= 42),
            "CatBoost": CatBoostClassifier(learning_rate= 0.4, max_depth= 9, n_estimators= 900),
            "LightGBM": LGBMClassifier(learning_rate= 0.4, max_depth= 9, n_estimators= 900, num_leaves= 1024)
        },
        "EDA_Phasic_EmotiBit+lf_PPG": {
            "Random Forest": RandomForestClassifier(warm_start= True, random_state= 42, n_estimators= 1500, min_samples_split= 6, min_samples_leaf= 4, max_samples= 0.5, max_features= "sqrt", max_depth= 15, criterion= "gini", class_weight= "balanced"),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=900, max_depth=8, learning_rate=0.1),
            "XGBoost": XGBClassifier(subsample= 0.7, n_estimators= 900, max_depth= 8, learning_rate= 0.1, colsample_bytree= 1.0, alpha= 0.5),
            "AdaBoosting": AdaBoostClassifier(algorithm= "SAMME", estimator= DecisionTreeClassifier(max_depth=5, min_samples_leaf=4), learning_rate= 0.9, n_estimators= 1200, random_state= 42),
            "CatBoost": CatBoostClassifier(n_estimators=900, max_depth=5, learning_rate=0.6),
            "LightGBM": LGBMClassifier(num_leaves=8, n_estimators=900, max_depth=8, learning_rate=0.3)
        },
        "EDA_Phasic_EmotiBit+hf_PPG": {
            "Random Forest": RandomForestClassifier(warm_start= False, random_state= 42, n_estimators= 200, min_samples_split= 3, min_samples_leaf= 3, max_samples= 0.5, max_features= "sqrt", max_depth= 10, criterion= "entropy", class_weight= "balanced_subsample"),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=500, max_depth=8, learning_rate=0.6),
            "XGBoost": XGBClassifier(subsample= 1.0, n_estimators= 800, max_depth= 6, learning_rate= 0.4, colsample_bytree= 0.9, alpha= 0.5),
            "AdaBoosting": AdaBoostClassifier(random_state=42, n_estimators=300, learning_rate=0.7,
                                              estimator=DecisionTreeClassifier(max_depth=5), algorithm="SAMME"),
            "CatBoost": CatBoostClassifier(n_estimators=800, max_depth=7, learning_rate=0.3),
            "LightGBM": LGBMClassifier(num_leaves=512, n_estimators=200, max_depth=6, learning_rate=0.2)
        },
        "EDA_Phasic_EmotiBit+lf_hf_PPG": {
            "Random Forest": RandomForestClassifier(class_weight= None, criterion= "entropy", max_depth= None, max_features= "log2", max_samples= None, min_samples_leaf= 9, min_samples_split= 10, n_estimators= 1000, random_state= 42, warm_start= True),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=600, max_depth=5, learning_rate=0.3),
            "XGBoost": XGBClassifier(alpha= 0.5, colsample_bytree= 0.9, learning_rate= 0.2, max_depth= 8, n_estimators= 100, subsample= 0.9),
            "AdaBoosting": AdaBoostClassifier(random_state= 42, n_estimators= 900, learning_rate= 0.4, estimator= DecisionTreeClassifier(max_depth=5), algorithm= "SAMME"),
            "CatBoost": CatBoostClassifier(n_estimators=800, max_depth=8, learning_rate=0.3),
            "LightGBM": LGBMClassifier(num_leaves=512, n_estimators=300, max_depth=2, learning_rate=0.9)
        },
        "EDA_Tonic_EmotiBit+lf_PPG": {
            "Random Forest": RandomForestClassifier(class_weight= "balanced", criterion= "entropy", max_depth= None, max_features= "log2", max_samples= 0.5, min_samples_leaf= 4, min_samples_split= 7, n_estimators= 1500, random_state= 42, warm_start= True),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=800, max_depth=8, learning_rate=0.4),
            "XGBoost": XGBClassifier(subsample= 0.9, n_estimators= 200, max_depth= 7, learning_rate= 0.25, colsample_bytree= 0.9, alpha= 0.1),
            "AdaBoosting": AdaBoostClassifier(algorithm= "SAMME", estimator= DecisionTreeClassifier(max_depth=4, min_samples_leaf=2), learning_rate= 1.0, n_estimators= 700, random_state= 42),
            "CatBoost": CatBoostClassifier(n_estimators=900, max_depth=9, learning_rate=0.2),
            "LightGBM": LGBMClassifier(num_leaves=1024, n_estimators=700, max_depth=8, learning_rate=0.2)
        },
        "EDA_Tonic_EmotiBit+hf_PPG": {
            "Random Forest": RandomForestClassifier(warm_start= True, random_state= 42, n_estimators= 1500, min_samples_split= 6, min_samples_leaf= 6, max_samples= 0.9, max_features= "log2", max_depth= 10, criterion= "entropy", class_weight= None),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=500, max_depth=7, learning_rate=0.6),
            "XGBoost": XGBClassifier(alpha= 1.0, colsample_bytree= 0.9, learning_rate= 0.4, max_depth= 9, n_estimators= 400, subsample= 0.8),
            "AdaBoosting": AdaBoostClassifier(algorithm= "SAMME", estimator= DecisionTreeClassifier(max_depth=5, min_samples_leaf=4), learning_rate= 0.5, n_estimators= 800, random_state= 42),
            "CatBoost": CatBoostClassifier(n_estimators=400, max_depth=8, learning_rate=0.8),
            "LightGBM": LGBMClassifier(num_leaves=32, n_estimators=100, max_depth=8, learning_rate=0.4)
        },
        "EDA_Tonic_EmotiBit+lf_hf_PPG": {
            "Random Forest": RandomForestClassifier(class_weight= None, criterion= "gini", max_depth= None, max_features= 0.7, max_samples= 0.5, min_samples_leaf= 2, min_samples_split= 6, n_estimators= 600, random_state= 42, warm_start= True),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=400, max_depth=9, learning_rate=0.3),
            "XGBoost": XGBClassifier(alpha= 0.1, colsample_bytree= 1.0, learning_rate= 0.3, max_depth= 4, n_estimators= 200, subsample= 1.0),
            "AdaBoosting": AdaBoostClassifier(random_state= 42, n_estimators= 100, learning_rate= 0.8, estimator= DecisionTreeClassifier(max_depth=5, min_samples_leaf=2), algorithm= 'SAMME'),
            "CatBoost": CatBoostClassifier(n_estimators=800, max_depth=7, learning_rate=0.1),
            "LightGBM": LGBMClassifier(num_leaves=32, n_estimators=500, max_depth=5, learning_rate=0.2)
        },
        "lf_PPG+hf_PPG": {
            "Random Forest": RandomForestClassifier(warm_start= False, random_state= 42, n_estimators= 500, min_samples_split= 4, min_samples_leaf= 7, max_samples= 0.9, max_features= 0.7, max_depth= 20, criterion= "gini", class_weight= "balanced"),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=7, learning_rate=0.5),
            "XGBoost": XGBClassifier(subsample= 1.0, n_estimators= 400, max_depth= 8, learning_rate= 0.25, colsample_bytree= 0.7, alpha= 1.0),
            "AdaBoosting": AdaBoostClassifier(random_state=42, n_estimators=400, learning_rate=0.5,
                                              estimator=DecisionTreeClassifier(max_depth=5),
                                              algorithm="SAMME"),
            "CatBoost": CatBoostClassifier(n_estimators=900, max_depth=9, learning_rate=0.3),
            "LightGBM": LGBMClassifier(num_leaves=512, n_estimators=200, max_depth=9, learning_rate=0.5)
        },
        "lf_PPG+lf_hf_PPG": {
            "Random Forest": RandomForestClassifier(warm_start= False, random_state= 42, n_estimators= 600, min_samples_split= 3, min_samples_leaf= 6, max_samples= 0.9, max_features= "log2", max_depth= None, criterion= "entropy", class_weight= "balanced_subsample"),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=900, max_depth=6, learning_rate=0.1),
            "XGBoost": XGBClassifier(alpha=0.9, colsample_bytree=1.0, learning_rate=0.5, max_depth=4,
                                     n_estimators=700, subsample=0.9),
            "AdaBoosting": AdaBoostClassifier(random_state=42, n_estimators=1200, learning_rate=0.5,
                                              estimator=DecisionTreeClassifier(max_depth=5),
                                              algorithm='SAMME'),
            "CatBoost": CatBoostClassifier(n_estimators=200, max_depth=8, learning_rate=0.8),
            "LightGBM": LGBMClassifier(num_leaves=4, n_estimators=900, max_depth=9, learning_rate=0.4)
        },
        "hf_PPG+lf_hf_PPG": {
            "Random Forest": RandomForestClassifier(class_weight= "balanced_subsample", criterion= "entropy", max_depth= 15, max_features= "sqrt", max_samples= None, min_samples_leaf= 3, min_samples_split= 4, n_estimators= 300, random_state= 42, warm_start= False),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=900, max_depth=6, learning_rate=0.3),
            "XGBoost": XGBClassifier(subsample= 0.7, n_estimators= 300, max_depth= 10, learning_rate= 0.2, colsample_bytree= 0.6, alpha= 0.1),
            "AdaBoosting": AdaBoostClassifier(random_state= 42, n_estimators= 500, learning_rate= 1.0, estimator= DecisionTreeClassifier(max_depth=4, min_samples_leaf=4), algorithm= 'SAMME'),
            "CatBoost": CatBoostClassifier(n_estimators=400, max_depth=7, learning_rate=0.3),
            "LightGBM": LGBMClassifier(num_leaves=128, n_estimators=700, max_depth=4, learning_rate=0.1)
        },
    # "SMOTE": {
    #     "EDA_Phasic_EmotiBit+EDA_Tonic_EmotiBit": {
    #         "Random Forest": RandomForestClassifier(warm_start=True, random_state=42, n_estimators=200,
    #                                                 min_samples_split=4, min_samples_leaf=7, max_samples=None,
    #                                                 max_features=None, max_depth=None, criterion="entropy",
    #                                                 class_weight="balanced"),
    #         "Gradient Boosting": GradientBoostingClassifier(n_estimators=700, max_depth=9, learning_rate=0.4),
    #         "XGBoost": XGBClassifier(subsample=0.8, n_estimators=500, max_depth=12, learning_rate=0.05,
    #                                  colsample_bytree=0.8, alpha=1.5),
    #         "AdaBoosting": AdaBoostClassifier(random_state=42, n_estimators=400, learning_rate=0.3,
    #                                           estimator=DecisionTreeClassifier(max_depth=5), algorithm="SAMME"),
    #         "CatBoost": CatBoostClassifier(n_estimators=900, max_depth=9, learning_rate=0.1),
    #         "LightGBM": LGBMClassifier(num_leaves=8, n_estimators=600, max_depth=8, learning_rate=0.3)
    #     },
    #     "EDA_Phasic_EmotiBit+lf_PPG": {
    #         "Random Forest": RandomForestClassifier(class_weight="balanced_subsample", criterion="gini", max_depth=20,
    #                                                 max_features=0.5, max_samples=0.9, min_samples_leaf=4,
    #                                                 min_samples_split=3, n_estimators=500, random_state=42,
    #                                                 warm_start=True),
    #         "Gradient Boosting": GradientBoostingClassifier(n_estimators=800, max_depth=7, learning_rate=0.6),
    #         "XGBoost": XGBClassifier(alpha=0.1, colsample_bytree=1.0, learning_rate=0.5, max_depth=4, n_estimators=400,
    #                                  subsample=1.0),
    #         "AdaBoosting": AdaBoostClassifier(random_state=42, n_estimators=500, learning_rate=0.8,
    #                                           estimator=DecisionTreeClassifier(max_depth=4, min_samples_leaf=4),
    #                                           algorithm="SAMME"),
    #         "CatBoost": CatBoostClassifier(n_estimators=800, max_depth=7, learning_rate=0.8),
    #         "LightGBM": LGBMClassifier(num_leaves=64, n_estimators=900, max_depth=7, learning_rate=0.6)
    #     },
    #     "EDA_Phasic_EmotiBit+hf_PPG": {
    #         "Random Forest": RandomForestClassifier(class_weight=None, criterion="entropy", max_depth=None,
    #                                                 max_features=0.9, max_samples=None, min_samples_leaf=2,
    #                                                 min_samples_split=15, n_estimators=400, random_state=42,
    #                                                 warm_start=False),
    #         "Gradient Boosting": GradientBoostingClassifier(n_estimators=600, max_depth=8, learning_rate=0.2),
    #         "XGBoost": XGBClassifier(alpha=0.5, colsample_bytree=0.8, learning_rate=0.35, max_depth=11,
    #                                  n_estimators=800, subsample=1.0),
    #         "AdaBoosting": AdaBoostClassifier(random_state=42, n_estimators=600, learning_rate=0.7,
    #                                           estimator=DecisionTreeClassifier(max_depth=5), algorithm="SAMME"),
    #         "CatBoost": CatBoostClassifier(n_estimators=300, max_depth=8, learning_rate=0.6),
    #         "LightGBM": LGBMClassifier(num_leaves=32, n_estimators=500, max_depth=7, learning_rate=0.1)
    #     },
    #     "EDA_Phasic_EmotiBit+lf_hf_PPG": {
    #         "Random Forest": RandomForestClassifier(class_weight="balanced_subsample", criterion="entropy",
    #                                                 max_depth=15, max_features="sqrt", max_samples=0.7,
    #                                                 min_samples_leaf=1, min_samples_split=2, n_estimators=600,
    #                                                 random_state=42, warm_start=True),
    #         "Gradient Boosting": GradientBoostingClassifier(n_estimators=600, max_depth=5, learning_rate=0.3),
    #         "XGBoost": XGBClassifier(subsample=1.0, n_estimators=300, max_depth=9, learning_rate=0.35,
    #                                  colsample_bytree=0.9, alpha=0.1),
    #         "AdaBoosting": AdaBoostClassifier(random_state=42, n_estimators=1200, learning_rate=0.7,
    #                                           estimator=DecisionTreeClassifier(max_depth=4, min_samples_leaf=4),
    #                                           algorithm="SAMME"),
    #         "CatBoost": CatBoostClassifier(n_estimators=700, max_depth=6, learning_rate=0.5),
    #         "LightGBM": LGBMClassifier(num_leaves=64, n_estimators=900, max_depth=5, learning_rate=0.2)
    #     },
    #     "EDA_Tonic_EmotiBit+lf_PPG": {
    #         "Random Forest": RandomForestClassifier(class_weight="balanced_subsample", criterion="gini", max_depth=15,
    #                                                 max_features=0.7, max_samples=0.5, min_samples_leaf=5,
    #                                                 min_samples_split=10, n_estimators=900, random_state=42,
    #                                                 warm_start=True),
    #         "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=7, learning_rate=0.5),
    #         "XGBoost": XGBClassifier(alpha=1.5, colsample_bytree=0.8, learning_rate=0.35, max_depth=9, n_estimators=600,
    #                                  subsample=0.8),
    #         "AdaBoosting": AdaBoostClassifier(algorithm="SAMME",
    #                                           estimator=DecisionTreeClassifier(max_depth=5, min_samples_leaf=4),
    #                                           learning_rate=0.7, n_estimators=1200, random_state=42),
    #         "CatBoost": CatBoostClassifier(n_estimators=800, max_depth=9, learning_rate=0.5),
    #         "LightGBM": LGBMClassifier(num_leaves=128, n_estimators=700, max_depth=9, learning_rate=0.1)
    #     },
    #     "EDA_Tonic_EmotiBit+hf_PPG": {
    #         "Random Forest": RandomForestClassifier(warm_start=False, random_state=42, n_estimators=1000,
    #                                                 min_samples_split=8, min_samples_leaf=1, max_samples=0.5,
    #                                                 max_features=0.5, max_depth=None, criterion="gini",
    #                                                 class_weight="balanced"),
    #         "Gradient Boosting": GradientBoostingClassifier(n_estimators=800, max_depth=7, learning_rate=0.3),
    #         "XGBoost": XGBClassifier(alpha=0.1, colsample_bytree=1.0, learning_rate=0.4, max_depth=8, n_estimators=400,
    #                                  subsample=0.8),
    #         "AdaBoosting": AdaBoostClassifier(random_state=42, n_estimators=500, learning_rate=0.7,
    #                                           estimator=DecisionTreeClassifier(max_depth=5, min_samples_leaf=2),
    #                                           algorithm="SAMME"),
    #         "CatBoost": CatBoostClassifier(n_estimators=600, max_depth=7, learning_rate=0.6),
    #         "LightGBM": LGBMClassifier(num_leaves=32, n_estimators=400, max_depth=6, learning_rate=0.13)
    #     },
    #     "EDA_Tonic_EmotiBit+lf_hf_PPG": {
    #         "Random Forest": RandomForestClassifier(warm_start=False, random_state=42, n_estimators=900,
    #                                                 min_samples_split=3, min_samples_leaf=2, max_samples=0.5,
    #                                                 max_features=0.9, max_depth=20, criterion="gini",
    #                                                 class_weight="balanced_subsample"),
    #         "Gradient Boosting": GradientBoostingClassifier(n_estimators=600, max_depth=8, learning_rate=0.2),
    #         "XGBoost": XGBClassifier(alpha=1.5, colsample_bytree=0.8, learning_rate=0.4, max_depth=11, n_estimators=200,
    #                                  subsample=0.8),
    #         "AdaBoosting": AdaBoostClassifier(algorithm="SAMME", estimator=DecisionTreeClassifier(max_depth=5),
    #                                           learning_rate=0.2, n_estimators=900, random_state=42),
    #         "CatBoost": CatBoostClassifier(n_estimators=500, max_depth=5, learning_rate=0.3),
    #         "LightGBM": LGBMClassifier(num_leaves=32, n_estimators=500, max_depth=9, learning_rate=0.7)
    #     },
    #     "lf_PPG+hf_PPG": {
    #         "Random Forest": RandomForestClassifier(warm_start=False, random_state=42, n_estimators=1200,
    #                                                 min_samples_split=8, min_samples_leaf=4, max_samples=None,
    #                                                 max_features=0.7, max_depth=20, criterion="gini",
    #                                                 class_weight=None),
    #         "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, max_depth=8, learning_rate=0.3),
    #         "XGBoost": XGBClassifier(alpha=1.5, colsample_bytree=1.0, learning_rate=0.3, max_depth=9, n_estimators=200,
    #                                  subsample=0.8),
    #         "AdaBoosting": AdaBoostClassifier(random_state=42, n_estimators=200, learning_rate=0.7,
    #                                           estimator=DecisionTreeClassifier(max_depth=5, min_samples_leaf=2),
    #                                           algorithm="SAMME"),
    #         "CatBoost": CatBoostClassifier(n_estimators=700, max_depth=7, learning_rate=0.3),
    #         "LightGBM": LGBMClassifier(num_leaves=512, n_estimators=400, max_depth=8, learning_rate=0.6)
    #     },
    #     "lf_PPG+lf_hf_PPG": {
    #         "Random Forest": RandomForestClassifier(class_weight="balanced", criterion="gini", max_depth=None,
    #                                                 max_features=None, max_samples=0.9, min_samples_leaf=1,
    #                                                 min_samples_split=15, n_estimators=100, random_state=42,
    #                                                 warm_start=True),
    #         "Gradient Boosting": GradientBoostingClassifier(n_estimators=400, max_depth=6, learning_rate=0.1),
    #         "XGBoost": XGBClassifier(alpha=0.5, colsample_bytree=0.9, learning_rate=0.05, max_depth=11,
    #                                  n_estimators=900, subsample=0.8),
    #         "AdaBoosting": AdaBoostClassifier(random_state=42, n_estimators=400, learning_rate=0.2,
    #                                           estimator=DecisionTreeClassifier(max_depth=5, min_samples_leaf=2),
    #                                           algorithm='SAMME'),
    #         "CatBoost": CatBoostClassifier(n_estimators=400, max_depth=5, learning_rate=0.8),
    #         "LightGBM": LGBMClassifier(num_leaves=16, n_estimators=800, max_depth=9, learning_rate=0.5)
    #     },
    #     "hf_PPG+lf_hf_PPG": {
    #         "Random Forest": RandomForestClassifier(class_weight="balanced_subsample", criterion="gini", max_depth=20,
    #                                                 max_features=0.7, max_samples=None, min_samples_leaf=1,
    #                                                 min_samples_split=3, n_estimators=700, random_state=42,
    #                                                 warm_start=True),
    #         "Gradient Boosting": GradientBoostingClassifier(n_estimators=800, max_depth=5, learning_rate=0.5),
    #         "XGBoost": XGBClassifier(alpha=0.1, colsample_bytree=0.8, learning_rate=0.05, max_depth=9, n_estimators=700,
    #                                  subsample=0.6),
    #         "AdaBoosting": AdaBoostClassifier(algorithm='SAMME',
    #                                           estimator=DecisionTreeClassifier(max_depth=5, min_samples_leaf=2),
    #                                           learning_rate=0.7, n_estimators=800, random_state=42),
    #         "CatBoost": CatBoostClassifier(n_estimators=400, max_depth=7, learning_rate=0.3),
    #         "LightGBM": LGBMClassifier(num_leaves=16, n_estimators=900, max_depth=6, learning_rate=0.9)
    #     },
    }
}

normalizeds = [" - OverSampling - Normalization", ]
methods = ["ADASYN"]
for normalized in normalizeds:
    for method in methods:
        result = {'Method Name': [], 'feature': [], "Name": [], 'Average CV Score': [], 'TotalTime': [],
                  'n_split': [], }
        for feature1, feature2 in combinations(
                ["EDA_Phasic_EmotiBit", "EDA_Tonic_EmotiBit", "lf_PPG", "hf_PPG", "lf_hf_PPG"], 2):
            df = pd.read_csv(
                fr"Y:\ALLfile\Real-Data (Processed)\Result - ResampleMax{normalized}\OverSampling{method} - Normalization.csv")
            features_select = ["SkinTemp_Emo", f"BMI", f"{feature1}", f"{feature2}"]
            X = df[features_select]
            y = df["PainLevel"]
            for n_split in nsplits:
                CurrentFeature = f"{feature1}+{feature2}"
                print(f"Current Feature : {CurrentFeature}")
                for Moname, model in params[method][CurrentFeature].items():
                    print('--------------- Start Crossing ---------------')
                    k_folds = KFold(n_splits=n_split)
                    start = time.time()
                    scores = cross_val_score(model, X, y, cv=k_folds)
                    TotalTime = time.time() - start
                    print(f"method : {method}")
                    print(f'\nmodel name: {Moname}')
                    print(f'Feature: {feature1} + {feature2} + BMI + SkinTemp_Emo')
                    print("Cross Validation Scores: ", scores)
                    print("Average CV Score: ", scores.mean())
                    print(f"Total Time: {TotalTime}")
                    print("Number of CV Scores used in Average: ", len(scores))
                    result['Method Name'].append(method)
                    result['feature'].append(f"{feature1} + {feature2} + BMI + SkinTemp_Emo")
                    result['Name'].append(Moname)
                    result['Average CV Score'].append(scores.mean())
                    result["TotalTime"].append(TotalTime)
                    result["n_split"].append(n_split)
            print(result)
        result = pd.DataFrame(result)
        print(result)
        result.to_csv(
            rf"Y:\ALLfile\Real-Data (Processed)\CrossValidation - Oversampling - Normalization (4Features)\OverSampling_{normalized}_{method}_4Features.csv")
