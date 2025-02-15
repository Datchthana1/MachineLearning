import time
import pandas as pd
from catboost import CatBoostClassifier
from itertools import combinations
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from skopt import BayesSearchCV
from xgboost import XGBClassifier

for normalization in [""]:
    paths = {
        # "ADASYN_OverData": rf"D:\DataCenter\ALLfile\Real-Data (Processed)\Concat - ResampleMax - OverSampling{normalization}\OverSamplingADASYN{normalization}.csv",
        # 'SMOTE_OverData': rf"D:\DataCenter\ALLfile\Real-Data (Processed)\Concat - ResampleMax - OverSampling{normalization}\OverSamplingSMOTE{normalization}.csv",
        'SMOTE_OverData': rf"Y:\ALLfile\Real-Data (Processed)\Concat - ResampleMax - OverSampling\OverSamplingSMOTE.csv",
    }

    cv = 5
    n = 10

    for method, path in paths.items():
        df = pd.read_csv(path, encoding="utf8")

        print(df.keys())

        param = {
            # "Phasic": "EDA_Phasic_EmotiBit",
            "Tonic": "EDA_Tonic_EmotiBit",
            # "LF": "lf_PPG",
            # "HF": "hf_PPG",
            "LFHF": "lf_hf_PPG",
        }
        additional_parmas = {
            # "Protein%": "ProteinPercentage",
            # "Muscle%": "MusclePercentage",
            "Fat%": "FatPercentage",
        }

        for (name1, feature1), (name2, feature2) in combinations(param.items(), 2):
            for name11, feature11 in additional_parmas.items():
                selected_features = [
                    # "BMI",
                    # "SkinTemp_Emo",
                    # feature1,
                    # feature2,
                    # feature11,
                    'lf_hf_PPG',
                    'hf_PPG',
                    "lf_PPG"
                    
                ]
                print(selected_features)
                x = df[selected_features]
                y = df["PainLevel"]

                X_train, X_test, y_train, y_test = train_test_split(
                    x, y, test_size=0.3, random_state=42, shuffle=True
                )

                model_results = {
                    "NAME": [],
                    "TIME": [],
                    "BestPara": [],
                    "Unseen_Accuracy": [],
                    "Unseen_Precision": [],
                    "Unseen_Recall": [],
                    "Unseen_f1-Score": [],
                    "seen_Accuracy": [],
                    "seen_Precision": [],
                    "seen_Recall": [],
                    "seen_f1-Score": [],
                }

                def model_performance(
                    name, model, X_test, y_test, selected_features, time
                ):
                    print("Model Characteristics")

                    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
                    para = model.best_params_
                    print(
                        "------------------------------------- Train Test Prediction Prediction ------------------------------------------------"
                    )
                    print("Model Prediction")
                    model_results["NAME"].append(name)
                    model_results["BestPara"].append(para)
                    test_predictions = model.predict(X_test)
                    accuracy_seen = accuracy_score(y_test, test_predictions)
                    CFR_seen = classification_report(y_test, test_predictions)
                    precision_score_seen = precision_score(
                        y_test, test_predictions, average="macro"
                    )
                    recall_score_seen = recall_score(
                        y_test, test_predictions, average="macro"
                    )
                    f1_score_seen = f1_score(y_test, test_predictions, average="macro")
                    print(f"Model's name : \n{name}")
                    print(f"Feature : {selected_features}")
                    print(f"Best Parameter : \n{para}")
                    print(f"Accuracy Score : {accuracy_seen}")
                    print(f"Classification Report : \n{CFR_seen}")
                    model_results["seen_Accuracy"].append(accuracy_seen)
                    model_results["seen_Precision"].append(precision_score_seen)
                    model_results["seen_Recall"].append(recall_score_seen)
                    model_results["seen_f1-Score"].append(f1_score_seen)
                    # cm1 = confusion_matrix(
                    #     y_test, test_predictions, labels=y_test["PainLevel"].unique()
                    # )

                    # print(f"Confusion Metrix : \n{confusion_matrix(y_test, test_predictions)}")

                    print(
                        "------------------------------------- Unseen Dataset Prediction ------------------------------------------------"
                    )
                    df_target = pd.read_csv(
                        rf"Y:\ALLfile\Real-Data (Processed)\Concat - TestData\SR1600_1s.csv"
                    )
                    X = df_target[selected_features]
                    test_predictions = model.predict(X)
                    accuracy_Unseen = accuracy_score(
                        df_target["PainLevel"], test_predictions
                    )
                    CFR_Unseen = classification_report(
                        df_target["PainLevel"], test_predictions
                    )
                    # cm2 = confusion_matrix(
                    #     df_target["PainLevel"],
                    #     test_predictions,
                    #     labels=df_target["PainLevel"].unique(),
                    # )
                    precisionScore_Unseen = precision_score(
                        df_target["PainLevel"], test_predictions, average="macro"
                    )
                    recallScore_Unseen = recall_score(
                        df_target["PainLevel"], test_predictions, average="macro"
                    )
                    f1Score_Unseen = f1_score(
                        df_target["PainLevel"], test_predictions, average="macro"
                    )

                    print("Model Prediction")
                    print(f"Model's name : \n{name}")
                    print(f"Feature : {selected_features}")
                    print(f"Best Parameter : \n{para}")
                    print(f"Accuracy Score : {accuracy_Unseen}")
                    print(f"Classification Report : \n{CFR_Unseen}")
                    print(f"Precision Score : {precisionScore_Unseen}")
                    print(f"Recall Score : {recallScore_Unseen}")
                    print(f"F1 Score : {f1Score_Unseen}")
                    # print(f"Confusion Metrix : \n{CFM}")
                    print(f"Total Time : {time}")
                    model_results["Unseen_Accuracy"].append(accuracy_Unseen)
                    model_results["Unseen_Precision"].append(precisionScore_Unseen)
                    model_results["Unseen_Recall"].append(recallScore_Unseen)
                    model_results["Unseen_f1-Score"].append(f1Score_Unseen)
                    model_results["TIME"].append(time)

                    # disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=y['PainLevel'].unique())
                    # disp1.plot(ax=axes[0])
                    # axes[0].set_title(f'Train_Test_Split Data : {name}')
                    # disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=df_target['PainLevel'].unique())
                    # disp2.plot(ax=axes[1])
                    # axes[1].set_title(f'Unseen Dataset Prediction : {name}')
                    # plt.tight_layout()
                    # plt.show()

                print(
                    "------------------------------------- XGBoosting Classifier Tuning ------------------------------------------------"
                )

                xgb_param = {
                    "n_estimators": (100, 200, 300, 400, 500, 600, 700, 800, 900),
                    "max_depth": (1, 2, 3, 4, 5, 6, 7, 8, 9),
                    'learning_rate': (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                    'subsample': (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
                    'colsample_bytree': (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
                    'alpha': (0.1, 0.5, 1.0, 1.5, 2.0),
                }

                xgb_rand = XGBClassifier()
                xgb_rand_clf = RandomizedSearchCV(
                    xgb_rand,
                    xgb_param,
                    cv=cv,
                    n_iter=n,
                    refit=True,
                )
                start = time.time()
                xgb_rand_clf.fit(X_train, y_train)
                total_time = time.time() - start
                model_performance(
                    "Rand XGB", xgb_rand_clf, X_test, y_test, selected_features, total_time
                )

                xgb_bayes = XGBClassifier()
                xgb_bayes_clf = BayesSearchCV(
                    xgb_bayes,
                    xgb_param,
                    cv=cv,
                    n_iter=n,
                    refit=True,
                )
                start = time.time()
                xgb_bayes_clf.fit(X_train, y_train)
                total_time = time.time() - start
                model_performance(
                    "Bayes XGB",
                    xgb_bayes_clf,
                    X_test,

                    y_test,
                    selected_features,
                    total_time,
                )

                print(
                    "------------------------------------- Random Forest Classifier Tuning ------------------------------------------------"
                )

                random_forest_param = {
                    "n_estimators": (
                        100,
                        200,
                        300,
                        400,
                        500,
                        600,
                        700,
                        800,
                        900,
                    ),
                    "max_depth": (None, 2, 3, 4, 5, 6, 7, 8, 9),
                    "min_samples_split": (2, 3, 4, 5, 6, 7, 8, 9),
                    "min_samples_leaf": (1, 2, 3, 4, 5, 6, 7, 8, 9),
                    "max_features": ("sqrt", "log2", None, 0.5, 0.7, 0.9),
                    "warm_start": (True, False),
                    "class_weight": ("balanced", "balanced_subsample", None),
                    "max_samples": (0.5, 0.7, 0.9, None),
                    "criterion": ("gini", "entropy"),
                    "random_state": [42],
                }

                # rf_grid = RandomForestClassifier()
                # rf_grid_clf = GridSearchCV(rf_grid, random_forest_param, cv=cv, refit=True, )
                # start = time.time()
                # rf_grid_clf.fit(X_test, y_test)
                # total_time = time.time() - start
                # model_performance("Grid RandomForest", rf_grid_clf, X_train, X_test, y_train, y_test, total_time)

                rf_rand = RandomForestClassifier()
                rf_rand_clf = RandomizedSearchCV(
                    rf_rand,
                    random_forest_param,
                    cv=cv,
                    n_iter=n,
                    refit=True,
                )
                start = time.time()
                rf_rand_clf.fit(X_train, y_train)
                total_time = time.time() - start
                model_performance(
                    "Rand RandomForest",
                    rf_rand_clf,
                    X_test,
                    y_test,
                    selected_features,
                    total_time,
                )

                rf_bayes = RandomForestClassifier()
                rf_bayes_clf = BayesSearchCV(
                    rf_bayes,
                    random_forest_param,
                    cv=cv,
                    n_iter=n,
                    refit=True,
                )
                start = time.time()
                rf_bayes_clf.fit(X_train, y_train)
                total_time = time.time() - start
                model_performance(
                    "Bayes RandomForest",
                    rf_bayes_clf,
                    X_test,
                    y_test,
                    selected_features,
                    total_time,
                )

                print(
                    "------------------------------------- GradientBoosting Classifier Tuning ------------------------------------------------"
                )
                gradient_boosting_param = {
                    "n_estimators": (100, 200, 300, 400, 500, 600, 700, 800, 900),
                    "max_depth": (1, 2, 3, 4, 5, 6, 7, 8, 9),
                    "learning_rate": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                }

                # gb_grid = GradientBoostingClassifier()
                # gb_grid_clf = GridSearchCV(gb_grid, gradient_boosting_param, cv=cv, refit=True, )
                # start = time.time()
                # gb_grid_clf.fit(X_test, y_test)
                # total_time = time.time() - start
                # model_performance("Grid GradientBoosting", gb_grid_clf, X_train, X_test, y_train, y_test, total_time)

                gb_rand = GradientBoostingClassifier()
                gb_rand_clf = RandomizedSearchCV(
                    gb_rand,
                    gradient_boosting_param,
                    cv=cv,
                    n_iter=n,
                    refit=True,
                )
                start = time.time()
                gb_rand_clf.fit(X_train, y_train)
                total_time = time.time() - start
                model_performance(
                    "Rand GradientBoosting",
                    gb_rand_clf,
                    X_test,
                    y_test,
                    selected_features,
                    total_time,
                )

                gb_bayes = GradientBoostingClassifier()
                gb_bayes_clf = BayesSearchCV(
                    gb_bayes,
                    gradient_boosting_param,
                    cv=cv,
                    n_iter=n,
                    refit=True,
                )
                start = time.time()
                gb_bayes_clf.fit(X_train, y_train)
                total_time = time.time() - start
                model_performance(
                    "Bayes GradientBoosting",
                    gb_bayes_clf,
                    X_test,
                    y_test,
                    selected_features,
                    total_time,
                )

                print(
                    "------------------------------------- AdaBoost Classifier Tuning ------------------------------------------------"
                )

                adaboost_param = {
                    "n_estimators": (
                        50,
                        100,
                        200,
                        300,
                        400,
                        500,
                        600,
                        700,
                        800,
                        900,
                    ),
                    "learning_rate": (
                        0.01,
                        0.05,
                        0.1,
                        0.2,
                        0.3,
                        0.4,
                        0.5,
                        0.6,
                        0.7,
                        0.8,
                        0.9,
                        1.0,
                    ),
                    "algorithm": ["SAMME"],
                    "random_state": [42],  # เพื่อให้ผลลัพธ์สามารถทำซ้ำได้
                    "estimator": (
                        DecisionTreeClassifier(max_depth=1, min_samples_leaf=1),
                        DecisionTreeClassifier(max_depth=1, min_samples_leaf=2),
                        DecisionTreeClassifier(max_depth=1, min_samples_leaf=4),
                        DecisionTreeClassifier(max_depth=2, min_samples_leaf=1),
                        DecisionTreeClassifier(max_depth=2, min_samples_leaf=2),
                        DecisionTreeClassifier(max_depth=2, min_samples_leaf=4),
                        DecisionTreeClassifier(max_depth=3, min_samples_leaf=1),
                        DecisionTreeClassifier(max_depth=3, min_samples_leaf=2),
                        DecisionTreeClassifier(max_depth=3, min_samples_leaf=4),
                        DecisionTreeClassifier(max_depth=4, min_samples_leaf=1),
                        DecisionTreeClassifier(max_depth=4, min_samples_leaf=2),
                        DecisionTreeClassifier(max_depth=4, min_samples_leaf=4),
                        DecisionTreeClassifier(max_depth=5, min_samples_leaf=1),
                        DecisionTreeClassifier(max_depth=5, min_samples_leaf=2),
                        DecisionTreeClassifier(max_depth=5, min_samples_leaf=4),
                        None,  # สำหรับใช้ weak learner ค่าเริ่มต้น
                    ),
                }

                # ab_grid = AdaBoostClassifier()
                # ab_drid_clf = GridSearchCV(ab_grid, adaboost_param, cv=5, refit=True, ccuracy')
                # start = time.time()
                # ab_drid_clf.fit(X_test, y_test)
                # total_time = time.time() - start
                # model_performance("Grid AdaBoost", ab_drid_clf, X_train, X_test, y_train, y_test, total_time)

                ab_rand = AdaBoostClassifier()
                ab_rand_clf = RandomizedSearchCV(
                    ab_rand,
                    adaboost_param,
                    cv=cv,
                    n_iter=n,
                    refit=True,
                )
                start = time.time()
                ab_rand_clf.fit(X_train, y_train)
                total_time = time.time() - start
                model_performance(
                    "Randomized AdaBoost",
                    ab_rand_clf,
                    X_test,
                    y_test,
                    selected_features,
                    total_time,
                )

                ab_bayes = AdaBoostClassifier()
                ab_bayes_clf = BayesSearchCV(
                    ab_bayes,
                    adaboost_param,
                    cv=cv,
                    n_iter=n,
                    refit=True,
                )
                start = time.time()
                ab_bayes_clf.fit(X_train, y_train)
                total_time = time.time() - start
                model_performance(
                    "Bayes AdaBoost",
                    ab_bayes_clf,
                    X_test,
                    y_test,
                    selected_features,
                    total_time,
                )

                print(
                    "------------------------------------- CatBoost Classifier Tuning ------------------------------------------------"
                )

                catboost_param = {
                    "n_estimators": (100, 200, 300, 400, 500, 600, 700, 800, 900),
                    "learning_rate": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                    "max_depth": (1, 2, 3, 4, 5, 6, 7, 8, 9),
                }

                # cb_grid = CatBoostClassifier()
                # cb_drid_clf = GridSearchCV(cb_grid, catboost_param, cv=cv, refit=True, ccuracy')
                # start = time.time()
                # cb_drid_clf.fit(X_test, y_test)
                # total_time = time.time() - start
                # model_performance("Grid CatBoost", cb_drid_clf, X_train, X_test, y_train, y_test, total_time)

                cb_rand = CatBoostClassifier()
                cb_rand_clf = RandomizedSearchCV(
                    cb_rand,
                    catboost_param,
                    cv=cv,
                    n_iter=n,
                    refit=True,
                )
                start = time.time()
                cb_rand_clf.fit(X_train, y_train)
                total_time = time.time() - start
                model_performance(
                    "Randomized CatBoost",
                    cb_rand_clf,
                    X_test,
                    y_test,
                    selected_features,
                    total_time,
                )

                cb_bayes = CatBoostClassifier()
                cb_bayes_clf = BayesSearchCV(
                    cb_bayes,
                    catboost_param,
                    cv=cv,
                    n_iter=n,
                    refit=True,
                )
                start = time.time()
                cb_bayes_clf.fit(X_train, y_train)
                total_time = time.time() - start
                model_performance(
                    "Bayes CatBoost",
                    cb_bayes_clf,
                    X_test,
                    y_test,
                    selected_features,
                    total_time,
                )

                print(
                    "------------------------------------- LGBM Classifier Tuning ------------------------------------------------"
                )

                light_gmb_param = {
                    "n_estimators": (100, 200, 300, 400, 500, 600, 700, 800, 900),
                    "learning_rate": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                    "max_depth": (1, 2, 3, 4, 5, 6, 7, 8, 9),
                    "num_leaves": (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
                }

                # lgbm_grid = LGBMClassifier()
                # lgbm_drid_clf = GridSearchCV(lgbm_grid, light_gbm_param, cv=cv, refit=True, )
                # start = time.time()
                # lgbm_drid_clf.fit(X_test, y_test)
                # total_time = time.time() - start
                # model_performance("Grid LGBM", lgbm_drid_clf, X_train, X_test, y_train, y_test, total_time)

                lgbm_rand = LGBMClassifier()
                lgbm_rand_clf = RandomizedSearchCV(
                    lgbm_rand,
                    light_gmb_param,
                    cv=cv,
                    n_iter=n,
                    refit=True,
                )
                start = time.time()
                lgbm_rand_clf.fit(X_train, y_train)
                total_time = time.time() - start
                model_performance(
                    "Randomized LGBM",
                    lgbm_rand_clf,
                    X_test,
                    y_test,
                    selected_features,
                    total_time,
                )

                lgbm_bayes = LGBMClassifier()
                lgbm_bayes_clf = BayesSearchCV(
                    lgbm_bayes,
                    light_gmb_param,
                    cv=cv,
                    n_iter=n,
                    refit=True,
                )
                start = time.time()
                lgbm_bayes_clf.fit(X_train, y_train)
                total_time = time.time() - start
                model_performance(
                    "Bayes LGBM",
                    lgbm_bayes_clf,
                    X_test,
                    y_test,
                    selected_features,
                    total_time,
                )

                model_results = pd.DataFrame(model_results)
                print(f"\nModel Report : \n{model_results.to_string()}")
                model_results.to_csv(
                    rf"D:\DataCenter\ALLfile\Real-Data (Processed)\HyperparameterOversampling\SMOTE - Oversampling (3 Features)\BestParameter_Tuning{normalization}_{method}_({selected_features}).csv"
                )
