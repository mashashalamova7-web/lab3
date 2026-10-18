import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from grader_contracts.linear_models import RegressionInput, ClassificationInput
from linear_models import (
    train_linear_regression,
    train_logistic_classifier,
    compare_classical_models,
)


def main():
    # 1. Линейная регрессия 
    print("1. ЛИНЕЙНАЯ РЕГРЕССИЯ")

    df_reg = pd.read_csv('Student_Performance.csv')
    print(f"Исходная форма: {df_reg.shape}")
    print(f"Столбцы: {df_reg.columns.tolist()}")

    # Приведение категориальных признаков к числам
    for col in df_reg.select_dtypes(include='object').columns:
        df_reg[col] = LabelEncoder().fit_transform(df_reg[col])
        print(f"  {col}: уникальные значения -> числа")

    X_reg = df_reg.drop(columns=['Performance Index']).values
    y_reg = df_reg['Performance Index'].values

    scaler_reg = StandardScaler()
    X_reg = scaler_reg.fit_transform(X_reg)

    X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    print(f"Train: {X_tr.shape}, Test: {X_te.shape}")

    reg_data = RegressionInput(X_tr, y_tr, X_te, y_te, learning_rate=0.01, epochs=1000)
    reg_result = train_linear_regression(reg_data)

    test_mse = float(np.mean((reg_result.predictions - y_te) ** 2))
    print(f"MSE на обучении (последняя эпоха): {reg_result.loss_history[-1]:.4f}")
    print(f"MSE на тесте: {test_mse:.4f}")

    #2. Логистическая регрессия 

    print("2. ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ")

    # ВАЖНО: разделитель — точка с запятой!
    df_cls = pd.read_csv('german.csv', sep=';')
    print(f"Форма: {df_cls.shape}")
    print(f"Столбцы: {df_cls.columns.tolist()}")

    y_cls = df_cls.iloc[:, 0].values          # Creditability
    X_cls = df_cls.iloc[:, 1:].values         # все остальные признаки

    # Приведение к числам (на случай если есть строки)
    X_cls = pd.DataFrame(X_cls).apply(pd.to_numeric, errors='coerce').values
    y_cls = pd.to_numeric(y_cls, errors='coerce').astype(int)

    scaler_cls = StandardScaler()
    X_cls = scaler_cls.fit_transform(X_cls)

    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=42
    )
    print(f"Train: {X_tr_c.shape}, Test: {X_te_c.shape}")

    cls_data = ClassificationInput(
        X_tr_c, y_tr_c, X_te_c, y_te_c,
        learning_rate=0.1, epochs=1000
    )
    cls_result = train_logistic_classifier(cls_data)

    own_metrics = {
        "roc_auc": float(roc_auc_score(y_te_c, cls_result.probabilities)),
        "accuracy": float(accuracy_score(y_te_c, cls_result.predictions)),
        "precision": float(precision_score(y_te_c, cls_result.predictions, zero_division=0)),
        "recall": float(recall_score(y_te_c, cls_result.predictions, zero_division=0)),
        "f1": float(f1_score(y_te_c, cls_result.predictions, zero_division=0)),
    }
    print(f"\nСвоя логистическая регрессия:")
    for k, v in own_metrics.items():
        print(f"  {k}: {v:.4f}")

    #  3. Сравнение моделей 
    print("3. СРАВНЕНИЕ МОДЕЛЕЙ")

    comparison = compare_classical_models(cls_data)

    print(f"\nLogisticRegression (sklearn):")
    print(f"  ROC-AUC: {comparison.logistic_regression.roc_auc:.4f}")
    print(f"  Accuracy: {comparison.logistic_regression.accuracy:.4f}")
    print(f"  F1: {comparison.logistic_regression.f1:.4f}")

    print(f"\nDecisionTree:")
    print(f"  ROC-AUC: {comparison.decision_tree.roc_auc:.4f}")
    print(f"  Accuracy: {comparison.decision_tree.accuracy:.4f}")
    print(f"  F1: {comparison.decision_tree.f1:.4f}")

    print(f"\nKNN:")
    print(f"  ROC-AUC: {comparison.knn.roc_auc:.4f}")
    print(f"  Accuracy: {comparison.knn.accuracy:.4f}")
    print(f"  F1: {comparison.knn.f1:.4f}")

    # Сохранение результатов 
    results = {
        "linear_regression": {"test_mse": test_mse},
        "classification": {
            "own_logistic": own_metrics,
            "logistic_regression": {
                "roc_auc": comparison.logistic_regression.roc_auc,
                "accuracy": comparison.logistic_regression.accuracy,
                "precision": comparison.logistic_regression.precision,
                "recall": comparison.logistic_regression.recall,
                "f1": comparison.logistic_regression.f1,
            },
            "decision_tree": {
                "roc_auc": comparison.decision_tree.roc_auc,
                "accuracy": comparison.decision_tree.accuracy,
                "precision": comparison.decision_tree.precision,
                "recall": comparison.decision_tree.recall,
                "f1": comparison.decision_tree.f1,
            },
            "knn": {
                "roc_auc": comparison.knn.roc_auc,
                "accuracy": comparison.knn.accuracy,
                "precision": comparison.knn.precision,
                "recall": comparison.knn.recall,
                "f1": comparison.knn.f1,
            },
        },
    }

    with open('experiment_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Результаты сохранены в experiment_results.json")


if __name__ == '__main__':
    main()