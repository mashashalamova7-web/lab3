import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score



from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    X_train = np.array(data.train_features, float)
    Y_train = np.array(data.train_target, float)
    X_test = np.array(data.test_features, float)

    n_samples, n_features = X_train.shape
    weights = np.zeros(n_features)
    bias = 0.0
    learning_rate = data.learning_rate
    epochs = data.epochs
    loss_history = []

    for epoch in range(epochs):
        y_pred = X_train @ weights + bias
        errors = y_pred - Y_train
        mse = np.mean(errors ** 2)
        loss_history.append(float(mse))

        grad_w = (2 / n_samples) * (X_train.T @ errors)
        grad_b = (2 / n_samples) * np.sum(errors)
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    predictions = X_test @ weights + bias
    return RegressionResult(weights, bias, loss_history, predictions)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def binary_cross_entropy(y_true, y_prob):
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    return -float(np.mean(y_true * np.log(y_prob_clipped) + (1 - y_true) * np.log(1 - y_prob_clipped)))


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    X_train = np.array(data.train_features, float)
    Y_train = np.array(data.train_target, float)
    X_test = np.array(data.test_features, float)

    n_samples, n_features = X_train.shape
    weights = np.zeros(n_features)
    bias = 0.0
    learning_rate = data.learning_rate
    epochs = data.epochs
    loss_history = []

    for epoch in range(epochs):
        z = X_train @ weights + bias
        y_prob = sigmoid(z)

        bce = binary_cross_entropy(Y_train, y_prob)
        loss_history.append(float(bce))

        errors = y_prob - Y_train
        grad_w = (1 / n_samples)  * (X_train.T @ errors)
        grad_b = (1 / n_samples) * np.sum(errors)
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    z_test = X_test @ weights + bias
    probabilities = sigmoid(z_test)
    predictions = (probabilities >= 0.5).astype(int)

    return ClassificationResult(weights, bias, loss_history, probabilities, predictions)


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    X_train = np.array(data.train_features, float)
    y_train = np.array(data.train_target, int)
    X_test = np.array(data.test_features, float)
    y_test = np.array(data.test_target, int)

    def eval_model(model):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probas = model.predict_proba(X_test)[:, 1]
        return ClassificationMetrics(
            roc_auc=float(roc_auc_score(y_test, probas)),
            accuracy=float(accuracy_score(y_test, preds)),
            precision=float(precision_score(y_test, preds, zero_division=0)),
            recall=float(recall_score(y_test, preds, zero_division=0)),
            f1=float(f1_score(y_test, preds, zero_division=0)),
        )
    lr_metrics = eval_model(LogisticRegression(max_iter = 1000, random_state = 42))
    dt_metrics = eval_model(DecisionTreeClassifier(random_state = 42))
    knn_metrics = eval_model(KNeighborsClassifier())

    return BaselineComparisonResult(lr_metrics, dt_metrics, knn_metrics)