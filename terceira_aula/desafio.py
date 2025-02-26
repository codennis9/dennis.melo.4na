# Importando as bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, f1_score, roc_curve, auc)

# Função para carregar e dividir os dados
def load_and_split_data(test_size=0.3, random_state=42):
    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

# Função para normalizar as features
def normalize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

# Função para treinar o modelo KNN
def train_knn_classifier(X_train, y_train, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

# Função para avaliar o modelo
def evaluate_model(knn, X_test, y_test):
    y_pred = knn.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1_score": f1_score(y_test, y_pred, average='weighted')
    }
    return metrics

# Função para plotar a Curva ROC
def plot_roc_curve(knn, X_test, y_test):
    n_classes = len(np.unique(y_test))
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, knn.predict_proba(X_test)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'Classe {i} (AUC = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC (Receiver Operating Characteristic)')
    plt.legend(loc='lower right')
    plt.show()

# Função principal para executar o fluxo completo
def main():
    X_train, X_test, y_train, y_test = load_and_split_data()
    X_train, X_test = normalize_data(X_train, X_test)
    knn = train_knn_classifier(X_train, y_train)
    metrics = evaluate_model(knn, X_test, y_test)

    print("Acurácia:", metrics["accuracy"])
    print("Matriz de Confusão:\n", metrics["confusion_matrix"])
    print("Precisão:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("F1-Score:", metrics["f1_score"])

    plot_roc_curve(knn, X_test, y_test)

if __name__ == "__main__":
    main()
