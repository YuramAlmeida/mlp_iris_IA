from train import train
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate():
    model, X_test, y_test = train()

    y_pred = model.predict(X_test)

    print("\nAcurácia:")
    print(accuracy_score(y_test, y_pred))

    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    evaluate()
