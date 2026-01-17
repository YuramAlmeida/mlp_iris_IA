import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_prepare_data():
    # Carregar dataset Zoo
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data"

    columns = [
        "animal_name", "hair", "feathers", "eggs", "milk", "airborne",
        "aquatic", "predator", "toothed", "backbone", "breathes",
        "venomous", "fins", "legs", "tail", "domestic", "catsize",
        "class_type"
    ]

    df = pd.read_csv(url, header=None, names=columns)

    # Remover coluna que não ajuda na classificação
    df = df.drop(columns=["animal_name"])

    X = df.drop("class_type", axis=1)
    y = df["class_type"]

    # Normalização
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
