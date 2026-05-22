import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os

os.makedirs('output', exist_ok=True)


def preprocessar_dados(data, salvar=True):
    """
    Realiza o pré-processamento completo dos dados:
    - Separação features/label
    - Divisão treino/teste
    - Normalização (StandardScaler)
    - Balanceamento com SMOTE

    Parâmetros:
        data: pd.DataFrame com os dados brutos
        salvar: bool, se True salva os artefatos em disco

    Retorna:
        dict com todos os conjuntos de dados e objetos de transformação
    """

    X = data.drop(['estado', 'estado_label'], axis=1)
    y = data['estado']

    print(f"Features: {list(X.columns)}")
    print(f"Shape de X: {X.shape}")
    print(f"Shape de y: {y.shape}")

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTreino: {X_train.shape[0]} amostras")
    print(f"Teste:  {X_test.shape[0]} amostras")

    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

    print("\nMédias (treino) após escalonamento:")
    print(X_train_scaled.mean().round(4))
    print("\nDesvios (treino) após escalonamento:")
    print(X_train_scaled.std().round(4))

    # Balanceamento com SMOTE
    print(f"\nDistribuição ANTES do SMOTE:")
    print(pd.Series(y_train).value_counts().sort_index())

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    print(f"\nDistribuição DEPOIS do SMOTE:")
    print(pd.Series(y_train_balanced).value_counts().sort_index())

    resultado = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_balanced': X_train_balanced,
        'y_train_balanced': y_train_balanced,
        'scaler': scaler
    }

    if salvar:
        with open('output/preprocessamento.pkl', 'wb') as f:
            pickle.dump(resultado, f)
        print("\n✅ Pré-processamento salvo em 'output/preprocessamento.pkl'")

    return resultado


if __name__ == '__main__':
    data = pd.read_csv('output/dados_sensores.csv')
    preprocessar_dados(data)
