import numpy as np
import pandas as pd
import os

os.makedirs('output', exist_ok=True)

def gerar_dados(n_samples=5000, seed=42):
    """
    Gera dados sintéticos de sensores industriais simulando
    diferentes estados operacionais de equipamentos.

    Estados:
        0: Normal
        1: Desgaste Inicial
        2: Sobrecarga
        3: Falha Iminente

    Retorna:
        pd.DataFrame com features dos sensores e coluna 'estado'
    """
    np.random.seed(seed)

    proportions = [0.45, 0.25, 0.15, 0.15]
    labels = np.random.choice([0, 1, 2, 3], size=n_samples, p=proportions)

    data = pd.DataFrame()

    # Estado 0: Normal
    mask_0 = labels == 0
    data.loc[mask_0, 'temperatura_motor'] = np.random.normal(75, 5, mask_0.sum())
    data.loc[mask_0, 'vibracao_eixo'] = np.random.normal(2.5, 0.5, mask_0.sum())
    data.loc[mask_0, 'pressao_oleo'] = np.random.normal(3.2, 0.3, mask_0.sum())
    data.loc[mask_0, 'rpm'] = np.random.normal(1500, 50, mask_0.sum())
    data.loc[mask_0, 'corrente_eletrica'] = np.random.normal(45, 5, mask_0.sum())
    data.loc[mask_0, 'temperatura_bearing'] = np.random.normal(60, 4, mask_0.sum())
    data.loc[mask_0, 'horas_operacao'] = np.random.uniform(0, 500, mask_0.sum())
    data.loc[mask_0, 'nivel_vibracao_hf'] = np.random.normal(0.3, 0.1, mask_0.sum())

    # Estado 1: Desgaste Inicial
    mask_1 = labels == 1
    data.loc[mask_1, 'temperatura_motor'] = np.random.normal(85, 8, mask_1.sum())
    data.loc[mask_1, 'vibracao_eixo'] = np.random.normal(4.0, 0.8, mask_1.sum())
    data.loc[mask_1, 'pressao_oleo'] = np.random.normal(2.8, 0.5, mask_1.sum())
    data.loc[mask_1, 'rpm'] = np.random.normal(1480, 60, mask_1.sum())
    data.loc[mask_1, 'corrente_eletrica'] = np.random.normal(52, 7, mask_1.sum())
    data.loc[mask_1, 'temperatura_bearing'] = np.random.normal(72, 6, mask_1.sum())
    data.loc[mask_1, 'horas_operacao'] = np.random.uniform(2000, 5000, mask_1.sum())
    data.loc[mask_1, 'nivel_vibracao_hf'] = np.random.normal(0.7, 0.2, mask_1.sum())

    # Estado 2: Sobrecarga
    mask_2 = labels == 2
    data.loc[mask_2, 'temperatura_motor'] = np.random.normal(100, 10, mask_2.sum())
    data.loc[mask_2, 'vibracao_eixo'] = np.random.normal(6.0, 1.2, mask_2.sum())
    data.loc[mask_2, 'pressao_oleo'] = np.random.normal(2.2, 0.6, mask_2.sum())
    data.loc[mask_2, 'rpm'] = np.random.normal(1600, 100, mask_2.sum())
    data.loc[mask_2, 'corrente_eletrica'] = np.random.normal(68, 10, mask_2.sum())
    data.loc[mask_2, 'temperatura_bearing'] = np.random.normal(90, 10, mask_2.sum())
    data.loc[mask_2, 'horas_operacao'] = np.random.uniform(1000, 4000, mask_2.sum())
    data.loc[mask_2, 'nivel_vibracao_hf'] = np.random.normal(1.5, 0.4, mask_2.sum())

    # Estado 3: Falha Iminente
    mask_3 = labels == 3
    data.loc[mask_3, 'temperatura_motor'] = np.random.normal(120, 15, mask_3.sum())
    data.loc[mask_3, 'vibracao_eixo'] = np.random.normal(9.0, 2.0, mask_3.sum())
    data.loc[mask_3, 'pressao_oleo'] = np.random.normal(1.5, 0.7, mask_3.sum())
    data.loc[mask_3, 'rpm'] = np.random.normal(1400, 150, mask_3.sum())
    data.loc[mask_3, 'corrente_eletrica'] = np.random.normal(85, 15, mask_3.sum())
    data.loc[mask_3, 'temperatura_bearing'] = np.random.normal(115, 15, mask_3.sum())
    data.loc[mask_3, 'horas_operacao'] = np.random.uniform(4000, 10000, mask_3.sum())
    data.loc[mask_3, 'nivel_vibracao_hf'] = np.random.normal(3.0, 0.8, mask_3.sum())

    # Adicionar ruído
    noise_cols = ['temperatura_motor', 'vibracao_eixo', 'pressao_oleo', 'rpm',
                  'corrente_eletrica', 'temperatura_bearing', 'nivel_vibracao_hf']
    for col in noise_cols:
        data[col] += np.random.normal(0, data[col].std() * 0.05, n_samples)

    # Engenharia de features
    data['razao_temp_bearing'] = data['temperatura_bearing'] / (data['temperatura_motor'] + 1e-6)
    data['potencia_aparente'] = data['corrente_eletrica'] * data['rpm'] / 1000
    data['indice_estresse'] = data['vibracao_eixo'] * data['temperatura_motor'] / 100

    data['estado'] = labels
    estado_map = {0: 'Normal', 1: 'Desgaste_Inicial', 2: 'Sobrecarga', 3: 'Falha_Iminente'}
    data['estado_label'] = data['estado'].map(estado_map)

    # Salvar CSV
    data.to_csv('output/dados_sensores.csv', index=False)

    print(f"✅ Dataset gerado e salvo em 'output/dados_sensores.csv'")
    print(f"   Shape: {data.shape}")
    print(f"   Distribuição das classes:")
    print(f"   {data['estado_label'].value_counts().to_dict()}")

    return data


if __name__ == '__main__':
    df = gerar_dados()
