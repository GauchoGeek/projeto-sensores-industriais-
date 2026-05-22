import pickle
import numpy as np
import pandas as pd
import os

os.makedirs('output', exist_ok=True)


class IndustrialSensorClassifier:
    """
    Sistema de Classificação de Estado de Equipamentos Industriais
    baseado em dados de sensores.
    """

    def __init__(self, model, scaler, selector):
        self.model = model
        self.scaler = scaler
        self.selector = selector
        self.estado_map = {
            0: '✅ Normal',
            1: '⚠️ Desgaste Inicial',
            2: '🔶 Sobrecarga',
            3: '🔴 Falha Iminente'
        }

    def preprocess(self, sensor_data: pd.DataFrame) -> pd.DataFrame:
        """Aplica engenharia de features e pré-processamento."""
        df = sensor_data.copy()
        df['razao_temp_bearing'] = df['temperatura_bearing'] / (df['temperatura_motor'] + 1e-6)
        df['potencia_aparente'] = df['corrente_eletrica'] * df['rpm'] / 1000
        df['indice_estresse'] = df['vibracao_eixo'] * df['temperatura_motor'] / 100
        return df

    def predict(self, sensor_data: pd.DataFrame) -> dict:
        """
        Classifica o estado do equipamento.

        Parâmetros:
            sensor_data: DataFrame com colunas dos sensores

        Retorna:
            dict com estado predito, probabilidades e recomendação
        """
        processed = self.preprocess(sensor_data)
        scaled = self.scaler.transform(processed)
        selected = self.selector.transform(scaled)

        prediction = self.model.predict(selected)[0]
        probabilities = self.model.predict_proba(selected)[0]
        confidence = probabilities.max()

        recommendations = {
            0: "Equipamento operando normalmente. Manutenção de rotina recomendada.",
            1: "Sinais iniciais de desgaste. Agendar inspeção nos próximos 7 dias.",
            2: "Condição de sobrecarga detectada! Inspeção imediata recomendada.",
            3: "⚠️ ALERTA CRÍTICO: Risco iminente de falha! Parada de emergência recomendada!"
        }

        return {
            'estado': self.estado_map[prediction],
            'classe': int(prediction),
            'confianca': float(confidence),
            'probabilidades': {
                self.estado_map[i]: float(p) for i, p in enumerate(probabilities)
            },
            'recomendacao': recommendations[prediction]
        }

    def salvar_modelo(self, caminho='output/modelo_classificador.pkl'):
        """Salva o classificador em disco."""
        with open(caminho, 'wb') as f:
            pickle.dump(self, f)
        print(f"✅ Modelo salvo em '{caminho}'")

    @staticmethod
    def carregar_modelo(caminho='output/modelo_classificador.pkl'):
        """Carrega o classificador de disco."""
        with open(caminho, 'rb') as f:
            return pickle.load(f)


def executar_classificacao(salvar=True):
    """Carrega os melhores resultados e faz uma classificação de exemplo."""

    with open('output/treinamento_resultados.pkl', 'rb') as f:
        results = pickle.load(f)
    with open('output/preprocessamento.pkl', 'rb') as f:
        preproc = pickle.load(f)
    with open('output/selecao_features.pkl', 'rb') as f:
        sel = pickle.load(f)

    # Melhor modelo
    best_name = max(results, key=lambda k: results[k]['f1'])
    best_model = results[best_name]['model']
    print(f"\n🏆 Melhor Modelo: {best_name}")

    # Criar classificador
    classifier = IndustrialSensorClassifier(best_model, preproc['scaler'], sel['selector'])

    if salvar:
        classifier.salvar_modelo()

    # Exemplo de uso
    novo_sensor = pd.DataFrame({
        'temperatura_motor': [95],
        'vibracao_eixo': [5.5],
        'pressao_oleo': [2.0],
        'rpm': [1550],
        'corrente_eletrica': [62],
        'temperatura_bearing': [88],
        'horas_operacao': [3200],
        'nivel_vibracao_hf': [1.8]
    })

    resultado = classifier.predict(novo_sensor)

    print("=" * 60)
    print("  🔧 RESULTADO DA CLASSIFICAÇÃO DO EQUIPAMENTO")
    print("=" * 60)
    print(f"  Estado:         {resultado['estado']}")
    print(f"  Confiança:      {resultado['confianca']:.2%}")
    print(f"  Recomendação:   {resultado['recomendacao']}")
    print(f"\n  Probabilidades:")
    for estado, prob in resultado['probabilidades'].items():
        barra = '█' * int(prob * 30)
        print(f"    {estado:25s} {prob:.2%} |{barra}|")
    print("=" * 60)

    return classifier, resultado


if __name__ == '__main__':
    executar_classificacao()
