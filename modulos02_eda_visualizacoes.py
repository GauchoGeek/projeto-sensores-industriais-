import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('output', exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12


def eda_completa(data):
    """
    Realiza Análise Exploratória de Dados completa com visualizações.

    Parâmetros:
        data: pd.DataFrame com os dados dos sensores e coluna 'estado_label'

    Retorna:
        None (salva gráficos em 'output/')
    """

    sensor_features = [
        'temperatura_motor', 'vibracao_eixo', 'pressao_oleo',
        'rpm', 'corrente_eletrica', 'temperatura_bearing',
        'nivel_vibracao_hf'
    ]

    cores = ['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad']
    estado_map = {0: 'Normal', 1: 'Desgaste_Inicial', 2: 'Sobrecarga', 3: 'Falha_Iminente'}

    # ========================================
    # 2.1 Distribuição das Classes
    # ========================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    class_counts = data['estado_label'].value_counts()
    axes[0].bar(class_counts.index, class_counts.values, color=cores,
                edgecolor='black', linewidth=1.2)
    axes[0].set_title('Distribuição dos Estados Operacionais', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Estado', fontsize=12)
    axes[0].set_ylabel('Quantidade', fontsize=12)
    axes[0].tick_params(axis='x', rotation=30)

    axes[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
                colors=cores, startangle=90, explode=(0.05, 0.05, 0.05, 0.05),
                shadow=True)
    axes[1].set_title('Proporção dos Estados', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/01_distribuicao_classes.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Gráfico salvo: output/01_distribuicao_classes.png")

    # ========================================
    # 2.2 Distribuição dos Sensores por Estado
    # ========================================
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.flatten()

    for idx, feature in enumerate(sensor_features):
        for estado, cor in zip([0, 1, 2, 3], cores):
            subset = data[data['estado'] == estado][feature]
            axes[idx].hist(subset, bins=40, alpha=0.5, color=cor,
                           label=estado_map[estado], density=True)
        axes[idx].set_title(feature.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        axes[idx].legend(fontsize=8)
        axes[idx].set_ylabel('Densidade')

    fig.delaxes(axes[8])
    fig.delaxes(axes[7])

    plt.suptitle('Distribuição dos Sensores por Estado Operacional',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('output/02_distribuicao_sensores.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Gráfico salvo: output/02_distribuicao_sensores.png")

    # ========================================
    # 2.3 Matriz de Correlação
    # ========================================
    plt.figure(figsize=(14, 10))
    corr_cols = sensor_features + ['razao_temp_bearing', 'potencia_aparente', 'indice_estresse']
    corr_matrix = data[corr_cols + ['estado']].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=1,
                cbar_kws={'label': 'Correlação'})
    plt.title('Matriz de Correlação - Sensores e Engenharia de Features',
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/03_correlacao.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Gráfico salvo: output/03_correlacao.png")

    # ========================================
    # 2.4 Pairplot
    # ========================================
    sample_data = data.sample(n=500, random_state=42)
    sns.pairplot(sample_data, vars=['temperatura_motor', 'vibracao_eixo',
                                     'temperatura_bearing', 'nivel_vibracao_hf'],
                 hue='estado_label', palette=cores, diag_kind='kde',
                 plot_kws={'alpha': 0.6, 's': 30})
    plt.suptitle('Pairplot - Sensores Principais', y=1.02, fontsize=14, fontweight='bold')
    plt.savefig('output/04_pairplot.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Gráfico salvo: output/04_pairplot.png")

    # ========================================
    # 2.5 Boxplots Comparativos
    # ========================================
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for idx, feature in enumerate(sensor_features + ['razao_temp_bearing']):
        bp_data = [data[data['estado'] == i][feature].values for i in range(4)]
        bp = axes[idx].boxplot(bp_data, labels=['Normal', 'Desgaste', 'Sobrecarga', 'Falha'],
                               patch_artist=True, showmeans=True)
        for patch, color in zip(bp['boxes'], cores):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[idx].set_title(feature.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        axes[idx].tick_params(axis='x', rotation=15)

    fig.delaxes(axes[7])
    plt.suptitle('Boxplots Comparativos dos Sensores por Estado',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/05_boxplots.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Gráfico salvo: output/05_boxplots.png")


if __name__ == '__main__':
    data = pd.read_csv('output/dados_sensores.csv')
    eda_completa(data)
