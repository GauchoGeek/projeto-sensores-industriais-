import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, roc_auc_score
import os

os.makedirs('output', exist_ok=True)


def gerar_dashboard(data, results, y_test, y_pred_best, y_prob_best, best_model, selected_features, salvar=True):
    """
    Gera o dashboard consolidado com todas as visualizações.
    """

    sensor_features = [
        'temperatura_motor', 'vibracao_eixo', 'pressao_oleo',
        'rpm', 'corrente_eletrica', 'temperatura_bearing',
        'nivel_vibracao_hf'
    ]

    cores = ['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad']
    class_names = ['Normal', 'Desgaste Inicial', 'Sobrecarga', 'Falha Iminente']
    n_classes = y_prob_best.shape[1]

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('📊 DASHBOARD - Monitoramento de Equipamentos Industriais',
                 fontsize=20, fontweight='bold', y=0.98)

    # (A) Heatmap de correlação
    ax1 = fig.add_subplot(3, 3, 1)
    sns.heatmap(data[sensor_features].corr(), annot=True, fmt='.2f',
                cmap='RdBu_r', ax=ax1, cbar_kws={'shrink': 0.8})
    ax1.set_title('Correlação dos Sensores', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45, labelsize=8)
    ax1.tick_params(axis='y', rotation=0, labelsize=8)

    # (B) Distribuição das classes
    ax2 = fig.add_subplot(3, 3, 2)
    data['estado_label'].value_counts().plot(kind='pie', ax=ax2, autopct='%1.1f%%',
                                              colors=cores, startangle=90)
    ax2.set_ylabel('')
    ax2.set_title('Distribuição de Estados', fontsize=12, fontweight='bold')

    # (C) Scatter: Temperatura vs Vibração
    ax3 = fig.add_subplot(3, 3, 3)
    for estado, cor in zip([0, 1, 2, 3], cores):
        mask = data['estado'] == estado
        ax3.scatter(data.loc[mask, 'temperatura_motor'],
                    data.loc[mask, 'vibracao_eixo'],
                    c=cor, alpha=0.5, s=20, label=class_names[estado])
    ax3.set_xlabel('Temperatura do Motor')
    ax3.set_ylabel('Vibração do Eixo')
    ax3.set_title('Temperatura vs Vibração', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8)

    # (D) Box plot - Temperatura do Bearing
    ax4 = fig.add_subplot(3, 3, 4)
    data.boxplot(column='temperatura_bearing', by='estado', ax=ax4)
    ax4.set_title('Temperatura do Bearing por Estado', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Estado')
    ax4.set_ylabel('Temperatura (°C)')
    plt.sca(ax4)
    plt.xticks([1, 2, 3, 4], ['Normal', 'Desgaste', 'Sobrecarga', 'Falha'])

    # (E) Comparação de Modelos
    ax5 = fig.add_subplot(3, 3, 5)
    model_names = list(results.keys())[:4]
    acc_values = [results[n]['accuracy'] for n in model_names]
    ax5.barh(model_names, acc_values,
             color=plt.cm.Set2(np.linspace(0.1, 0.9, len(model_names))))
    ax5.set_xlabel('Acurácia')
    ax5.set_title('Acurácia dos Modelos', fontsize=12, fontweight='bold')
    ax5.set_xlim([0, 1.1])
    for i, v in enumerate(acc_values):
        ax5.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)

    # (F) Série temporal simulada
    ax6 = fig.add_subplot(3, 3, 6)
    n_time = 200
    time = np.arange(n_time)
    temp_series = 75 + 0.03 * time + 5 * np.sin(time / 10) + np.random.normal(0, 3, n_time)
    threshold = 100
    ax6.plot(time, temp_series, 'b-', linewidth=1, label='Temperatura')
    ax6.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold}°C)')
    ax6.fill_between(time[temp_series > threshold], threshold,
                     temp_series * (temp_series > threshold) + threshold * (temp_series <= threshold),
                     alpha=0.3, color='red')
    ax6.set_xlabel('Amostras')
    ax6.set_ylabel('Temperatura (°C)')
    ax6.set_title('Monitoramento Contínuo - Temperatura', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=8)

    # (G) Curvas ROC
    ax7 = fig.add_subplot(3, 3, 7)
    for i in range(n_classes):
        y_binary = (y_test == i).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, y_prob_best[:, i])
        auc_score = roc_auc_score(y_binary, y_prob_best[:, i])
        ax7.plot(fpr, tpr, linewidth=1.5, label=f'{class_names[i]} (AUC={auc_score:.2f})')
    ax7.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax7.set_xlabel('FPR')
    ax7.set_ylabel('TPR')
    ax7.set_title('Curvas ROC', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=7)

    # (H) Importância de Features
    ax8 = fig.add_subplot(3, 3, 8)
    if hasattr(best_model, 'feature_importances_'):
        imp_sorted_idx = np.argsort(best_model.feature_importances_)[::-1]
        ax8.bar(range(len(selected_features)),
                best_model.feature_importances_[imp_sorted_idx],
                color=plt.cm.viridis(np.linspace(0.2, 0.8, len(selected_features))))
        ax8.set_xticks(range(len(selected_features)))
        ax8.set_xticklabels([selected_features[j] for j in imp_sorted_idx],
                            rotation=45, ha='right', fontsize=8)
        ax8.set_title('Importância das Features', fontsize=12, fontweight='bold')

    # (I) Matriz de Confusão
    ax9 = fig.add_subplot(3, 3, 9)
    cm = confusion_matrix(y_test, y_pred_best)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,
                                         display_labels=['N', 'D', 'S', 'F'])
    cm_display.plot(cmap='Blues', ax=ax9, values_format='d')
    ax9.set_title('Matriz de Confusão', fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('output/11_dashboard_completo.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Dashboard salvo: output/11_dashboard_completo.png")


if __name__ == '__main__':
    from sklearn.svm import SVC
    import seaborn as sns

    with open('output/dados_sensores.csv', 'r') as f:
        data = pd.read_csv(f)
    with open('output/treinamento_resultados.pkl', 'rb') as f:
        results = pickle.load(f)
    with open('output/preprocessamento.pkl', 'rb') as f:
        preproc = pickle.load(f)
    with open('output/selecao_features.pkl', 'rb') as f:
        sel = pickle.load(f)

    best_name = max(results, key=lambda k: results[k]['f1'])
    best_model = results[best_name]['model']

    y_pred_best = best_model.predict(sel['X_test_selected'])
    y_prob_best = best_model.predict_proba(sel['X_test_selected'])]

    gerar_dashboard(
        data, results,
        preproc['y_test'], y_pred_best, y_prob_best,
        best_model, sel['selected_features']
    )
