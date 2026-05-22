import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay
)
import os

os.makedirs('output', exist_ok=True)


def treinar_e_avaliar(X_train_selected, y_train_balanced, X_test_selected, y_test, salvar=True):
    """
    Treina e avalia múltiplos algoritmos de classificação.

    Parâmetros:
        X_train_selected: array com features selecionadas de treino
        y_train_balanced: array com labels balanceados de treino
        X_test_selected: array com features selecionadas de teste
        y_test: array com labels de teste
        salvar: bool, se True salva artefatos

    Retorna:
        dict com resultados de todos os modelos
    """

    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        'SVM (RBF)': SVC(
            C=10, kernel='rbf', gamma='scale', probability=True, random_state=42
        ),
        'SVM (Linear)': SVC(
            C=1, kernel='linear', probability=True, random_state=42
        ),
        'KNN': KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=-1),
        'Logistic Regression': LogisticRegression(
            C=10, max_iter=1000, multi_class='multinomial', random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42
        )
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print("=" * 80)
    print("                    AVALIAÇÃO DOS MODELOS")
    print("=" * 80)

    for name, model in models.items():
        print(f"\n{'─' * 60}")
        print(f"  📊 Modelo: {name}")
        print(f"{'─' * 60}")

        model.fit(X_train_selected, y_train_balanced)
        y_pred = model.predict(X_test_selected)
        y_prob = model.predict_proba(X_test_selected)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        try:
            if y_prob.shape[1] > 2:
                auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
            else:
                auc = roc_auc_score(y_test, y_prob[:, 1])
        except:
            auc = 0.0

        cv_scores = cross_val_score(model, X_train_selected, y_train_balanced,
                                     cv=cv, scoring='accuracy')

        results[name] = {
            'accuracy': acc, 'precision': prec, 'recall': rec,
            'f1': f1, 'auc': auc,
            'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(),
            'model': model
        }

        print(f"  Acurácia:            {acc:.4f}")
        print(f"  Precisão (weighted):  {prec:.4f}")
        print(f"  Recall (weighted):    {rec:.4f}")
        print(f"  F1-Score (weighted):  {f1:.4f}")
        print(f"  AUC (OVR):            {auc:.4f}")
        print(f"  CV Acurácia:          {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred,
              target_names=['Normal', 'Desgaste Inicial', 'Sobrecarga', 'Falha Iminente'],
              zero_division=0))

    # Tabela comparativa
    results_display = {k: {kk: vv for kk, vv in v.items() if kk != 'model'} for k, v in results.items()}
    results_df = pd.DataFrame(results_display).T.round(4)
    results_df = results_df.sort_values('f1', ascending=False)

    print("\n" + "=" * 80)
    print("  📋 TABELA COMPARATIVA DOS MODELOS")
    print("=" * 80)
    print(results_df.to_string())

    # Gráfico comparativo
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    metrics_to_plot = ['accuracy', 'f1', 'precision', 'recall']
    x = np.arange(len(models))
    width = 0.2

    for i, metric in enumerate(metrics_to_plot):
        axes[0].bar(x + i * width, results_df[metric].values, width,
                    label=metric.capitalize(), alpha=0.85,
                    color=plt.cm.Set2(i / len(metrics_to_plot)))

    axes[0].set_xlabel('Modelo', fontsize=12)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Comparação de Métricas por Modelo', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(results_df.index, rotation=15, ha='right')
    axes[0].legend(loc='lower right')
    axes[0].set_ylim([0, 1.05])

    axes[1].barh(results_df.index, results_df['cv_mean'].values,
                 xerr=results_df['cv_std'].values,
                 color=plt.cm.coolwarm(results_df['cv_mean'].values /
                                        results_df['cv_mean'].max()),
                 edgecolor='black', linewidth=0.8, capsize=5)
    axes[1].set_xlabel('Acurácia (CV)', fontsize=12)
    axes[1].set_title('Validação Cruzada (média ± desvio)', fontsize=14, fontweight='bold')
    axes[1].set_xlim([0, 1.05])

    plt.tight_layout()
    plt.savefig('output/07_comparacao_modelos.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Gráfico salvo: output/07_comparacao_modelos.png")

    if salvar:
        with open('output/treinamento_resultados.pkl', 'wb') as f:
            pickle.dump(results, f)
        print("✅ Resultados salvos em 'output/treinamento_resultados.pkl'")

    return results


if __name__ == '__main__':
    with open('output/selecao_features.pkl', 'rb') as f:
        sel = pickle.load(f)
    with open('output/preprocessamento.pkl', 'rb') as f:
        preproc = pickle.load(f)

    treinar_e_avaliar(
        sel['X_train_selected'],
        preproc['y_train_balanced'],
        sel['X_test_selected'],
        preproc['y_test']
    )
