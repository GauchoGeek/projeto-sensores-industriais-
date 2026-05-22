import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
import os

os.makedirs('output', exist_ok=True)


def selecionar_features(X_train_balanced, y_train_balanced, X_test_scaled, k=8, salvar=True):
    """
    Realiza a seleção de features usando ANOVA F-test.

    Parâmetros:
        X_train_balanced: DataFrame com features de treino balanceadas
        y_train_balanced: Series com labels de treino balanceados
        X_test_scaled: DataFrame com features de teste normalizadas
        k: número de features a selecionar
        salvar: bool, se True salva artefatos

    Retorna:
        dict com features selecionadas e o seletor
    """

    # ANOVA F-test - todas as features
    selector_anova = SelectKBest(score_func=f_classif, k='all')
    selector_anova.fit(X_train_balanced, y_train_balanced)

    feature_scores = pd.DataFrame({
        'Feature': X_train_balanced.columns,
        'F_Score_ANOVA': selector_anova.scores_,
        'P_Value': selector_anova.pvalues_
    }).sort_values('F_Score_ANOVA', ascending=False)

    print("=== Seleção de Features - ANOVA F-test ===")
    print(feature_scores.to_string(index=False))

    # Visualização
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(feature_scores['Feature'], feature_scores['F_Score_ANOVA'],
                   color=plt.cm.viridis(feature_scores['F_Score_ANOVA'] /
                                         feature_scores['F_Score_ANOVA'].max()))
    ax.set_xlabel('F-Score (ANOVA)', fontsize=12)
    ax.set_title('Importância das Features - ANOVA F-test', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    for bar, score in zip(bars, feature_scores['F_Score_ANOVA']):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{score:.1f}', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('output/06_feature_importance_anova.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Gráfico salvo: output/06_feature_importance_anova.png")

    # Seleção das melhores k features
    selector_best = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector_best.fit_transform(X_train_balanced, y_train_balanced)
    X_test_selected = selector_best.transform(X_test_scaled)

    selected_features = X_train_balanced.columns[selector_best.get_support()].tolist()
    print(f"\nMelhores features selecionadas ({len(selected_features)}): {selected_features}")

    resultado = {
        'X_train_selected': X_train_selected,
        'X_test_selected': X_test_selected,
        'selector': selector_best,
        'selected_features': selected_features,
        'feature_scores': feature_scores
    }

    if salvar:
        with open('output/selecao_features.pkl', 'wb') as f:
            pickle.dump(resultado, f)
        print("✅ Seleção de features salva em 'output/selecao_features.pkl'")

    return resultado


if __name__ == '__main__':
    with open('output/preprocessamento.pkl', 'rb') as f:
        preproc = pickle.load(f)

    resultado = selecionar_features(
        preproc['X_train_balanced'],
        preproc['y_train_balanced'],
        preproc['X_test']
    )
