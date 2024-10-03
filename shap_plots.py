import matplotlib
matplotlib.use('Agg')  # Usar backend para geração de gráficos sem exibição
import matplotlib.pyplot as plt
import shap
from configs import SHAP_CONFIGS
from utils import ensure_directory_exists
import os
# Definir os diretórios de salvamento
SAVE_DIR_SUMMARY = "C:/Users/Gonza/Desktop/TCC/outputs/Bettle/plots/SummaryPlots/"
SAVE_DIR_WATERFALL = "C:/Users/Gonza/Desktop/TCC/outputs/Bettle/plots/WaterfallPlots/"

# Gera o tipo de explainer com base no modelo
def generate_shap_plots(model, model_name, X_train, X_test):
    if model_name in SHAP_CONFIGS['tree_based']:
        plot_shap_tree_based(model, model_name, X_test)
    elif model_name in SHAP_CONFIGS['kernel_based']:
        plot_shap_kernel(model, model_name, X_train, X_test)
    elif model_name in SHAP_CONFIGS['gradient_based']:
        plot_shap_gradient_based(model, model_name, X_train, X_test)
    elif model_name in SHAP_CONFIGS['logistic_regression']:
        plot_shap_logistic_regression(model, model_name, X_train, X_test)

def plot_shap_tree_based(model, model_name, X_test, SAVE_DIR_SUMMARY=SAVE_DIR_SUMMARY, SAVE_DIR_WATERFALL=SAVE_DIR_WATERFALL ):

    # Verificar e criar os diretórios, se necessário
    ensure_directory_exists(SAVE_DIR_SUMMARY)
    ensure_directory_exists(SAVE_DIR_WATERFALL)

    # Cria o explicador SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    if len(shap_values.shape) == 3:  # Ele devolve uma matriz onde a terceira coluna é a classe
        # Para cada classe
        for i in range(shap_values.shape[2]):
            plt.figure(figsize=(12, 6))
            plt.title(f"SHAP Summary Plot for {model_name} - Class {i}")
            shap.summary_plot(shap_values[:, :, i], X_test, feature_names=X_test.columns, max_display=20, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR_SUMMARY, f"{model_name}_shap_summary_class_{i}.png"), bbox_inches='tight')
            print(f"Gráfico salvo: {model_name}_shap_summary_class_{i}.png")
            plt.close()

            # Exibir Waterfall Plot para uma amostra (primeira amostra)
            shap_exp = shap.Explanation(
                values=shap_values[0, :, i],  # Primeira amostra, classe i
                base_values=explainer.expected_value[i],
                data=X_test.iloc[0]
            )
            plt.figure(figsize=(10, 6))
            plt.title(f"SHAP Waterfall Plot for {model_name} - Class {i} (Primeira Amostra)")
            shap.plots.waterfall(shap_exp, max_display=10)  # Limitar a 10 atributos
            plt.savefig(os.path.join(SAVE_DIR_WATERFALL, f"{model_name}_waterfall_class_{i}.png"), bbox_inches='tight')
            print(f"Gráfico salvo: {model_name}_waterfall_class_{i}.png")
            plt.close()

    else:
        # Caso o modelo seja binário (shap_values será uma matriz 2D)
        plt.figure(figsize=(12, 6))
        plt.title(f"SHAP Summary Plot for {model_name}")
        shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR_SUMMARY, f"{model_name}_shap_summary.png"), bbox_inches='tight')
        print(f"Gráfico salvo: {model_name}_shap_summary.png")
        plt.close()

        # Exibir Waterfall Plot para uma amostra (primeira amostra)
        shap_exp = shap.Explanation(
            values=shap_values[0],  # Primeira amostra
            base_values=explainer.expected_value,
            data=X_test.iloc[0]
        )
        plt.figure(figsize=(10, 6))
        plt.title(f"SHAP Waterfall Plot for {model_name} - Primeira Amostra")
        shap.plots.waterfall(shap_exp, max_display=10)  # Limitar a 10 atributos
        plt.savefig(os.path.join(SAVE_DIR_WATERFALL, f"{model_name}_waterfall_primeira_amostra.png"), bbox_inches='tight')
        print(f"Gráfico salvo: {model_name}_waterfall_primeira_amostra.png")
        plt.close()

def plot_shap_kernel(model, model_name, X_train, X_test, SAVE_DIR_SUMMARY=SAVE_DIR_SUMMARY, SAVE_DIR_WATERFALL=SAVE_DIR_WATERFALL):

    # Verificar e criar os diretórios, se necessário
    ensure_directory_exists(SAVE_DIR_SUMMARY)
    ensure_directory_exists(SAVE_DIR_WATERFALL)

    # Cria o explicador SHAP para modelos genéricos usando KernelExplainer
    explainer = shap.KernelExplainer(model.predict_proba, X_train)  # Usar os dados de treino como baseline
    shap_values = explainer.shap_values(X_test)

    if len(shap_values.shape) == 3:  # Ele devolve uma matriz onde a terceira coluna é a classe
        # Para cada classe
        for i in range(shap_values.shape[2]):
            plt.figure(figsize=(12, 6))
            plt.title(f"SHAP Summary Plot for {model_name} - Class {i}")
            shap.summary_plot(shap_values[:, :, i], X_test, feature_names=X_test.columns, max_display=15, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR_SUMMARY, f"{model_name}_shap_summary_class_{i}.png"), bbox_inches='tight')
            print(f"Gráfico salvo: {model_name}_shap_summary_class_{i}.png")
            plt.close()

            # Exibir Waterfall Plot para uma amostra (primeira amostra)
            shap_exp = shap.Explanation(
                values=shap_values[0, :, i],  # Primeira amostra, classe i
                base_values=explainer.expected_value[i],
                data=X_test.iloc[0]
            )
            plt.figure(figsize=(10, 6))
            plt.title(f"SHAP Waterfall Plot for {model_name} - Class {i} (Primeira Amostra)")
            shap.plots.waterfall(shap_exp, max_display=10)
            plt.savefig(os.path.join(SAVE_DIR_WATERFALL, f"{model_name}_waterfall_class_{i}.png"), bbox_inches='tight')
            print(f"Gráfico salvo: {model_name}_waterfall_class_{i}.png")
            plt.close()

    else:
        # Caso o modelo seja binário (shap_values será uma matriz 2D)
        plt.figure(figsize=(12, 6))
        plt.title(f"SHAP Summary Plot for {model_name}")
        shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR_SUMMARY, f"{model_name}_shap_summary.png"), bbox_inches='tight')
        print(f"Gráfico salvo: {model_name}_shap_summary.png")
        plt.close()

        # Exibir Waterfall Plot para uma amostra (primeira amostra)
        shap_exp = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_test.iloc[0]
        )
        plt.figure(figsize=(10, 6))
        plt.title(f"SHAP Waterfall Plot for {model_name} - Primeira Amostra")
        shap.plots.waterfall(shap_exp, max_display=10)
        plt.savefig(os.path.join(SAVE_DIR_WATERFALL, f"{model_name}_waterfall_primeira_amostra.png"), bbox_inches='tight')
        print(f"Gráfico salvo: {model_name}_waterfall_primeira_amostra.png")
        plt.close()

def plot_shap_gradient_based(model, model_name, X_train, X_test, SAVE_DIR_SUMMARY=SAVE_DIR_SUMMARY, SAVE_DIR_WATERFALL=SAVE_DIR_WATERFALL):

    # Verificar e criar os diretórios, se necessário
    ensure_directory_exists(SAVE_DIR_SUMMARY)
    ensure_directory_exists(SAVE_DIR_WATERFALL)

    # Cria o explicador SHAP para modelos lineares
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test)

    # Exibir o Summary Plot
    plt.figure(figsize=(12, 6))
    plt.title(f"SHAP Summary Plot for {model_name}")
    shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR_SUMMARY, f"{model_name}_shap_summary.png"), bbox_inches='tight')
    print(f"Gráfico salvo: {model_name}_shap_summary.png")
    plt.close()

    # Exibir o Waterfall Plot para a primeira amostra
    shap_exp = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_test.iloc[0]
    )
    plt.figure(figsize=(10, 6))
    plt.title(f"SHAP Waterfall Plot for {model_name} - Primeira Amostra")
    shap.plots.waterfall(shap_exp, max_display=10)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR_WATERFALL, f"{model_name}_waterfall_primeira_amostra.png"), bbox_inches='tight')
    print(f"Gráfico salvo: {model_name}_waterfall_primeira_amostra.png")
    plt.close()

def plot_shap_logistic_regression(model, model_name, X_train, X_test, SAVE_DIR_SUMMARY=SAVE_DIR_SUMMARY, SAVE_DIR_WATERFALL=SAVE_DIR_WATERFALL):

    # Verificar e criar os diretórios, se necessário
    ensure_directory_exists(SAVE_DIR_SUMMARY)
    ensure_directory_exists(SAVE_DIR_WATERFALL)

    # Cria o explicador SHAP para modelos lineares
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test)

    # Exibir o Summary Plot
    plt.figure(figsize=(12, 6))
    plt.title(f"SHAP Summary Plot for {model_name}")
    shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR_SUMMARY, f"{model_name}_shap_summary.png"), bbox_inches='tight')
    print(f"Gráfico salvo: {model_name}_shap_summary.png")
    plt.close()

    # Exibir o Waterfall Plot para a primeira amostra
    shap_exp = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_test.iloc[0]
    )
    plt.figure(figsize=(10, 6))
    plt.title(f"SHAP Waterfall Plot for {model_name} - First Sample")
    shap.plots.waterfall(shap_exp, max_display=10)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR_WATERFALL, f"{model_name}_waterfall_first_sample.png"), bbox_inches='tight')
    print(f"Gráfico salvo: {model_name}_waterfall_first_sample.png")
    plt.close()

