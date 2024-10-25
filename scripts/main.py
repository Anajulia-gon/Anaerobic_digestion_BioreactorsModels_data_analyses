#Importing
# pip install -r requirements.txt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from processamentoBioreactor import  X_train, X_test, y_train, y_test
from model_evaluation import evaluate_model2
from model_evaluation import print_validation_evaluation
from model_evaluation import print_test_evaluation
from utils import save_model, load_model
from utils import exportar_csv
from configs import pipelines  # Hiperparâmetros
from configs import params
from configs import SCALERS
from sklearn.preprocessing import LabelEncoder
from shap_plots import generate_shap_plots
from limeplots import generate_lime_explanation
# from processamento import X_train, X_test, y_train, y_test
from configs import pipelines, params 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings
import os
warnings.filterwarnings("ignore")

# # Carregar os conjuntos de dados - Experimento biorreator
# train_df = pd.read_csv("train_HRT_2Class.csv")
# test_df = pd.read_csv("test_HRT_2Class.csv")

# # LabelEncoder para variável alvo y
# label_encoder = LabelEncoder()
# y_train = label_encoder.fit_transform(train_df['y'])
# y_test = label_encoder.transform(test_df['y'])
# X_train = train_df.drop(columns=['y']).reset_index(drop=True)
# X_test = test_df.drop(columns=['y']).reset_index(drop=True)


# Verificação dos formatos
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Lista para armazenar os resultados
output_dir = 'C:/Users/Gonza/Desktop/TCC/outputs/Bettle/Div1/plots/Confusion_matriz'
if __name__ == "__main__":
        # Iteração sobre os modelos
        best_models = {}
        for model_name in pipelines:
            print(f"Running GridSearch for {model_name}")

            # Rodar o GridSearchCV para encontrar o melhor modelo
            model_search = evaluate_model2(pipelines[model_name], params[model_name], X_train, y_train)
            # Salvando o modelo
            save_model(model_search.best_estimator_, model_name)  # salva o melhor modelo treinado após o GridSearchCV.

            # Carregar o melhor modelo salvo
            best_model = load_model(model_name)

            # Armazenar o melhor modelo
            best_models[model_name] = best_model
            print(f"Best parameters for {model_name}: {model_search.best_params_}")
            print(f"Best cross-validation accuracy for {model_name}: {model_search.best_score_}")

            # exportar_csv(model_search, model_name, pipelines, SCALERS)
            print_validation_evaluation(model_search)
            print_test_evaluation(model_search, X_test, y_test)

            # Plotar a matriz de confusão
            CM = confusion_matrix(y_test, best_model.predict(X_test))
            disp = ConfusionMatrixDisplay(confusion_matrix=CM)
            disp.plot(cmap='Blues', colorbar=True)
            plt.tight_layout()  # Ajusta automaticamente o layout para evitar corte
            # Salvar o gráfico como PNG
            plt.savefig(os.path.join(output_dir, f'matriz_confusao_{model_name}.png'), bbox_inches='tight')

            # Fechar a figura após o salvamento
            plt.close()

            # Avaliar no conjunto de teste
            print(classification_report(y_test, best_model.predict(X_test)))

            # # Gerar gráficos SHAP
            # generate_shap_plots(best_model.named_steps['classifier'], model_name, X_train, X_test)

            # # Gerar explicação LIME
            # generate_lime_explanation(best_model, X_train, X_test)