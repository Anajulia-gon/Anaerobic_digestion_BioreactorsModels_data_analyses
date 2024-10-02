import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler, QuantileTransformer, Binarizer, PowerTransformer, FunctionTransformer, Normalizer, LabelEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb
import xgboost as xgb
from sklearn.svm import SVC
import shap
from lime.lime_tabular import LimeTabularExplainer
from processamento import X_train, X_test, y_train, y_test
import warnings
import logging  
# Ignorar todos os avisos
warnings.filterwarnings('ignore')

# Definir os modelos e seus parâmetros para GridSearch
# Definir nível de logging para ERROR (evitar mensagens de INFO ou WARNING de bibliotecas)
logging.basicConfig(level=logging.ERROR)

n_samples = X_train.shape[0]

# Definindo as pipelines sem warnings
pipelines = {
    "Logistic Regression": Pipeline([
        ('scaler', 'passthrough'),
        ('classifier', LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=500))
    ]),
    "Naive Bayes": Pipeline([
        ('scaler', 'passthrough'),
        ('classifier', GaussianNB())
    ]),
    "Decision Tree": Pipeline([
        ('classifier', DecisionTreeClassifier())
    ]),
    "Random Forest": Pipeline([
        ('classifier', RandomForestClassifier())
    ]),
    "Bagging": Pipeline([
        ('classifier', BaggingClassifier())
    ]),
    "AdaBoost": Pipeline([
        ('classifier', AdaBoostClassifier(algorithm='SAMME.R'))
    ]),
    "Gradient Boosting": Pipeline([
        ('classifier', GradientBoostingClassifier())
    ]),
    "Extra Trees": Pipeline([
        ('classifier', ExtraTreesClassifier())
    ]),
    "LightGBM": Pipeline([
        ('scaler', 'passthrough'),
        ('classifier', lgb.LGBMClassifier(verbosity=-1))  # Suprimir logs do LightGBM
    ]),
    "K-Neighbors": Pipeline([
        ('scaler', 'passthrough'),
        ('classifier', KNeighborsClassifier())
    ]),
    "SGD": Pipeline([
        ('scaler', 'passthrough'),
        ('classifier', SGDClassifier())
    ]),
    "XGBoost": Pipeline([
        ('scaler', 'passthrough'),
        ('classifier', xgb.XGBClassifier(
            use_label_encoder=False,  # Desativa o uso do codificador de rótulos
            eval_metric='logloss',    # Função de perda para problemas binários
            objective='binary:logistic',  # Configuração padrão para classificação binária
            verbosity=0  # Suprimir logs do XGBoost
        ))
    ]),
    "SVM": Pipeline([
        ('scaler', 'passthrough'),
        ('classifier', SVC())  # Configuração padrão para classificação binária
    ])
}
params = {
    "Logistic Regression": {
        'scaler': [None, MinMaxScaler(), MaxAbsScaler(), RobustScaler(), StandardScaler(), QuantileTransformer(), Binarizer(), PowerTransformer(), FunctionTransformer()],
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__multi_class': ['ovr', 'multinomial']
    },
    "Naive Bayes": {
        'scaler': [None, MinMaxScaler(), MaxAbsScaler(), RobustScaler(), StandardScaler(), PowerTransformer(), Binarizer()]
    },
    "Decision Tree": {
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__criterion': ['gini', 'entropy']
    },
    "Random Forest": {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__criterion': ['gini', 'entropy'],
        'classifier__bootstrap': [True, False]
    },
    "Bagging": {
        'classifier__n_estimators': [10, 20, 30],
        'classifier__bootstrap': [True, False],
        'classifier__bootstrap_features': [True, False]
    },
    "AdaBoost": {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.1, 1]
    },
    "Gradient Boosting": {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.1, 1],
        'classifier__max_depth': [3, 5, 7],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    },
    "Extra Trees": {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    },
    "LightGBM": {
        'scaler': [None, Normalizer()],
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.1, 1],
        'classifier__max_depth': [-1, 10, 20, 30]
        # 'classifier__objective': ['multiclass']
    },
    "K-Neighbors": {
        'scaler': [None, MinMaxScaler(), MaxAbsScaler(), RobustScaler(), StandardScaler(), PowerTransformer(), FunctionTransformer(), QuantileTransformer(), Binarizer()],
        'classifier__n_neighbors': [3, 5, 7, 9],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    },
    "SGD": {
        'scaler': [None, MinMaxScaler(), MaxAbsScaler(), RobustScaler(), StandardScaler(), PowerTransformer(), FunctionTransformer(), QuantileTransformer(), Binarizer()],
        'classifier__alpha': [0.0001, 0.001, 0.01, 0.1],
        'classifier__max_iter': [1000, 2000, 3000],
        'classifier__penalty': ['l2', 'l1', 'elasticnet'],
        'classifier__loss': ['hinge', 'log', 'modified_huber']
    },
    "XGBoost": {
        'scaler': [None, MinMaxScaler(), MaxAbsScaler(), RobustScaler(), StandardScaler(), Binarizer(), PowerTransformer(), FunctionTransformer(), QuantileTransformer()],
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.1, 0.3],
        'classifier__max_depth': [3, 5, 7],
        'classifier__min_child_weight': [1, 3, 5],
        'classifier__gamma': [0, 0.1, 0.3],
        'classifier__subsample': [0.8, 1.0],
        'classifier__colsample_bytree': [0.8, 1.0],
        # 'classifier__objective': ['multi:softmax', 'multi:softprob']
    },
    "SVM": {
        'scaler': [None, MinMaxScaler(), MaxAbsScaler(), RobustScaler(), StandardScaler(), Binarizer(), PowerTransformer(), FunctionTransformer(), QuantileTransformer()],
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'classifier__gamma': ['scale', 'auto']
    }
}


# Função para avaliação dos modelos
def evaluate_model(model, params, X_train, y_train):
    """
    Função para realizar a busca em grade (GridSearchCV) para um modelo específico.

    Args:
    model: O pipeline do modelo que será avaliado.
    params: O dicionário de hiperparâmetros para o GridSearchCV.
    X_train (array-like): Características dos dados de treino.
    y_train (array-like): Rótulos dos dados de treino.

    Retorna:
    grid_search: O objeto GridSearchCV ajustado.
    """
    grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search

def print_evaluation(y_true, y_pred):
    """
    Função para calcular e imprimir as métricas de avaliação.

    Args:
    y_true (array-like): Rótulos verdadeiros dos dados de teste.
    y_pred (array-like): Rótulos previstos pelo modelo.
    """
    accuracy = accuracy_score(y_true, y_pred)
    print(f"----------------------------------------------------")
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_true, y_pred))
    print(f"----------------------------------------------------")
# Função para plotar a importância das variáveis
def plot_feature_importance(model, model_name, top_n=30):
    if hasattr(model, 'feature_importances_'):   #Verifica se o modelo fornecido possui o atributo feature_importances
        importances = model.feature_importances_
        # # Criar uma lista de tuplas (índice, importância) e ordenar em ordem decrescente pela importância
        # indices = sorted(enumerate(importances), key=lambda x: x[1], reverse=True)
        # sorted_indices = [i[0] for i in indices]
        indices = np.argsort(importances)[::-1]   #Ordena os índices das features em ordem decrescente de importância - operador slice  - [start:stop:step]
        X_test.columns
        features = [f'{i}' for i in range(X_train.shape[1])]

        # Mostrar apenas as top_n características
        top_indices = indices[:top_n]  #Seleciona os primeiros top_n índices das features mais importantes
        top_importances = importances[top_indices] #fitra as features de maior indice (mais importantes)
        top_features = [features[i] for i in top_indices]  #nomes das top_n features mais importantes

        plt.figure(figsize=(10, 8))
        plt.title(f'Feature Importances for {model_name}')
        plt.barh(range(top_n), top_importances, align='center')
        plt.yticks(range(top_n), top_features)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.gca().invert_yaxis()  # Inverte o eixo Y para mostrar as features mais importantes no topo
        plt.tight_layout()
        plt.show()
def plot_shap(model, model_name, X_test):

    # Cria o explicador SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    # Plota os valores SHAP para a classe 0
    # shap_values_class_0 = shap_values[:, :, 0]
    plt.title(f"SHAP Values for {model_name} - Class 0")
    shap.summary_plot(shap_values[:, :, 0], X_test, feature_names=X_test.columns) #For single output explanations this is a matrix of SHAP values (# samples x # features).
    plt.show()
    # Plota os valores SHAP para a classe 1
    # shap_values_class_0 = shap_values[:, :, 0]
    plt.title(f"SHAP Values for {model_name} - Class 1")
    shap.summary_plot(shap_values[:, :, 1], X_test, feature_names=X_test.columns) #For single output explanations this is a matrix of SHAP values (# samples x # features).
    plt.show()
    # Plota os valores SHAP para a classe 0
    # shap_values_class_0 = shap_values[:, :, 0]
    plt.title(f"SHAP Values for {model_name} - Class 2")
    shap.summary_plot(shap_values[:, :, 2], X_test, feature_names=X_test.columns) #For single output explanations this is a matrix of SHAP values (# samples x # features).
    plt.show()



### Função `plot_lime`
def plot_lime(model, model_name, X_test):
    classifier = model
    explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns,
        class_names=['0', '1'],
        discretize_continuous=True
    )

    i = 0  # Apenas para o primeiro exemplo
    exp = explainer.explain_instance(X_test.iloc[i].values, classifier.predict_proba, num_features=5)
    exp.show_in_notebook(show_table=True)

def evaluate_models(pipelines, params, X_train, y_train, X_test, y_test):
    """
    Função para avaliar múltiplos modelos de aprendizado de máquina usando GridSearchCV e retornar os melhores modelos.

    Args:
    pipelines (dict): Um dicionário onde a chave é o nome do modelo e o valor é o objeto do pipeline.
    params (dict): Um dicionário de hiperparâmetros para cada modelo.
    X_train (array-like): Características dos dados de treino.
    y_train (array-like): Rótulos dos dados de treino.
    X_test (array-like): Características dos dados de teste.
    y_test (array-like): Rótulos dos dados de teste.

    Retorna:
    dict: Um dicionário contendo os melhores modelos para cada algoritmo.
    """
    best_models = {}

    for model_name, pipeline in pipelines.items():
        print(f"Executando GridSearch para {model_name}")

        # Chama a função evaluate_model para realizar o GridSearchCV
        model_search = evaluate_model(pipeline, params[model_name], X_train, y_train)

        # Armazena o melhor estimador no dicionário
        best_models[model_name] = model_search.best_estimator_

        # Imprime os melhores parâmetros e a acurácia da validação cruzada
        print(f"Melhores parâmetros para {model_name}: {model_search.best_params_}")
        print(f"Melhor acurácia de validação cruzada para {model_name}: {model_search.best_score_}")

        # Avalia o modelo nos dados de teste
        y_pred = best_models[model_name].predict(X_test)
        print(f"Avaliação nos dados de teste para {model_name}:")
        print_evaluation(y_test, y_pred)
    best_models = {
    model_name: instantiate_model(model_name, extract_model_params(pipeline))
    for model_name, pipeline in best_estimators.items()
    }
    return best_models



# print("y=================")
# print(y_train)
# print(y_test)


# # Verificação dos formatos
# print(f"X_train shape: {X_train.shape}")
# print(f"y_train shape: {y_train.shape}")
# print(f"X_test shape: {X_test.shape}")
# print(f"y_test shape: {y_test.shape}")
X_train = X_train.copy()
X_test = X_test.copy()
y_train= y_train.copy()
y_test= y_test.copy()

best_models = evaluate_models(pipelines, params, X_train, y_train, X_test, y_test)
print(best_models)