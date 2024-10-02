import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# Função para carregar arquivos e verificar se estão vazios
def load_asv_file(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, sep=',', index_col=0)
        # df = pd.read_csv(file_path, sep='\t', index_col=0)
        if df.empty:
            raise ValueError(f"Erro: O arquivo ASV foi carregado, mas está vazio: {file_path}")
        
        # Transpor a tabela (inversão de linhas e colunas)
        df_transposed = df.T
        
        return df_transposed
    else:
        raise FileNotFoundError(f"Erro: Arquivo ASV não encontrado: {file_path}")

def transform_sample_counts(df):
    #df.sum(axis=1): 
    # Calcula a soma das contagens de todas as espécies (ou bins) para cada amostra
    #(cada linha), ou seja, a soma das colunas para cada linha.
    return df.div(df.sum(axis=1), axis=0)
    #df.div(df.sum(axis=1), axis=0): Divide cada valor de uma linha (amostra)
    #  pelo total de contagens daquela linha (soma dos bins para aquela amostra)


TRAIN_PATH="data/C6-C8_productivity_experiment/06_train_HRT_2Class.csv"
TEST_PATH="data/C6-C8_productivity_experiment/06_test_HRT_2Class.csv"
HRT_all_abs_ML_PATH="data/C6-C8_productivity_experiment/05_asv_bin_ML_abs.csv"
HRT_all_ra_ML_PATH="data/C6-C8_productivity_experiment/05_asv_bin_ML_ra.csv"
asv_all_csv2_PATH="data/C6-C8_productivity_experiment/05_asv_bin_ML_ra_read.csv"
meta_all_csv_PATH="data/C6-C8_productivity_experiment/04_metadata_bin_ML_abs.csv"


# Carregando arquivos ASV
asv_all_file = "C:/Users/Gonza/Desktop/dados_beetle_project/data/C6-C8_productivity_experiment/02_ASV_table_rarefied_13518_all.csv"

asv_all_csv = load_asv_file(asv_all_file)
print(asv_all_csv.head())
# Salvar o original
asv_all_csv_original = asv_all_csv.copy()

# Converter todas as colunas para numérico (se apropriado)
asv_all_csv = asv_all_csv.apply(pd.to_numeric, errors='coerce')


# Substituir NA por um pseudo-count pequeno (1e-6)
pseudo_count = 1e-6
asv_all_csv.fillna(pseudo_count, inplace=True)
print(asv_all_csv.isna().sum().sum())


# Carregar o arquivo de metadados
meta_all_file = "C:/Users/Gonza/Desktop/dados_beetle_project/data/C6-C8_productivity_experiment/01_metadata_productivity.txt"

meta_all_csv = pd.read_csv(meta_all_file, sep=",", index_col=0)


HRT_all_ra = transform_sample_counts(asv_all_csv)


# Extração de amostras relevantes para predição
HRT_metadata_ML_abs_all = meta_all_csv.copy()
HRT_asv_ML_abs_all = HRT_all_ra.copy()
HRT_asv_ML_abs_all.to_csv(HRT_all_ra_ML_PATH)

# Adicionar SampleID como coluna temporária
HRT_asv_ML_abs_all['SampleID'] = HRT_asv_ML_abs_all.index
HRT_metadata_ML_abs_all['SampleID'] = HRT_metadata_ML_abs_all.index

#Diferenças nos valores das chaves de junção:
HRT_asv_ML_abs_all['SampleID'] = HRT_asv_ML_abs_all['SampleID'].astype(str).str.strip()
HRT_metadata_ML_abs_all['SampleID'] = HRT_metadata_ML_abs_all['SampleID'].astype(str).str.strip()

# Realizar merge para unir as tabelas
HRT_ML_reactor_abs = pd.merge(HRT_asv_ML_abs_all, HRT_metadata_ML_abs_all[['SampleID', 'Hydraulic_retention_time', "Restriction_HRT_Time"]], on='SampleID', how='inner')
HRT_ML_reactor_abs = HRT_ML_reactor_abs[HRT_ML_reactor_abs["Restriction_HRT_Time"] == "yes"]
# Transformar SampleID de volta em índice e remover a coluna SampleID
HRT_ML_reactor_abs.set_index('SampleID', inplace=True)
HRT_ML_reactor_abs.index.name = None

HRT_ML_reactor_abs.drop(columns="Restriction_HRT_Time", inplace=True)

# Salvando em arquivos CSV
HRT_ML_reactor_abs.to_csv(HRT_all_abs_ML_PATH)




# Renomear a coluna 'Reactor' para 'y' (variável resposta)
HRT_ML_reactor_abs.rename(columns={'Hydraulic_retention_time': 'y'}, inplace=True)

# Verificar se o merge foi bem-sucedido
if HRT_ML_reactor_abs.empty:
    print("O merge resultou em um DataFrame vazio. Verifique se as chaves de junção estão corretas.")
else:
    print(HRT_ML_reactor_abs.head(6))


# 1) Filtro de prevalência (remover ASVs que são zero em todas as amostras)
print(HRT_ML_reactor_abs.shape)
zero_asvs = HRT_ML_reactor_abs.columns[(HRT_ML_reactor_abs == 0).all()]
HRT_ML_reactor_abs.drop(columns=zero_asvs, inplace=True)

print(HRT_ML_reactor_abs.shape)



# # Carregando arquivos ASV
# asv_all_file = "C:/Users/Gonza/Desktop/dados_beetle_project/data/data_all/map_complete_absolute_n_hits_table.tsv"

# asv_all_csv = load_asv_file(asv_all_file)

# asv_all_file2 = "C:/Users/Gonza/Desktop/dados_beetle_project/data/data_all/map_complete_relative_abundance_table.tsv"
# asv_all_csv2 = load_asv_file(asv_all_file2)

# # Salvar o original
# asv_all_csv_original = asv_all_csv.copy()

# # Converter todas as colunas para numérico (se apropriado)
# asv_all_csv = asv_all_csv.apply(pd.to_numeric, errors='coerce')
# asv_all_csv2 = asv_all_csv2.apply(pd.to_numeric, errors='coerce')



# # Substituir NA por um pseudo-count pequeno (1e-6)
# pseudo_count = 1e-6
# asv_all_csv.fillna(pseudo_count, inplace=True)
# asv_all_csv2.fillna(pseudo_count, inplace=True)
# print(asv_all_csv.isna().sum().sum())


# # Carregar o arquivo de metadados
# meta_all_file = "data/data_all/01_metadata_all.txt"
# meta_all_csv = pd.read_csv(meta_all_file, sep="\t", index_col=0)


# HRT_all_ra = transform_sample_counts(asv_all_csv)


# # Subset de samples (simulando como em R)
# HRT_all_abs_ML = asv_all_csv.copy()
# HRT_all_ra_ML = HRT_all_ra.copy()

# # Salvando em arquivos CSV
# HRT_all_abs_ML.to_csv(HRT_all_abs_ML_PATH)
# HRT_all_ra_ML.to_csv(HRT_all_ra_ML_PATH)
# asv_all_csv2.to_csv(asv_all_csv2_PATH)
# meta_all_csv.to_csv(meta_all_csv_PATH)

# # Extração de amostras relevantes para predição
# HRT_metadata_ML_abs_all = meta_all_csv.copy()
# HRT_asv_ML_abs_all = asv_all_csv.copy()

# HRT_metadata_ML_all = meta_all_csv.copy()
# HRT_asv_ML_all = asv_all_csv2.copy()


# print(f"shape1:{HRT_asv_ML_all.shape}")

# # Adicionar SampleID como coluna temporária
# HRT_asv_ML_abs_all['SampleID'] = HRT_asv_ML_abs_all.index
# HRT_metadata_ML_abs_all['SampleID'] = HRT_metadata_ML_abs_all.index
# HRT_asv_ML_all['SampleID'] = HRT_asv_ML_all.index
# HRT_metadata_ML_all['SampleID'] = HRT_metadata_ML_all.index


# # asv_sample_ids = list(HRT_asv_ML_all['SampleID'])
# # metadata_sample_ids = list(HRT_metadata_ML_all['SampleID'])

# # # Encontrar elementos que estão na primeira lista, mas não na segunda
# # asv_not_in_metadata = set(asv_sample_ids) - set(metadata_sample_ids)

# # # Encontrar elementos que estão na segunda lista, mas não na primeira
# # metadata_not_in_asv = set(metadata_sample_ids) - set(asv_sample_ids)

# # # Exibir as diferenças
# # print("Elementos presentes no HRT_asv_ML_all, mas não no HRT_metadata_ML_all:")
# # print(asv_not_in_metadata)

# # print("\nElementos presentes no HRT_metadata_ML_all, mas não no HRT_asv_ML_all:")
# # print(metadata_not_in_asv)

# #Diferenças nos valores das chaves de junção:
# HRT_asv_ML_abs_all['SampleID'] = HRT_asv_ML_abs_all['SampleID'].astype(str).str.strip()
# HRT_metadata_ML_abs_all['SampleID'] = HRT_metadata_ML_abs_all['SampleID'].astype(str).str.strip()
# HRT_asv_ML_all['SampleID'] = HRT_asv_ML_all['SampleID'].astype(str).str.strip()
# HRT_metadata_ML_all['SampleID'] = HRT_metadata_ML_all['SampleID'].astype(str).str.strip()




# # Realizar merge para unir as tabelas
# HRT_ML_reactor_abs = pd.merge(HRT_asv_ML_abs_all, HRT_metadata_ML_abs_all[['SampleID', 'Experiment']], on='SampleID', how='inner')
# HRT_ML_reactor = pd.merge(HRT_asv_ML_all, HRT_metadata_ML_all[['SampleID', 'Experiment']], on='SampleID', how='inner')


# # Transformar SampleID de volta em índice e remover a coluna SampleID
# HRT_ML_reactor_abs.set_index('SampleID', inplace=True)
# HRT_ML_reactor_abs.index.name = None


# HRT_ML_reactor.set_index('SampleID', inplace=True)
# HRT_ML_reactor.index.name = None


# # Renomear a coluna 'Reactor' para 'y' (variável resposta)
# HRT_ML_reactor_abs.rename(columns={'Experiment': 'y'}, inplace=True)
# HRT_ML_reactor.rename(columns={'Experiment': 'y'}, inplace=True)

# # Verificar se o merge foi bem-sucedido
# if HRT_ML_reactor_abs.empty:
# print("O merge resultou em um DataFrame vazio. Verifique se as chaves de junção estão corretas.")
# else:
# print(HRT_ML_reactor_abs.head(6))

# # Verificar se o merge foi bem-sucedido
# if HRT_ML_reactor.empty:
# print("O merge resultou em um DataFrame vazio. Verifique se as chaves de junção estão corretas.")
# else:
# print(f"shape4:{HRT_ML_reactor.shape}")


# # 1) Filtro de prevalência (remover ASVs que são zero em todas as amostras)
# zero_asvs = HRT_ML_reactor.columns[(HRT_ML_reactor == 0).all()]
# HRT_ML_reactor.drop(columns=zero_asvs, inplace=True)
# HRT_ML_reactor_abs.drop(columns=zero_asvs, inplace=True)

# 3) Divisão dos dados em conjunto de treino e teste
train_HRT_ML_reactor, test_HRT_ML_reactor = train_test_split(HRT_ML_reactor_abs, test_size=0.25, stratify=HRT_ML_reactor_abs['y'])



# Exportar os dados de treino e teste
train_HRT_ML_reactor.to_csv(TRAIN_PATH, index=False)
test_HRT_ML_reactor.to_csv(TEST_PATH, index=False)

# Carregar os conjuntos de dados - Experimento biorreator
train_df = train_HRT_ML_reactor.copy()
test_df = test_HRT_ML_reactor.copy()
print(f"train shape: {train_df.shape}")
print(f"train shape: {test_df.shape}")


# LabelEncoder para variável alvo y
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['y'])
y_test = label_encoder.transform(test_df['y'])
X_train = train_df.drop(columns=['y']).reset_index(drop=True)
X_test = test_df.drop(columns=['y']).reset_index(drop=True)


# # Verificação dos formatos
# print(f"X_train shape: {X_train.shape}")
# print(f"y_train shape: {y_train.shape}")
# print(f"X_test shape: {X_test.shape}")
# print(f"y_test shape: {y_test.shape}")




