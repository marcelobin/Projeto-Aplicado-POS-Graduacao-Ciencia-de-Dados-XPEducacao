import pandas as pd
from joblib import load

def preprocessar_e_aplicar_modelo(caminho_arquivo_csv, caminho_modelo):
    # Carregar o modelo treinado
    modelo = load(caminho_modelo)

    # Ler o arquivo CSV
    df = pd.read_csv(caminho_arquivo_csv)

    # Converter a coluna 'pressao_sanguinea'
    df[['pressao_sistolica', 'pressao_diastolica']] = df['pressao_sanguinea'].str.split('/', expand=True)
    df.drop('pressao_sanguinea', axis=1, inplace=True)
    df['pressao_diastolica'] = df['pressao_diastolica'].astype('int')
    df['pressao_sistolica'] = df['pressao_sistolica'].astype('int')

    # Criar variáveis dummies
    df = pd.get_dummies(df, columns=['sexo', 'categoria_IMC'], drop_first=True, dtype='int')

    # Remover colunas desnecessárias
    X = df.drop(['ID', 'ocupacao'], axis=1)

    # Aplicar o modelo
    previsao = modelo.predict(X)

    return previsao
