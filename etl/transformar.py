import pandas as pd

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        'Unnamed: 0': 'Ano',
        'Unnamed: 1': 'Regiao',
        'Unnamed: 2': 'UF',
        'Unnamed: 3': 'Cod_Municipio',
        'Unnamed: 4': 'Municipio',
        'Unnamed: 5': 'Localizacao',
        'Unnamed: 6': 'Dependencia',
        'Total': 'Aprovacao_Total',
        'Anos Iniciais': 'Aprovacao_Anos_Iniciais',
        'Anos Finais': 'Aprovacao_Anos_Finais',
        'Total  .1': 'Reprovacao_Total',
        'Anos Iniciais.1': 'Reprovacao_Anos_Iniciais',
        'Anos Finais.1': 'Reprovacao_Anos_Finais',
        'Total  .2': 'Abandono_Total',
        'Anos Iniciais.2': 'Abandono_Anos_Iniciais',
        'Anos Finais.2': 'Abandono_Anos_Finais',
    })

    # Substituir '--' por NaN e garantir que tipos não sejam forçados silenciosamente
    df = df.replace('--', pd.NA)
    df = df.infer_objects(copy=False)  # recomendado pela warning

    # Converte colunas numéricas com segurança
    for col in df.columns:
        if isinstance(col, str) and col.startswith(('Aprovacao', 'Reprovacao', 'Abandono')):
            if col in df.columns and isinstance(df[col], pd.Series):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    print(f"Erro ao converter a coluna {col}: {e}")
    return df
