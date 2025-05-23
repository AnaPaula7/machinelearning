from etl.extrair import extract_data
from etl.transformar import transform_data
from etl.carregar import load_data

# Caminhos
input_file = r'D:\Unialfa\2025.1\trabalhomachine\data\raw\tx_rend_municipios_2023.xlsx'
output_file = r'D:\Unialfa\2025.1\trabalhomachine\data\processed\tx_rend_limpo.csv'

# ETL
df_raw = extract_data(input_file)
df_clean = transform_data(df_raw)
load_data(df_clean, output_file)
