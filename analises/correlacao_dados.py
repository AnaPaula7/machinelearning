import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Caminho do arquivo tratado
file_path = r'D:\Unialfa\2025.1\trabalhomachine\data\processed\tx_rend_limpo.csv'

# Carregar dados
df = pd.read_csv(file_path)

# Selecionar colunas numéricas de interesse
colunas_numericas = [
    'Aprovacao_Total',
    'Aprovacao_Anos_Iniciais',
    'Aprovacao_Anos_Finais',
    'Reprovacao_Total',
    'Reprovacao_Anos_Iniciais',
    'Reprovacao_Anos_Finais',
    'Abandono_Total',
    'Abandono_Anos_Iniciais',
    'Abandono_Anos_Finais'
]

df_corr = df[colunas_numericas].dropna()

# Calcular matriz de correlação
matriz_corr = df_corr.corr(method='pearson')  

# Plotar heatmap
plt.figure(figsize=(10,8))
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('📈 Matriz de Correlação entre Indicadores Educacionais')
plt.tight_layout()
plt.show()


''' Correlações próximas de +1 indicam forte relação positiva.

Correlações próximas de -1 indicam relação inversa.

Ex: se Aprovação tem correlação negativa forte com Reprovação, isso faz sentido lógico e 
confirma a consistência dos dados '''