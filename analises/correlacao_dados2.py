import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# Caminho para os dados tratados
file_path = r'D:\Unialfa\2025.1\trabalhomachine\data\processed\tx_rend_limpo.csv'

# ===============================
# 1. Carregamento e pré-processamento
# ===============================
df = pd.read_csv(file_path, low_memory=False)
df = df.dropna(subset=['Aprovacao_Total', 'Reprovacao_Total', 'Abandono_Total'])

# ===============================
# 2. Correlação entre indicadores
# ===============================
colunas_numericas = [
    'Aprovacao_Total', 'Aprovacao_Anos_Iniciais', 'Aprovacao_Anos_Finais',
    'Reprovacao_Total', 'Reprovacao_Anos_Iniciais', 'Reprovacao_Anos_Finais',
    'Abandono_Total', 'Abandono_Anos_Iniciais', 'Abandono_Anos_Finais'
]

df_corr = df[colunas_numericas].dropna()
corr_matrix = df_corr.corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matriz de Correlação entre Indicadores Educacionais')
plt.tight_layout()
plt.show()

# ===============================
# 3. Comparação entre Localização (Urbana x Rural)
# ===============================
df_local = df[df['Localizacao'].isin(['Urbana', 'Rural'])]

plt.figure(figsize=(8,6))
sns.boxplot(x='Localizacao', y='Aprovacao_Total', data=df_local, hue='Localizacao', palette='Set2', legend=False)
plt.title('Diferença na Aprovação Total por Localização')
plt.tight_layout()
plt.show()

# ===============================
# 4. Comparação por Dependência Administrativa
# ===============================
df_dep = df[df['Dependencia'].notna()]

plt.figure(figsize=(10,6))
sns.boxplot(x='Dependencia', y='Aprovacao_Total', data=df_dep, palette='Set3')
plt.title('Diferença na Aprovação Total por Dependência Administrativa')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ===============================
# 5. Clusterização de Municípios
# ===============================
# Selecionar e normalizar os dados numéricos
features = df[['Aprovacao_Total', 'Reprovacao_Total', 'Abandono_Total']].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Aplicar KMeans com 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Adicionar resultado ao DataFrame original
df_clustered = df.copy()
df_clustered = df_clustered.loc[features.index]
df_clustered['Cluster'] = clusters

# Visualização dos clusters
plt.figure(figsize=(10,6))
sns.scatterplot(
    x='Aprovacao_Total', y='Abandono_Total',
    hue='Cluster', data=df_clustered, palette='Set1'
)
plt.title('Clusterização de Municípios (Aprovação vs Abandono)')
plt.tight_layout()
plt.show()

# ===============================
# 6. Modelo preditivo: Regressão Linear para Abandono_Total
# ===============================
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Selecionar colunas para o modelo
df_model = df[['Aprovacao_Total', 'Reprovacao_Total', 'Localizacao', 'Dependencia', 'Abandono_Total']].dropna()

# Codificar variáveis categóricas (One-Hot Encoding)
encoder = OneHotEncoder(drop='first', sparse_output=False)
cat_vars = df_model[['Localizacao', 'Dependencia']]
cat_encoded = encoder.fit_transform(cat_vars)

# Juntar com variáveis numéricas
X_num = df_model[['Aprovacao_Total', 'Reprovacao_Total']].values
X = pd.DataFrame(
    data = np.hstack([X_num, cat_encoded]),
    columns = ['Aprovacao_Total', 'Reprovacao_Total'] + list(encoder.get_feature_names_out(['Localizacao', 'Dependencia']))
)

y = df_model['Abandono_Total'].values

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Avaliação
y_pred = modelo.predict(X_test)

# Calcula RMSE e R²
from math import sqrt
rmse = sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(" Regressão Linear para Abandono Escolar")
print("RMSE:", round(rmse, 2))
print("R²:", round(r2, 3))
print("Coeficientes:")
for nome, coef in zip(X.columns, modelo.coef_):
    print(f"  {nome}: {round(coef, 2)}")
