import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Estilo dos gráficos
sns.set(style="whitegrid")

# Caminho do arquivo limpo
file_path = r'D:\Unialfa\2025.1\trabalhomachine\data\processed\tx_rend_limpo.csv'

# Carrega os dados
df = pd.read_csv(file_path)

# Remove registros com UF ou valores nulos
df = df[df['UF'].notna()]

# ========================
# 1. MÉDIA DE APROVAÇÃO POR UF
# ========================
media_aprov_uf = df.groupby('UF')['Aprovacao_Total'].mean().sort_values(ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(x=media_aprov_uf.index, y=media_aprov_uf.values, palette="viridis")
plt.title('Média de Aprovação por UF')
plt.ylabel('% de Aprovação')
plt.xlabel('UF')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ========================
# 2. TOP 10 MUNICÍPIOS COM MAIOR ABANDONO
# ========================
top_abandono = df[['Municipio', 'UF', 'Abandono_Total']].dropna()
top_abandono = top_abandono.sort_values(by='Abandono_Total', ascending=False).head(10)

plt.figure(figsize=(12,6))
sns.barplot(data=top_abandono, x='Abandono_Total', y='Municipio', hue='UF', dodge=False, palette="Reds_r")
plt.title('Top 10 Municípios com Maior Taxa de Abandono')
plt.xlabel('% de Abandono')
plt.ylabel('Município')
plt.tight_layout()
plt.show()

# ========================
# 3. COMPARAÇÃO URBANO x RURAL (APROVAÇÃO)
# ========================
locais = df[df['Localizacao'].isin(['Urbana', 'Rural'])]
media_local = locais.groupby('Localizacao')['Aprovacao_Total'].mean()

plt.figure(figsize=(6,5))
sns.barplot(x=media_local.index, y=media_local.values, palette='pastel')
plt.title('Média de Aprovação: Urbano x Rural')
plt.ylabel('% de Aprovação')
plt.xlabel('Localização')
plt.tight_layout()
plt.show()
