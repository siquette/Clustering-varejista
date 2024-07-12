## README para Análise de Clusters em Dados de Varejo

Este README detalha um script Python que realiza uma análise de clusters em um conjunto de dados de varejo. O script utiliza diversos métodos de clustering, gera visualizações para interpretação e realiza análises estatísticas descritivas e inferenciais. Abaixo está uma descrição detalhada de cada seção do código.

### Importando os Pacotes

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import scipy.stats as stats
from scipy.stats import zscore
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pingouin as pg
import plotly.express as px 
import plotly.io as pio
pio.renderers.default='browser'
```

Este bloco importa os pacotes necessários para a análise. Inclui bibliotecas para manipulação de dados (`pandas`, `numpy`), visualização (`matplotlib`, `seaborn`, `plotly`), clustering (`scipy`, `sklearn`), e análise estatística (`pingouin`).

### Importando o Banco de Dados

```python
varejista = pd.read_excel('regional_varejista.xlsx')
## Fonte: Fávero & Belfiore (2024, Capítulo 9)
```

Este bloco importa o conjunto de dados do arquivo Excel. O conjunto de dados contém informações sobre várias lojas de uma rede de varejo.

### Visualizando os Dados

```python
print(varejista.info())
print(varejista[['atendimento','sortimento', 'organização']].describe())
```

Este bloco exibe informações gerais sobre o conjunto de dados, incluindo o tipo de dados e estatísticas descritivas das variáveis de interesse (`atendimento`, `sortimento`, `organização`).

### Ajustando o Banco de Dados

```python
varejo = varejista.drop(columns=['loja','regional'])
```

Remove colunas não numéricas (`loja` e `regional`) para focar nas variáveis numéricas durante a análise de clusters.

### Cluster Hierárquico Aglomerativo

#### Método Single Linkage com Distância Cityblock

```python
plt.figure(figsize=(16,8))
dend_sing = sch.linkage(varejo, method = 'single', metric = 'cityblock')
dendrogram_s = sch.dendrogram(dend_sing, color_threshold = 60, labels = list(varejista.loja))
plt.title('Dendrograma', fontsize=16)
plt.xlabel('Lojas', fontsize=16)
plt.ylabel('Distância Cityblock (Manhattan)', fontsize=16)
plt.axhline(y = 60, color = 'red', linestyle = '--')
plt.show()
```

Gera um dendrograma usando o método de encadeamento `single` e a métrica de distância `cityblock`.

```python
cluster_sing = AgglomerativeClustering(n_clusters = 3, metric = 'cityblock', linkage = 'single')
indica_cluster_sing = cluster_sing.fit_predict(varejo)
varejista['cluster_single'] = indica_cluster_sing
varejista['cluster_single'] = varejista['cluster_single'].astype('category')
```

Cria uma nova coluna no dataframe `varejista` indicando a que cluster cada loja pertence, com base no método `single linkage` e métrica `cityblock`.

#### Método Complete Linkage com Distância Euclidiana

```python
plt.figure(figsize=(16,8))
dend_sing_euc = sch.linkage(varejo, method = 'complete', metric = 'euclidean')
dendrogram_euc = sch.dendrogram(dend_sing_euc, color_threshold = 55, labels = list(varejista.loja))
plt.title('Dendrograma', fontsize=16)
plt.xlabel('Lojas', fontsize=16)
plt.ylabel('Distância Euclideana', fontsize=16)
plt.axhline(y = 55, color = 'red', linestyle = '--')
plt.show()
```

Gera um dendrograma usando o método de encadeamento `complete` e a métrica de distância `euclidiana`.

```python
cluster_comp = AgglomerativeClustering(n_clusters = 3, metric = 'euclidean', linkage = 'complete')
indica_cluster_comp = cluster_comp.fit_predict(varejo)
varejista['cluster_complete'] = indica_cluster_comp
varejista['cluster_complete'] = varejista['cluster_complete'].astype('category')
```

Cria uma nova coluna no dataframe `varejista` indicando a que cluster cada loja pertence, com base no método `complete linkage` e métrica `euclidiana`.

### Visualização dos Clusters

```python
plt.figure(figsize=(10,10))
fig = sns.scatterplot(x='atendimento', y='sortimento', s=60, data=varejista, hue='cluster_single')
plt.title('Clusters', fontsize=16)
plt.xlabel('Atendimento', fontsize=16)
plt.ylabel('Sortimento', fontsize=16)
plt.show()
```

Plota um gráfico de dispersão mostrando os clusters identificados pelo método `single linkage` e métrica `cityblock`.

### Método K-Means

```python
kmeans_varejista = KMeans(n_clusters=3, init='random', random_state=100).fit(varejo)
kmeans_clusters = kmeans_varejista.labels_
varejista['cluster_kmeans'] = kmeans_clusters
varejista['cluster_kmeans'] = varejista['cluster_kmeans'].astype('category')
```

Aplica o algoritmo K-Means para identificar 3 clusters no conjunto de dados.

### Método da Silhueta

```python
silhueta = []
I = range(2,9)
for i in I: 
    kmeansSil = KMeans(n_clusters=i, init='random', random_state=100).fit(varejo)
    silhueta.append(silhouette_score(varejo, kmeansSil.labels_))

plt.figure(figsize=(16,8))
plt.plot(range(2, 9), silhueta, color = 'purple', marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.ylabel('Silhueta Média', fontsize=16)
plt.title('Método da Silhueta', fontsize=16)
plt.axvline(x = silhueta.index(max(silhueta))+2, linestyle = 'dotted', color = 'red') 
plt.show()
```

Determina o número ideal de clusters usando o método da silhueta, plotando a pontuação da silhueta para diferentes números de clusters.

### Coordenadas dos Centroides dos Clusters

```python
cent_finais = pd.DataFrame(kmeans_varejista.cluster_centers_)
cent_finais.columns = varejo.columns
cent_finais.index.name = 'cluster'
cent_finais
```

Exibe as coordenadas dos centroides dos clusters identificados pelo algoritmo K-Means.

### Visualização dos Clusters e seus Centroides

```python
plt.figure(figsize=(10,10))
sns.scatterplot(x='atendimento', y='sortimento', data=varejista, hue='cluster_kmeans', palette='viridis', s=100)
sns.scatterplot(x='atendimento', y='sortimento', data=cent_finais, s=40, c='red', label='Centroides', marker="X")
plt.title('Clusters e centroides', fontsize=16)
plt.xlabel('Atendimento', fontsize=16)
plt.ylabel('Sortimento', fontsize=16)
plt.legend()
plt.show()
```

Plota um gráfico de dispersão dos clusters e seus centroides.

### Estatística F das Variáveis

```python
pg.anova(dv='atendimento', between='cluster_kmeans', data=varejista, detailed=True).T
pg.anova(dv='sortimento', between='cluster_kmeans', data=varejista, detailed=True).T
pg.anova(dv='organização', between='cluster_kmeans', data=varejista, detailed=True).T
```

Realiza análises de variância (ANOVA) para verificar se há diferenças significativas nas variáveis `atendimento`, `sortimento` e `organização` entre os clusters identificados.

### Gráfico 3D dos Clusters

```python
fig = px.scatter_3d(varejista, x='atendimento', y='sortimento', z='organização', color='cluster_kmeans')
fig.show()
```

Cria uma visualização 3D dos clusters com as variáveis `atendimento`, `sortimento` e `organização`.

### Identificação das Características dos Clusters

```python
analise_varejista = varejista.drop(columns=['loja']).groupby(by=['cluster_kmeans'])
analise_varejista.describe().T
```

Agrupa o conjunto de dados pelos clusters identificados e exibe estatísticas descritivas para cada grupo.

### FIM

Este README fornece uma visão geral detalhada do script de análise de clusters, explicando cada etapa e os resultados esperados. Se você tiver alguma dúvida ou precisar de mais informações, sinta-se à vontade para entrar em contato.
