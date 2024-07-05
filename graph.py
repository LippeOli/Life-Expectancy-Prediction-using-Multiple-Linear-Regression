#Felipe Oliveira Carvalho   -14613879

import numpy as np
import matplotlib.pyplot as plt

# Dados dos países do conjunto de teste
paises = ['República Checa', 'Portugal', 'Iraque', 'Nova Zelândia', 'Cuba']
valores_reais = np.array([79.31, 81.32, 70.27, 82.42, 79.18])
previsoes = np.array([77.86, 77.84, 74.59, 80.50, 75.87])

# Criando um gráfico de barras para comparar previsões e valores reais
fig, ax = plt.subplots(figsize=(10, 6))

# Definindo a largura das barras
bar_width = 0.4

# Definindo a posição das barras no eixo x
index = np.arange(len(paises))

# Plotando as barras de valores reais e previsões
rects1 = ax.bar(index - bar_width/2, valores_reais, bar_width, label='Valor Real', color='orange')
rects2 = ax.bar(index + bar_width/2, previsoes, bar_width, label='Previsão', color='blue')

# Adicionando rótulos, título e legenda
ax.set_xlabel('Países')
ax.set_ylabel('Expectativa de Vida')
ax.set_title('Comparação entre Previsões e Valores Reais da Expectativa de Vida por País')
ax.set_xticks(index)
ax.set_xticklabels(paises, rotation=45, ha='right')
ax.legend()

# Função para adicionar rótulos com os valores numéricos acima das barras
def autolabel(rects):
    """Função para adicionar rótulos de valor nas barras."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # Deslocamento vertical
                    textcoords="offset points",
                    ha='center', va='bottom')

# Chamando a função para adicionar os valores nas barras
autolabel(rects1)
autolabel(rects2)

# Exibindo o gráfico
plt.tight_layout()
plt.show()
