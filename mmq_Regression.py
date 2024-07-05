#Felipe Oliveira Carvalho   -14613879

import numpy as np

# Dados para analise 
data = {
    'País': ['Estados Unidos', 'China', 'Japão', 'Alemanha', 'Índia', 'Brasil', 'Reino Unido', 'França', 'Itália', 'Canadá',
             'Rússia', 'Austrália', 'Coreia do Sul', 'Espanha', 'México', 'Indonésia', 'Holanda', 'Arábia Saudita', 'Turquia',
             'Suíça', 'Argentina', 'Suécia', 'Polônia', 'Bélgica', 'Noruega', 'Irã', 'Áustria', 'Tailândia', 'Nigéria',
             'Emirados Árabes', 'Egito', 'Filipinas', 'Singapura', 'Vietnã', 'Malásia', 'Chile', 'Paquistão', 'Colômbia',
             'Peru', 'Ucrânia', 'Bangladesh', 'Romênia', 'Hungria', 'Grécia', 'República Checa', 'Portugal', 'Iraque',
             'Nova Zelândia', 'Cuba'],
    'Renda per Capita (US$)': [65118, 10262, 40847, 46259, 2338, 8717, 41059, 40493, 34320, 46233, 11585, 53825, 31846, 29715,
                               10118, 4135, 52331, 23139, 8958, 82839, 10549, 52477, 15450, 46482, 81694, 6463, 51641, 7808,
                               2149, 43470, 3019, 3485, 65233, 3537, 10240, 15010, 1543, 6498, 6978, 3727, 1998, 12896, 17846,
                               19418, 37926, 23566, 4774, 41243, 8707],
    'PIB (US$ bilhões)': [21137, 14140, 5156, 3846, 2875, 2056, 2829, 2716, 2001, 1736, 1641, 1381, 1658, 1403, 1195, 1042,
                          907, 778, 760, 705, 449, 552, 586, 529, 403, 454, 458, 456, 448, 421, 332, 304, 364, 262, 315, 268,
                          231, 323, 232, 130, 249, 248, 153, 142, 245, 265, 224, 122, 103, 97],
    'População (milhões)': [331, 1441, 126, 83, 1380, 213, 68, 67, 60, 38, 146, 26, 51, 47, 128, 273, 17, 34, 84, 8, 45, 10,
                            38, 11, 5, 84, 9, 70, 206, 10, 100, 109, 6, 96, 32, 19, 229, 50, 32, 43, 165, 19, 10, 11, 10, 11,
                            10, 41, 5, 11],
    'Expectativa de Vida': [78.9, 76.7, 84.5, 81.2, 69.7, 75.9, 81.3, 82.5, 83.2, 82.3, 72.7, 82.9, 83.3, 83.0, 75.0, 71.8,
                            82.1, 75.3, 78.6, 83.4, 76.5, 82.3, 77.5, 81.6, 82.4, 76.2, 81.4, 78.6, 54.5, 77.8, 70.5, 71.2,
                            83.6, 75.3, 76.9, 80.2, 67.0, 76.7, 76.4, 72.3, 72.6, 75.4, 75.8, 81.4, 79.7, 81.1, 70.8, 82.3, 79.7]
}

# Definindo os dados de treinamento e teste
X_train = np.array([
    [1, 65118, 21433, 331.0],
    [1, 10262, 14342, 1402.0],
    [1, 40847, 5082, 126.3],
    [1, 46259, 3861, 83.2],
    [1, 2338, 2869, 1366.0],
    [1, 8717, 2056, 212.6],
    [1, 41059, 2828, 67.9],
    [1, 40493, 2778, 65.3],
    [1, 3432, 2002, 60.4],
    [1, 46233, 1736, 37.6],
    [1, 11585, 1699, 144.1],
    [1, 53825, 1397, 25.5],
    [1, 31846, 1629, 51.7],
    [1, 29715, 1393, 46.7],
    [1, 10118, 1269, 128.9],
    [1, 4135, 1042, 273.5],
    [1, 52331, 912, 17.4],
    [1, 23139, 779, 34.8],
    [1, 8958, 761, 84.3],
    [1, 82839, 746, 8.6],
    [1, 10549, 449, 45.4],
    [1, 52477, 541, 10.4],
    [1, 1545, 596, 38.3],
    [1, 46482, 533, 11.5],
    [1, 81694, 434, 5.4],
    [1, 60, 463, 83.0],
    [1, 51641, 449, 8.9],
    [1, 7808, 514, 69.8],
    [1, 2149, 432, 206.1],
    [1, 4347, 421, 9.9],
    [1, 3019, 363, 102.3],
    [1, 3485, 362, 108.1],
    [1, 65233, 372, 5.7],
    [1, 3537, 340, 97.3],
    [1, 1024, 336, 32.4],
    [1, 1501, 282, 19.1],
    [1, 1543, 284, 220.9],
    [1, 6498, 336, 50.9],
    [1, 6978, 229, 32.9],
    [1, 3727, 155, 41.0],
    [1, 1998, 324, 164.7],
    [1, 12896, 244, 19.1],
    [1, 17846, 180, 9.6],
    [1, 19193, 209, 10.4]
])

#Array para dados de treinamento Y_train

y_train = np.array([79.11, 76.91, 84.67, 81.33, 69.66, 75.88, 81.27, 82.57, 83.24, 82.30, 72.58, 82.89, 82.59, 83.51,
                    75.13, 71.72, 82.17, 75.13, 77.93, 83.93, 76.28, 82.78, 78.47, 81.69, 82.94, 76.02, 81.74, 77.15,
                    54.33, 77.97, 72.54, 71.16, 83.45, 75.28, 75.04, 80.10, 67.27, 77.29, 76.02, 71.76, 72.32, 75.45,
                    76.75, 81.04])

#Array para os testes, em X_testes
X_testes = np.array([
    [1, 23496, 251, 10.7],
    [1, 23333, 237, 10.2],
    [1, 496, 224, 40.2],
    [1, 42634, 212, 4.9],
    [1, 9099, 100, 11.3]
])

#Y_testes 
y_test = np.array([79.31, 81.32, 70.27, 82.42, 79.18])


#Desenvolver os calculos das matrizes para o coef. da regressao

print("\n\n============================================================")
print("\nCalculo do Coeficiente...\n")
# Cálculo da matriz transposta de X_train
X_train_T = X_train.T

# Cálculo da matriz produto X_train^T * X_train
XTX = X_train_T @ X_train

# Cálculo da inversa de X_train^T * X_train
XTX_inv = np.linalg.inv(XTX)

# Cálculo dos coeficientes de regressão (beta)
beta = XTX_inv @ X_train_T @ y_train

# Imprimindo os coeficientes de regressão (beta)
print(f"\nCoeficientes de regressão (beta):\nIntercepto: {beta[0]:.8f}\nRenda per Capita: {
      beta[1]:.8f}\nPIB: {beta[2]:.8f}\nPopulação: {beta[3]:.8f}")



#Desencvolvimento das previsoes com base nos treinamentos e testes feitos 
#Apos obter o coef. da regressao utilizar os arrays para testar em cada variaves
#dos paises escolhidos


# Previsões para os dados de treinamento
y_train_pred = X_train @ beta

# Previsões para os dados de teste
y_test_pred = X_testes @ beta

print("\n\n============================================================")
print("\nPrevisões do Modelo....\n")

# Imprimindo previsões para os dados de treinamento e teste
print("\n\nPrevisões -> dados de treinamento:\n", y_train_pred)
print("\nPrevisões -> dados de teste:\n")
print("\nAqui estão os dados base para os testes, desse modo foi possível analisar as previsões\n")
print("\n",y_test_pred)

# Imprimindo Previsões vs Valores Reais para o conjunto de testes
print("\nPrevisões vs Valores Reais para o conjunto de testes:")
for i in range(len(y_test)):
    print(f"País: {data['País'][44 + i]}, Previsão: {y_test_pred[i]:.2f}, Valor Real: {y_test[i]}")


#Desenvolver calculo do Erro (MSE) com o array Y do treinamento

# Cálculo do Erro Quadrático Médio (MSE) para os dados de treinamento
MSE_train = np.mean((y_train - y_train_pred)**2)

# Cálculo do Erro Quadrático Médio (MSE) para os dados de teste
MSE_test = np.mean((y_test - y_test_pred)**2)

# Cálculo do Coeficiente de Determinação (R²) para os dados de treinamento
R2_train = 1 - (np.sum((y_train - y_train_pred)**2) /
                np.sum((y_train - np.mean(y_train))**2))

# Cálculo do Coeficiente de Determinação (R²) para os dados de teste
R2_test = 1 - (np.sum((y_test - y_test_pred)**2) /
               np.sum((y_test - np.mean(y_test))**2))


print("\n\n============================================================")
print("\nErros de Métrica...\n")


# Imprimindo as métricas do modelo
print("\nErro Quadrático Médio (MSE) -> dados de treinamento:", MSE_train)
print("\nErro Quadrático Médio (MSE) -> dados de teste:", MSE_test)


print("\n\n============================================================")
print("\nCoeficiente de Determinação...\n")

print("\nCoeficiente de Determinação (R²) -> dados de treinamento:", R2_train)
print("\nCoeficiente de Determinação (R²) -> dados de teste:", R2_test)
