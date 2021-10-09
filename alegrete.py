import numpy as np
import math


def compute_mse(theta_0, theta_1, data):
    """
    Calcula o erro quadratico medio
    :param theta_0: float - intercepto da reta
    :param theta_1: float - inclinacao da reta
    :param data: np.array - matriz com o conjunto de dados, x na coluna 0 e y na coluna 1
    :return: float - o erro quadratico medio
    """
    # TODO: theta_1 é o ângulo em radianos da reta com o eixo X?

    # x0 = 0
    # y0 = theta_0
    # y = m * x + b
    # => b = y - (m * x)
    # => b = y0 - (m * x0)
    #                x0 = 0 
    # => b = y0 

    m = theta_1
    b = theta_0

    errors = []
    for point in data:

        x = point[0]
        y = point[1]

        regressed_y = b+ m * x 
        error = regressed_y - y

        squared_error = error ** 2

        errors.append(squared_error)

    mean_squared_errors = sum(errors) / len(errors)

    return mean_squared_errors


def getTheta1Derivative(theta_0, theta_1, data):
    totalSum = 0
    constant = 2 / (len(data))

    for point in data:
        functionValue = (theta_0 + theta_1 * point[0]) - point[1]
        totalSum += functionValue

    return constant * totalSum

def getTheta2Derivative(theta_0, theta_1, data):
    totalSum = 0
    constant = 2 / (len(data))

    for point in data:
        functionValue = (theta_0 + theta_1 * point[0]) - point[1]
        totalSum += functionValue * point[0]

    return constant * totalSum

def step_gradient(theta_0, theta_1, data, alpha):
    """
    Executa uma atualização por descida do gradiente  e retorna os valores atualizados de theta_0 e theta_1.
    :param theta_0: float - intercepto da reta
    :param theta_1: float -inclinacao da reta
    :param data: np.array - matriz com o conjunto de dados, x na coluna 0 e y na coluna 1
    :param alpha: float - taxa de aprendizado (a.k.a. tamanho do passo)
    :return: float,float - os novos valores de theta_0 e theta_1, respectivamente
    """

    newTheta1 = theta_0 - alpha * (getTheta1Derivative(theta_0, theta_1, data))
    newTheta2 = theta_1 - alpha * (getTheta2Derivative(theta_0, theta_1, data))

    return newTheta1, newTheta2


def fit(data, theta_0, theta_1, alpha, num_iterations):
    """
    Para cada época/iteração, executa uma atualização por descida de
    gradiente e registra os valores atualizados de theta_0 e theta_1.
    Ao final, retorna duas listas, uma com os theta_0 e outra com os theta_1
    obtidos ao longo da execução (o último valor das listas deve
    corresponder à última época/iteração).

    :param data: np.array - matriz com o conjunto de dados, x na coluna 0 e y na coluna 1
    :param theta_0: float - intercepto da reta
    :param theta_1: float -inclinacao da reta
    :param alpha: float - taxa de aprendizado (a.k.a. tamanho do passo)
    :param num_iterations: int - numero de épocas/iterações para executar a descida de gradiente
    :return: list,list - uma lista com os theta_0 e outra com os theta_1 obtidos ao longo da execução
    """

    newTheta0List = []
    newTheta1List = []

    theta1ToUse = theta_0
    theta2ToUse = theta_1

    for iteration in range(num_iterations):
        newTheta0, newTheta1 = step_gradient(theta1ToUse, theta2ToUse, data, alpha)

        newTheta0List.append(newTheta0)
        newTheta1List.append(newTheta1)

        theta1ToUse = newTheta0
        theta2ToUse = newTheta1

    return newTheta0List, newTheta1List
    
if __name__ == '__main__':
    data = np.genfromtxt('alegrete.csv', delimiter=',')

    
