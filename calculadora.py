import numpy as np
from typing import Callable
import sys

class Calculadora:

    @staticmethod
    def calcular_falsa_posicao(a_primeiro: float, b_primeiro: float, f: Callable[[float], float], num_max_iter: int, epsilon: float) -> float:
        num_iter_atual = 0
        a_atual = a_primeiro
        b_atual = b_primeiro
        
        fa = f(a_atual)
        fb = f(b_atual)

        if fa * fb >= 0:
            raise ValueError("A função não tem sinais opostos nos extremos do intervalo [a, b]")

        # Inicializa x_velho com um valor (ex: o início do intervalo)
        x_velho = a_primeiro
        
        # Inicializa 'teste' com um valor maior que epsilon para garantir a entrada no loop
        teste = abs(b_atual - a_atual) 

        while num_max_iter > num_iter_atual and teste > epsilon:
            # Verifica se o denominador é muito próximo de zero
            if abs(fb - fa) < sys.float_info.epsilon:
                break  # Evita divisão por zero se fa e fb forem iguais

            x_novo = (a_atual * fb - b_atual * fa) / (fb - fa)
            fx = f(x_novo)

            # Usa a diferença absoluta, que não falha se x_novo for 0.
            teste = abs(x_novo - x_velho)
            
            # Condição alternativa de parada: se a função estiver próxima de zero
            # if abs(fx) < epsilon:
            #     return x_novo

            x_velho = x_novo

            # Atualiza o intervalo
            if fa * fx > 0:
                a_atual = x_novo
                fa = fx
            else:
                b_atual = x_novo
                fb = fx
            
            num_iter_atual += 1

        return x_velho

    @staticmethod
    def derivada(f: Callable[[float], float]) -> Callable[[float], float]:
        h = 1e-6  # Passo pequeno para a diferença central
        return lambda x: (f(x + h) - f(x - h)) / (2 * h)

    @staticmethod
    def calcular_newton_raphson(f: Callable[[float], float], x0: float, num_max_iter: int, epsilon: float) -> float:
        num_iter_atual = 0
        xi = x0
        
        # Inicializa 'teste' com um valor maior que epsilon
        teste = 1 + epsilon 
        
        # Obtém a função derivada (chamando como método estático)
        f_linha = Calculos.derivada(f)

        while num_max_iter > num_iter_atual and teste > epsilon:
            fx = f(xi)
            f_linha_x = f_linha(xi)

            # Previne divisão por zero (tangente horizontal)
            if abs(f_linha_x) < sys.float_info.epsilon:
                raise ValueError(f"Derivada próxima de zero em x = {xi}. O método de Newton-Raphson falhou.")

            xi_mais1 = xi - (fx / f_linha_x)

            teste = abs(xi_mais1 - xi)

            xi = xi_mais1
            num_iter_atual += 1

        return xi
    
    @staticmethod
    def eliminacao_gauss(matriz):

    matriz = np.array(matriz)
    matriz = matriz.astype(float)

    tolerancia = 1e-15
    n = len(matriz)

    # Fase 1: Eliminação 
    for i in range(n - 1):


        pivot = matriz[i][i]
        linhaPivotPosicao = i
        
        maiorNumeroDaColuna = abs(pivot)


        # Pivoteamento parcial
        for j in range(i+1, n):
            if abs(matriz[j][i]) > maiorNumeroDaColuna:
                maiorNumeroDaColuna = abs(matriz[j][i])
                linhaPivotPosicao = j

        matriz[[i, linhaPivotPosicao]] = matriz[[linhaPivotPosicao, i]]
        pivot = matriz[i][i]

        if abs(pivot) < tolerancia:
            raise ValueError("A matriz é singular. O sistema pode não ter solução única.")

        for j in range(i+1, n):
            if matriz[j][i] != 0:
                m = matriz[j][i] / pivot
                matriz[j] = matriz[j] - matriz[i] * m
        

    # Fase 2: Retrosubstituição
    
    resultados = []
    
    for i in range(n - 1, -1, -1):
        
        for j in range(i + 1, n):
            
            indice_x_j = (n - 1) - j
            matriz[i][-1] -= matriz[i][j] * resultados[indice_x_j]
            
        matriz[i][-1] /= matriz[i][i]
        resultados.append(float(matriz[i][-1]))

    resultados.reverse()
    
    return resultados
        


        