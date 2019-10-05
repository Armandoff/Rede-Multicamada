# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 10:15:47 2019

@author: Armando
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

def sigmoidDerivada(sig):
    return sig * (1 - sig)

'''
def relu(soma):
    return soma*(soma>0)

def reluDerivada(relu):
    return 1*(relu>=0)
'''


#a = sigmoid(0.5)
#b = sigmoidDerivada(a)

#a = sigmoid(-1.5)
#b = np.exp(0)
    
dados_treinamento = np.genfromtxt('Xt.txt', skip_header=False)

'''
entradas = np.array([[0,0],
                     [0,1],
                     [1,0],
                     [1,1]])
    
saidas = np.array([[0],[1],[1],[0]])
'''
    
entradas = dados_treinamento[:, :10]

scaler.fit(entradas)

entradas = scaler.transform(entradas)

saidas = dados_treinamento[:, 10:]


#pesos0 = np.array([[-0.424, -0.740, -0.961],
#                   [0.358, -0.577, -0.469]])
    
#pesos1 = np.array([[-0.017], [-0.893], [0.148]])

pesos0 = np.random.random((10,3))
pesos1 = np.random.random((3,1))

epocas = 10000
taxaAprendizagem = 0.3
momento = 1

for j in range(epocas):
    camadaEntrada = entradas
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0)
    
    somaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)
    
    erroCamadaSaida = saidas - camadaSaida
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))
    print("Erro: " + str(mediaAbsoluta))
    
    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = erroCamadaSaida * derivadaSaida
    
    pesos1Transposta = pesos1.T
    deltaSaidaXPeso = deltaSaida.dot(pesos1Transposta)
    deltaCamadaOculta = deltaSaidaXPeso * sigmoidDerivada(camadaOculta)
    
    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaAprendizagem)
    
    camadaEntradaTransposta = camadaEntrada.T
    pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 = (pesos0 * momento) + (pesosNovo0 * taxaAprendizagem)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


