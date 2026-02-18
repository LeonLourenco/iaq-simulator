"""
IAQ Simulator - Core Package
============================

Este pacote contém o motor matemático e lógico da simulação híbrida (LBM + ABM).

Estrutura do Pacote:
--------------------
- simulation_engine: Orquestrador da simulação, gerencia o loop de tempo e a integração entre física e agentes.
- lbm_core: Núcleo numérico de alta performance para Dinâmica dos Fluidos Computacional (Método Lattice Boltzmann).
- agents: Modelagem dos indivíduos, contendo a lógica comportamental e o modelo epidemiológico (SEIR).

Como importar:
--------------
A partir da raiz do projeto:
    from src.simulation_engine import IAQSimulator

"""

# Metadados do Projeto
__version__ = '4.0.0'
__author__ = 'Leon Lourenço (UFRPE)'
__license__ = 'MIT'

# Lista de módulos exportados (para 'from src import *')
__all__ = ['simulation_engine', 'lbm_core', 'agents']