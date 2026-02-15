"""
Pacote principal do Simulador IAQ (Indoor Air Quality).

Este pacote contém os módulos essenciais para a simulação epidemiológica baseada em agentes
acoplada à dinâmica de fluidos computacional (ABM + CFD).

Módulos:
    - config: Definições de cenários e parâmetros científicos.
    - model: Orquestrador da simulação (IAQModel).
    - agents: Lógica comportamental e infecciosa dos ocupantes.
    - physics: Motor de transporte de aerossóis (Advecção-Difusão).
    - environment: Fachada para geometria e obstáculos.
"""

# Expõe as classes principais para acesso direto
from .config import (
    ScenarioConfig, 
    AgentsConfig, 
    PhysicsConfig,
    create_school_scenario,
    create_office_scenario,
    create_gym_scenario
)

from .model import IAQModel
from .agents import HumanAgent
from .physics import PhysicsEngine
from .environment import Environment

__all__ = [
    "IAQModel",
    "HumanAgent",
    "PhysicsEngine",
    "Environment",
    "ScenarioConfig",
    "create_school_scenario",
    "create_office_scenario", 
    "create_gym_scenario"
]

__version__ = "1.0.0"
__author__ = "Leon Lourenço da Silva Santos"