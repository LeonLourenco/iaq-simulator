"""
Simulador IAQ Avançado
======================

Sistema de simulação integrada para análise de qualidade do ar interno (IAQ)
combinando física CFD, agentes inteligentes e otimização multiobjetivo.

Módulos Principais:
------------------
- config_final: Configurações e constantes do sistema
- unified_physics: Motor físico unificado para simulação CFD
- advanced_agents: Sistema de agentes inteligentes
- main_model: Modelo principal de simulação
- final_dashboard: Interface interativa Streamlit
- run_simulation: Interface de linha de comando

Uso Básico:
----------
    >>> import config_final as cfg
    >>> from main_model import IAQSimulationModel
    >>> 
    >>> scenario = cfg.get_scenario_config('office')
    >>> model = IAQSimulationModel(scenario, cfg.PhysicsConfig())
    >>> 
    >>> while model.running:
    ...     model.step()
    >>> 
    >>> print(f"CO₂: {model.current_metrics['average_co2']:.0f} ppm")

Versão: 1.0.0
Licença: MIT
Autores: IAQ Simulator Team
"""

__version__ = '1.0.0'
__author__ = 'IAQ Simulator Team'
__license__ = 'MIT'

# Importações principais para facilitar uso
from . import config_final
from .main_model import IAQSimulationModel
from .unified_physics import UnifiedPhysicsEngine
from .advanced_agents import HumanAgent, LearningAgent

__all__ = [
    'config_final',
    'IAQSimulationModel',
    'UnifiedPhysicsEngine',
    'HumanAgent',
    'LearningAgent',
]
