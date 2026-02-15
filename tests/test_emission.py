"""
Testes Unitários de Emissão Viral.

Objetivo:
    Validar se o cálculo de emissão de quanta por segundo (q/s) respeita:
    1. A taxa base definida pela atividade (ActivityLevel).
    2. A carga viral do agente.
    3. A eficiência da máscara (se utilizada).

Contexto:
    Garante que não existem valores "mágicos" ou hardcoded ignorando 
    a configuração do cenário, corrigindo bugs legados.
"""

import pytest
import numpy as np
from src.config import (
    AgentsConfig, 
    ActivityLevel, 
    AgentState, 
    EmissionRates
)
from src.agents import HumanAgent

# ============================================================================
# MOCKS E FIXTURES
# ============================================================================

class MockModel:
    """Mock simples para isolar o agente de dependências do Grid/Physics."""
    def __init__(self):
        self.time = 0.0
        # O agente pode tentar acessar grid/environment em step(), 
        # mas não no __init__ ou calculate_emission.
        self.grid = None
        self.environment = None

@pytest.fixture
def mock_model():
    return MockModel()

def create_infected_agent(model, activity_level, mask_eff=0.0, wears_mask=False):
    """Helper para criar um agente infectado com carga viral máxima."""
    
    # Configuração dummy
    config = AgentsConfig(
        total_occupants=1,
        initial_infected=1,
        activity_level=activity_level,
        mask_compliance=1.0 if wears_mask else 0.0,
        mask_efficiency=mask_eff
    )
    
    # Cria agente na posição (0,0)
    agent = HumanAgent(
        unique_id=1,
        model=model,
        pos=(0, 0),
        agent_config=config,
        initial_state=AgentState.INFECTED
    )
    
    # Força parâmetros para teste determinístico
    agent.viral_load = 1.0  # Pico da infecção
    agent.wears_mask = wears_mask
    agent.mask_efficiency = mask_eff
    
    return agent

# ============================================================================
# TESTES DE ESCALA E LINEARIDADE
# ============================================================================

def test_emission_linear_scaling(mock_model):
    """
    Verifica se a emissão final escala linearmente com a taxa base.
    
    Como as taxas são constantes (Enums) no código, injetamos valores manuais
    no atributo 'emission_rate_base' do agente para simular configurações 
    arbitrárias e verificar a fórmula matemática.
    """
    # Cria um agente base
    agent = create_infected_agent(mock_model, ActivityLevel.SEDENTARY)
    
    target_emissions = [10.0, 50.0, 100.0, 200.0, 500.0]
    
    for target in target_emissions:
        # Injeção de dependência manual para teste de fórmula
        agent.emission_rate_base = target
        
        # Calcula emissão por segundo e converte de volta para hora
        calculated_q_s = agent.calculate_emission_quanta_per_s()
        calculated_q_h = calculated_q_s * 3600.0
        
        print(f"Target: {target} | Calculated: {calculated_q_h}")
        
        # Assert com tolerância de ponto flutuante
        assert abs(calculated_q_h - target) < 1e-5, \
            f"Erro de escala linear. Esperado {target}, obteve {calculated_q_h}"

def test_emission_viral_load_scaling(mock_model):
    """
    Verifica se a emissão é proporcional à carga viral (evolução temporal).
    """
    agent = create_infected_agent(mock_model, ActivityLevel.SEDENTARY)
    base_rate = EmissionRates.SEATED_QUIET # 10.0
    
    # Testa cargas virais variadas: 0%, 50%, 100%
    test_loads = [0.0, 0.5, 1.0]
    
    for load in test_loads:
        agent.viral_load = load
        
        expected = base_rate * load
        calculated = agent.calculate_emission_quanta_per_s() * 3600.0
        
        assert abs(calculated - expected) < 1e-5, \
            f"Erro na escala de carga viral {load}. Esperado {expected}, obteve {calculated}"

# ============================================================================
# TESTES DE ATIVIDADE (PRESETS)
# ============================================================================

def test_emission_activity_multipliers(mock_model):
    """
    Verifica se o agente seleciona a taxa correta baseada no Enum de Atividade.
    Valida o mapeamento entre ActivityLevel e EmissionRates.
    """
    # Mapeamento esperado (Baseado em src/config.py e src/agents.py)
    scenarios = [
        (ActivityLevel.SEDENTARY, EmissionRates.SEATED_QUIET), # 10.0
        (ActivityLevel.LIGHT, EmissionRates.TALKING),          # 50.0 (Aula/Conversa)
        (ActivityLevel.MODERATE, EmissionRates.EXERCISE_LIGHT),# 30.0
        (ActivityLevel.HEAVY, EmissionRates.EXERCISE_HEAVY)    # 100.0
    ]
    
    for activity, expected_rate in scenarios:
        agent = create_infected_agent(mock_model, activity)
        
        # Obtém emissão horária
        actual_rate = agent.calculate_emission_quanta_per_s() * 3600.0
        
        assert abs(actual_rate - expected_rate) < 0.1, \
            f"Falha no mapeamento de atividade {activity}. Esperado {expected_rate}, obteve {actual_rate}"

# ============================================================================
# TESTES DE INTERVENÇÃO (MÁSCARAS)
# ============================================================================

def test_emission_mask_reduction(mock_model):
    """
    Verifica se a máscara reduz a emissão na fonte (Source Control).
    Fórmula esperada: Emissão = Base * (1 - Eficiência)
    """
    base_activity = ActivityLevel.HEAVY # 100 q/h
    base_value = EmissionRates.EXERCISE_HEAVY
    
    efficiencies = [0.0, 0.3, 0.5, 0.95] # Sem máscara, Pano, Cirúrgica, N95
    
    for eff in efficiencies:
        # Cria agente forçando o uso de máscara
        agent = create_infected_agent(
            mock_model, 
            base_activity, 
            mask_eff=eff, 
            wears_mask=True
        )
        
        expected_emission = base_value * (1.0 - eff)
        actual_emission = agent.calculate_emission_quanta_per_s() * 3600.0
        
        print(f"Eff: {eff} | Base: {base_value} | Expected: {expected_emission} | Actual: {actual_emission}")
        
        assert abs(actual_emission - expected_emission) < 1e-5, \
            f"Erro no cálculo de máscara (Eff: {eff}). Esperado {expected_emission}, obteve {actual_emission}"

def test_no_emission_if_susceptible(mock_model):
    """Garante que agentes não infectados emitem 0 quanta."""
    config = AgentsConfig(
        total_occupants=1,
        initial_infected=0,
        activity_level=ActivityLevel.HEAVY,
        mask_compliance=0.0,
        mask_efficiency=0.0
    )
    
    agent = HumanAgent(
        unique_id=1, 
        model=mock_model, 
        pos=(0,0), 
        agent_config=config, 
        initial_state=AgentState.SUSCEPTIBLE
    )
    
    assert agent.calculate_emission_quanta_per_s() == 0.0, \
        "Agente SUSCETÍVEL não deve emitir vírus."

if __name__ == "__main__":
    pytest.main(["-v", __file__])