"""
Testes para agentes inteligentes
"""

import sys
import os
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config_final as cfg
from advanced_agents import HumanAgent, LearningAgent


class TestAgents:
    """Testes para os agentes do sistema"""
    
    def setup_method(self):
        """Configuração antes de cada teste"""
        class MockModel:
            def __init__(self):
                self.time = 0.0
                self.grid = type('obj', (object,), {
                    'width': 20,
                    'height': 16,
                    'is_cell_empty': lambda pos: True,
                    'place_agent': lambda agent, pos: None
                })()
                self.physics = type('obj', (object,), {
                    'cells_x': 20,
                    'cells_y': 16
                })()
                self.current_agent_emissions = []
            
            def add_agent_emissions(self, emissions):
                self.current_agent_emissions.extend(emissions)
        
        self.mock_model = MockModel()
        
        self.zone_config = cfg.ZoneConfig(
            name="Test Zone",
            zone_type=cfg.ZoneType.WORKSPACE,
            x=0, y=0, width=10, height=8,
            occupancy_density=0.5,
            target_ach=4.0,
            materials=["carpet"]
        )
        
        self.agent_config = cfg.AgentConfig(
            intelligence_level="adaptive",
            movement_pattern="random",
            mask_wearing_prob=0.3
        )
    
    def test_human_agent_creation(self):
        """Testa criação de agente humano básico"""
        agent = HumanAgent(
            unique_id=1,
            model=self.mock_model,
            zone_config=self.zone_config,
            agent_config=self.agent_config,
            initial_infected=False
        )
        
        assert agent.unique_id == 1
        assert not agent.infected
        assert hasattr(agent, 'emission_rates')
    
    def test_learning_agent_creation(self):
        """Testa criação de agente com aprendizado"""
        agent = LearningAgent(
            unique_id=2,
            model=self.mock_model,
            zone_config=self.zone_config,
            agent_config=self.agent_config,
            initial_infected=True
        )
        
        assert isinstance(agent, HumanAgent)
        assert agent.infected
        assert hasattr(agent, 'learning_enabled')
    
    def test_agent_step_method(self):
        """Testa execução do método step do agente"""
        agent = HumanAgent(
            unique_id=3,
            model=self.mock_model,
            zone_config=self.zone_config,
            agent_config=self.agent_config,
            initial_infected=False
        )
        
        agent.pos = (5, 5)
        self.mock_model.current_agent_emissions = []
        
        agent.step()
        
        assert len(self.mock_model.current_agent_emissions) > 0
    
    def test_agent_infection(self):
        """Testa lógica de infecção do agente"""
        agent = HumanAgent(
            unique_id=5,
            model=self.mock_model,
            zone_config=self.zone_config,
            agent_config=self.agent_config,
            initial_infected=False
        )
        
        assert not agent.infected
        
        infection_time = 3600.0
        agent.infect(infection_time)
        
        assert agent.infected
        assert agent.infection_start_time == infection_time
    
    def test_thermal_comfort_calculation(self):
        """Testa cálculo de conforto térmico"""
        agent = HumanAgent(
            unique_id=8,
            model=self.mock_model,
            zone_config=self.zone_config,
            agent_config=self.agent_config,
            initial_infected=False
        )
        
        comfort = agent.calculate_thermal_comfort(22.0, 50.0, 0.1)
        assert 0.0 <= comfort <= 1.0


if __name__ == "__main__":
    test = TestAgents()
    test.setup_method()
    
    print("🧪 Executando testes de agentes...")
    
    tests = [
        test.test_human_agent_creation,
        test.test_learning_agent_creation,
        test.test_agent_step_method,
        test.test_agent_infection,
        test.test_thermal_comfort_calculation
    ]
    
    for test_func in tests:
        try:
            test_func()
            print(f"  ✅ {test_func.__name__}")
        except Exception as e:
            print(f"  ❌ {test_func.__name__}: {str(e)}")
    
    print("\n✅ Testes de agentes concluídos!")
