"""
Testes de integração do sistema completo
"""

import sys
import os
import json
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config_final as cfg
from main_model import IAQSimulationModel


class TestIntegration:
    """Testes de integração do sistema completo"""
    
    def setup_method(self):
        """Configuração antes de cada teste"""
        self.scenario = cfg.BuildingScenario(
            name="integration_test",
            building_type=cfg.BuildingType.OFFICE,
            total_width=12.0,
            total_height=10.0,
            floor_height=3.0,
            total_occupants=20,
            initial_infected_ratio=0.1,
            co2_setpoint=800,
            temperature_setpoint=22.0,
            humidity_setpoint=50.0,
            overall_ventilation_strategy="constant_volume",
            zones=[
                cfg.ZoneConfig(
                    name="Main Zone",
                    zone_type=cfg.ZoneType.WORKSPACE,
                    x=0, y=0, width=8, height=6,
                    occupancy_density=0.7,
                    target_ach=4.0,
                    materials=["carpet"]
                )
            ],
            agent_config=cfg.AgentConfig()
        )
        
        self.physics_config = cfg.PhysicsConfig(
            cell_size=0.3,
            kalman_enabled=False
        )
    
    def test_model_physics_integration(self):
        """Testa integração entre modelo e motor físico"""
        model = IAQSimulationModel(
            scenario=self.scenario,
            physics_config=self.physics_config,
            simulation_duration_hours=0.5,
            real_time_factor=10.0,
            use_learning_agents=False
        )
        
        assert model is not None
        assert model.physics is not None
        assert len(model.agents) > 0
        
        initial_time = model.time
        for _ in range(10):
            model.step()
        
        assert model.time > initial_time
    
    def test_intervention_effects(self):
        """Testa efeito das intervenções no sistema integrado"""
        model = IAQSimulationModel(
            scenario=self.scenario,
            physics_config=self.physics_config,
            simulation_duration_hours=0.5,
            real_time_factor=10.0,
            use_learning_agents=False
        )
        
        initial_mask_wearing = sum(1 for a in model.agents if a.mask_wearing)
        
        model.apply_interventions("mask_mandate", {"compliance": 0.8})
        
        mask_wearing_after = sum(1 for a in model.agents if a.mask_wearing)
        assert mask_wearing_after > initial_mask_wearing
    
    def test_data_export(self):
        """Testa exportação de dados"""
        model = IAQSimulationModel(
            scenario=self.scenario,
            physics_config=self.physics_config,
            simulation_duration_hours=0.1,
            real_time_factor=10.0,
            use_learning_agents=False
        )
        
        for _ in range(5):
            model.step()
        
        json_data = model.export_simulation_data('json')
        parsed_data = json.loads(json_data)
        
        assert 'metadata' in parsed_data
        assert 'metrics_final' in parsed_data
    
    def test_performance_metrics(self):
        """Testa métricas de desempenho"""
        import time
        
        model = IAQSimulationModel(
            scenario=self.scenario,
            physics_config=self.physics_config,
            simulation_duration_hours=0.25,
            real_time_factor=5.0,
            use_learning_agents=False
        )
        
        start_time = time.time()
        
        while model.running:
            model.step()
        
        execution_time = time.time() - start_time
        
        assert model.time > 0
        assert len(model.simulation_data['time']) > 0


if __name__ == "__main__":
    test = TestIntegration()
    test.setup_method()
    
    print("🧪 Executando testes de integração...")
    
    tests = [
        test.test_model_physics_integration,
        test.test_intervention_effects,
        test.test_data_export,
        test.test_performance_metrics
    ]
    
    for test_func in tests:
        try:
            test_func()
            print(f"  ✅ {test_func.__name__}")
        except Exception as e:
            print(f"  ❌ {test_func.__name__}: {str(e)}")
    
    print("\n✅ Testes de integração concluídos!")
