"""
Testes para o motor físico unificado
"""

import sys
import os
import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config_final as cfg
from unified_physics import UnifiedPhysicsEngine


class TestUnifiedPhysics:
    """Testes para UnifiedPhysicsEngine"""
    
    def setup_method(self):
        """Configuração antes de cada teste"""
        self.scenario = cfg.BuildingScenario(
            name="test_scenario",
            building_type=cfg.BuildingType.OFFICE,
            total_width=10.0,
            total_height=8.0,
            floor_height=3.0,
            total_occupants=10,
            initial_infected_ratio=0.1,
            co2_setpoint=800,
            temperature_setpoint=22.0,
            humidity_setpoint=50.0,
            overall_ventilation_strategy="constant_volume",
            zones=[
                cfg.ZoneConfig(
                    name="Test Zone",
                    zone_type=cfg.ZoneType.WORKSPACE,
                    x=0, y=0, width=10, height=8,
                    occupancy_density=0.5,
                    target_ach=4.0,
                    materials=["carpet"]
                )
            ],
            agent_config=cfg.AgentConfig()
        )
        
        self.physics_config = cfg.PhysicsConfig(
            cell_size=0.5,
            kalman_enabled=False,
            pem_correction_active=False
        )
        
        self.physics = UnifiedPhysicsEngine(self.scenario, self.physics_config)
    
    def test_initialization(self):
        """Testa inicialização do motor físico"""
        assert self.physics is not None
        assert self.physics.cells_x == 20
        assert self.physics.cells_y == 16
        assert 'co2_ppm' in self.physics.grids
        assert 'temperature_c' in self.physics.grids
    
    def test_zone_map_creation(self):
        """Testa criação do mapa de zonas"""
        zone_map = self.physics.zone_map
        assert zone_map is not None
        assert zone_map.shape == (self.physics.cells_y, self.physics.cells_x)
    
    def test_diffusion(self):
        """Testa cálculo de difusão"""
        test_grid = np.zeros((self.physics.cells_y, self.physics.cells_x))
        center_x = self.physics.cells_x // 2
        center_y = self.physics.cells_y // 2
        test_grid[center_y, center_x] = 1000.0
        
        diffused = self.physics._apply_diffusion(test_grid, 0.1, 0.1)
        
        assert diffused[center_y, center_x] < 1000.0
        assert abs(test_grid.sum() - diffused.sum()) / test_grid.sum() < 1e-6
    
    def test_step_method(self):
        """Testa o método step completo"""
        agent_data = {
            'emissions': [{
                'x': 5, 'y': 5,
                'co2_ppm_per_s': 0.05,
                'heat_w': 75.0
            }],
            'positions': [(5, 5)],
            'activities': ['seated']
        }
        
        initial_co2 = self.physics.grids['co2_ppm'].copy()
        self.physics.step(self.physics.dt, 0.0, agent_data)
        
        assert not np.allclose(initial_co2, self.physics.grids['co2_ppm'])
    
    def test_get_zone_statistics(self):
        """Testa extração de estatísticas por zona"""
        stats = self.physics.get_zone_statistics()
        
        assert isinstance(stats, dict)
        assert len(stats) == 1
        assert 'concentrations' in stats[1]


if __name__ == "__main__":
    test = TestUnifiedPhysics()
    test.setup_method()
    
    print("🧪 Executando testes do motor físico...")
    
    tests = [
        test.test_initialization,
        test.test_zone_map_creation,
        test.test_diffusion,
        test.test_step_method,
        test.test_get_zone_statistics
    ]
    
    for test_func in tests:
        try:
            test_func()
            print(f"  ✅ {test_func.__name__}")
        except AssertionError as e:
            print(f"  ❌ {test_func.__name__}: {str(e)}")
        except Exception as e:
            print(f"  ⚠️  {test_func.__name__}: Erro - {str(e)}")
    
    print("\n✅ Testes concluídos!")
