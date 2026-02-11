"""
MOTOR FÍSICO
Integra toda a física do modelo e define obstáculos (paredes/móveis) e POIs.
"""

import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d
from typing import Dict, List, Tuple, Optional, Any
import config_final as cfg

class UnifiedPhysicsEngine:
    """
    Motor físico que integra todos os componentes:
    - Difusão molecular e turbulenta
    - Advecção por ventilação
    - Deposição em superfícies
    - Pluma térmica humana
    - Química de materiais
    - Transferência multizona
    - Troca de calor e umidade
    - Gestão de obstáculos e Pontos de Interesse (POIs)
    """
    
    def __init__(self, scenario: cfg.BuildingScenario, physics_config: cfg.PhysicsConfig):
        self.scenario = scenario
        self.config = physics_config
        
        # Condições externas (padrão)
        self.external_temperature = 20.0 + 273.15  # K
        self.external_humidity = 0.6  # 60%
        self.external_co2 = 400 * cfg.CONVERSION_FACTORS['co2_ppm_to_kgm3']

        # Calcula dimensões do grid
        self.cells_x = int(scenario.total_width / physics_config.cell_size)
        self.cells_y = int(scenario.total_height / physics_config.cell_size)
        
        # Inicializa grids com epsilon
        self.grids = self._initialize_grids()
        
        # Mapa de zonas
        self.zone_map = self._create_zone_map()
        
        # Inicializa dicionário de POIs
        self.pois = {} # Ex: {'workstations': [(x,y), ...], 'equipment': [(x,y), ...]}
        
        # Mapa de Obstáculos e POIs baseados no cenário
        self.obstacle_grid = self._create_obstacle_grid()
        
        # Mapa de materiais
        self.material_grid = self._create_material_grid()
        
        # Campo de velocidade
        self.velocity_field = np.zeros((self.cells_y, self.cells_x, 2))
        self._initialize_velocity_field()
        
        # Parâmetros de difusão efetiva
        self.diffusion_coeffs = self._calculate_diffusion_coeffs()
        
        # Inicializa fontes e sumidouros
        self._initialize_sources_sinks()
        
        # Ganhos de calor por zona
        self.heat_gains = self._calculate_heat_gains()
        
        # Filtro de Kalman (opcional)
        if physics_config.kalman_enabled:
            self.kalman_filters = self._initialize_kalman_filters()
        else:
            self.kalman_filters = {}
        
        # Parâmetros para transferência de calor
        self.U_walls = 2.0
        self.U_windows = 3.0
        
        # Histórico para análise
        self.history = {
            'time': [],
            'zone_concentrations': [],
            'energy_consumption': []
        }
    
    def _initialize_grids(self) -> Dict[str, np.ndarray]:
        shape = (self.cells_y, self.cells_x)
        return {
            'co2': np.ones(shape) * self.external_co2,
            'hcho': np.ones(shape) * 10 * cfg.CONVERSION_FACTORS['hcho_ppb_to_kgm3'],
            'voc': np.ones(shape) * 50 * cfg.CONVERSION_FACTORS['voc_ppb_to_kgm3'],
            'virus': np.ones(shape) * 1e-12,
            'pm25': np.ones(shape) * 5e-9,
            'pm10': np.ones(shape) * 10e-9,
            'temperature': np.ones(shape) * self.scenario.temperature_setpoint + 273.15,
            'humidity': np.ones(shape) * (self.scenario.humidity_setpoint / 100.0),
            'air_age': np.zeros(shape),
            'puf': np.ones(shape),
        }
    
    def _create_zone_map(self) -> np.ndarray:
        """Cria mapa de zonas baseado no cenário."""
        zone_map = np.zeros((self.cells_y, self.cells_x), dtype=np.int16)
        current_x = 0
        current_y = 0
        
        for zone_idx, zone in enumerate(self.scenario.zones):
            x_start = int(current_x * self.scenario.total_width / self.config.cell_size)
            x_end = int((current_x + zone.width_ratio) * self.scenario.total_width / self.config.cell_size)
            y_start = int(current_y * self.scenario.total_height / self.config.cell_size)
            y_end = int((current_y + zone.height_ratio) * self.scenario.total_height / self.config.cell_size)
            
            x_start = max(0, min(x_start, self.cells_x - 1))
            x_end = max(0, min(x_end, self.cells_x))
            y_start = max(0, min(y_start, self.cells_y - 1))
            y_end = max(0, min(y_end, self.cells_y))
            
            zone_map[y_start:y_end, x_start:x_end] = zone_idx + 1
            
            current_x += zone.width_ratio
            if current_x >= 1.0:
                current_x = 0
                current_y += zone.height_ratio
        
        return zone_map

    def _create_obstacle_grid(self) -> np.ndarray:
        """
        Gera um mapa de obstáculos e define POIs baseado no BuildingType.
        0 = Livre
        1 = Parede (Estrutural)
        2 = Móvel/Obstáculo
        """
        obstacles = np.zeros((self.cells_y, self.cells_x), dtype=np.int8)
        self.pois = {'workstations': [], 'equipment': [], 'seats': [], 'beds': []}
        
        # 1. Paredes Externas e Divisões de Zona (Padrão para todos)
        obstacles[0, :] = 1; obstacles[-1, :] = 1; obstacles[:, 0] = 1; obstacles[:, -1] = 1
        
        # Divisões baseadas no zone map
        diff_x = np.abs(np.diff(self.zone_map, axis=1))
        obstacles[:, 1:][(diff_x > 0) & (self.zone_map[:, 1:] != 0)] = 1
        diff_y = np.abs(np.diff(self.zone_map, axis=0))
        obstacles[1:, :][(diff_y > 0) & (self.zone_map[1:, :] != 0)] = 1
        
        # Cria portas nas divisões de zona
        self._create_doors_in_walls(obstacles)

        # 2. Layout Específico por Tipo de Prédio
        b_type = self.scenario.building_type
        
        if b_type == cfg.BuildingType.OFFICE:
            self._create_office_layout(obstacles)
        elif b_type == cfg.BuildingType.GYM:
            self._create_gym_layout(obstacles)
        elif b_type == cfg.BuildingType.RESIDENTIAL:
            self._create_residential_layout(obstacles)
        elif b_type == cfg.BuildingType.SCHOOL:
            self._create_school_layout(obstacles)
        else:
            # Fallback genérico (alguns obstáculos aleatórios)
            rng = np.random.default_rng(42)
            potential = (obstacles == 0) & (self.zone_map > 0)
            furniture_mask = potential & (rng.random(obstacles.shape) < 0.03)
            obstacles[furniture_mask] = 2
            
            # POIs genéricos onde há espaço livre
            free_y, free_x = np.where((obstacles == 0) & (self.zone_map > 0))
            if len(free_x) > 0:
                indices = np.random.choice(len(free_x), size=min(len(free_x), 20), replace=False)
                for i in indices:
                    self.pois['seats'].append((free_x[i], free_y[i]))

        return obstacles

    def _create_doors_in_walls(self, obstacles: np.ndarray):
        """Abre portas nas paredes estruturais internas."""
        for y in range(1, self.cells_y - 1):
            for x in range(1, self.cells_x - 1):
                if obstacles[y, x] == 1:
                    # Porta Vertical (parede acima e abaixo)
                    if obstacles[y-1, x] == 1 and obstacles[y+1, x] == 1:
                        if y % 15 in [7, 8]: obstacles[y, x] = 0
                    # Porta Horizontal (parede esq e dir)
                    elif obstacles[y, x-1] == 1 and obstacles[y, x+1] == 1:
                        if x % 15 in [7, 8]: obstacles[y, x] = 0

    def _create_office_layout(self, obstacles: np.ndarray):
        """Cria ilhas de trabalho (mesas)."""
        # Margem de segurança das paredes
        margin = 3
        # Espaçamento entre ilhas
        stride_x = 5  
        stride_y = 6
        
        for y in range(margin, self.cells_y - margin, stride_y):
            for x in range(margin, self.cells_x - margin, stride_x):
                # Verifica se está dentro de uma zona válida e não em cima de parede
                if obstacles[y, x] == 0 and self.zone_map[y, x] > 0:
                    # Desenha Mesa (2x1) - Obstáculo
                    if x + 1 < self.cells_x:
                        obstacles[y, x] = 2
                        obstacles[y, x+1] = 2
                        
                        # Adiciona POIs (Cadeiras) acima e abaixo da mesa
                        if y - 1 >= 0 and obstacles[y-1, x] == 0:
                            self.pois['workstations'].append((x, y-1))
                        if y + 1 < self.cells_y and obstacles[y+1, x] == 0:
                            self.pois['workstations'].append((x+1, y+1))

    def _create_gym_layout(self, obstacles: np.ndarray):
        """Cria áreas de equipamentos espaçadas."""
        margin = 4
        stride = 7
        
        for y in range(margin, self.cells_y - margin, stride):
            for x in range(margin, self.cells_x - margin, stride):
                if obstacles[y, x] == 0 and self.zone_map[y, x] > 0:
                    # Equipamento Grande (2x2) - Obstáculo
                    if x+1 < self.cells_x and y+1 < self.cells_y:
                        obstacles[y:y+2, x:x+2] = 2
                        
                        # POIs de uso ao redor do equipamento
                        if x-1 >= 0: self.pois['equipment'].append((x-1, y))
                        if x+2 < self.cells_x: self.pois['equipment'].append((x+2, y+1))

    def _create_residential_layout(self, obstacles: np.ndarray):
        """Cria subdivisões de quartos e mobília básica."""
        # Divide zonas grandes em cômodos menores (Paredes internas finas)
        # Apenas exemplo simples: Cruz no meio de cada zona grande
        for zone_idx in range(1, np.max(self.zone_map) + 1):
            rows, cols = np.where(self.zone_map == zone_idx)
            if len(rows) == 0: continue
            
            min_y, max_y = np.min(rows), np.max(rows)
            min_x, max_x = np.min(cols), np.max(cols)
            
            w = max_x - min_x
            h = max_y - min_y
            
            # Se a sala for grande, divide
            if w > 10 and h > 10:
                mid_x = min_x + w // 2
                mid_y = min_y + h // 2
                
                # Paredes divisórias
                obstacles[mid_y, min_x:max_x] = 1
                obstacles[min_y:max_y, mid_x] = 1
                
                # Portas nas divisórias
                obstacles[mid_y, min_x + w//4] = 0
                obstacles[mid_y, min_x + 3*w//4] = 0
                obstacles[min_y + h//4, mid_x] = 0
                obstacles[min_y + 3*h//4, mid_x] = 0
                
                # Adiciona Camas/Sofás (POIs) nos cantos
                # Canto superior esquerdo
                if obstacles[min_y+2, min_x+2] == 0:
                    obstacles[min_y+2:min_y+4, min_x+2:min_x+4] = 2
                    self.pois['beds'].append((min_x+3, min_x+3))
                
                # Canto inferior direito
                if obstacles[max_y-3, max_x-3] == 0:
                    obstacles[max_y-3:max_y-2, max_x-3:max_x-2] = 2
                    self.pois['seats'].append((max_x-4, max_y-4))

    def _create_school_layout(self, obstacles: np.ndarray):
        """Cria fileiras de carteiras voltadas para um lado."""
        margin = 2
        for y in range(margin, self.cells_y - margin, 3): # Fileiras densas
            for x in range(margin, self.cells_x - margin, 3):
                if obstacles[y, x] == 0 and self.zone_map[y, x] > 0:
                    obstacles[y, x] = 2 # Carteira
                    if y+1 < self.cells_y and obstacles[y+1, x] == 0:
                        self.pois['seats'].append((x, y+1)) # Cadeira atrás da mesa

    def is_walkable(self, x: int, y: int) -> bool:
        """Verifica se uma célula é navegável (não é parede nem móvel)."""
        if 0 <= x < self.cells_x and 0 <= y < self.cells_y:
            return self.obstacle_grid[y, x] == 0
        return False

    def _create_material_grid(self) -> Dict[str, np.ndarray]:
        material_grid = {
            'type': np.zeros((self.cells_y, self.cells_x), dtype=np.int16),
            'age': np.zeros((self.cells_y, self.cells_x)),
            'emission_rate_hcho': np.zeros((self.cells_y, self.cells_x)),
            'emission_rate_voc': np.zeros((self.cells_y, self.cells_x)),
            'surface_factor': np.ones((self.cells_y, self.cells_x)),
            'moisture_coefficient': np.zeros((self.cells_y, self.cells_x))
        }
        
        for zone_idx, zone in enumerate(self.scenario.zones):
            zone_cells = np.where(self.zone_map == zone_idx + 1)
            if len(zone_cells[0]) == 0: continue
                
            for material_config in zone.materials:
                material_type = material_config['type']
                if isinstance(material_type, str):
                    try:
                        material_type = cfg.MaterialType(material_type)
                    except ValueError:
                        continue
                
                material_props = self.scenario.default_materials.get(material_type)
                if material_props:
                    surface_area = self._calculate_surface_area(zone_cells, material_config.get('surface', 'walls'), material_config.get('density', 1.0))
                    material_idx = list(cfg.MaterialType).index(material_type)
                    material_grid['type'][zone_cells] = material_idx
                    material_grid['age'][zone_cells] = material_config.get('age_days', material_props.age_days)
                    
                    age_factor = np.exp(-material_props.decay_rate * material_grid['age'][zone_cells])
                    temp_factor = 1.0 + material_props.temperature_coefficient * (self.grids['temperature'][zone_cells] - 293.15)
                    humidity_factor = 1.0 + material_props.moisture_coefficient * (self.grids['humidity'][zone_cells] - 0.5)
                    
                    material_grid['emission_rate_hcho'][zone_cells] = (material_props.hcho_emission_rate * age_factor * temp_factor * humidity_factor * material_props.surface_factor * surface_area)
                    material_grid['emission_rate_voc'][zone_cells] = (material_props.voc_emission_rate * age_factor * temp_factor * humidity_factor * material_props.surface_factor * surface_area)
                    material_grid['moisture_coefficient'][zone_cells] = material_props.moisture_coefficient
        return material_grid
    
    def _calculate_surface_area(self, cells: Tuple, surface_type: str, density: float) -> float:
        cell_area = self.config.cell_size ** 2
        if surface_type == 'walls': return self.scenario.floor_height * (4 * self.config.cell_size) * density
        elif surface_type == 'floor': return cell_area * density
        elif surface_type == 'ceiling': return cell_area * density
        elif surface_type == 'furniture': return cell_area * 2.5 * density
        elif surface_type in ['windows', 'glass']: return cell_area * density * 0.7
        else: return cell_area * density
    
    def _calculate_diffusion_coeffs(self) -> Dict[str, np.ndarray]:
        coeffs = {}
        species_diffusion = {
            'co2': self.config.molecular_diffusion_co2,
            'hcho': self.config.molecular_diffusion_hcho,
            'virus': self.config.molecular_diffusion_virus,
            'voc': getattr(self.config, 'molecular_diffusion_voc', 7.0e-6),
            'pm25': 2.0e-6, 'pm10': 3.0e-6, 'temperature': 2.1e-5, 'humidity': 2.4e-5
        }
        
        for species, base_diffusion in species_diffusion.items():
            turbulent_component = np.zeros((self.cells_y, self.cells_x))
            for zone_idx, zone in enumerate(self.scenario.zones):
                zone_cells = np.where(self.zone_map == zone_idx + 1)
                if len(zone_cells[0]) == 0: continue
                if zone.ventilation_mode == cfg.VentilationMode.MECHANICAL: turbulent_component[zone_cells] = self.config.turbulent_diffusion_high_vent
                elif zone.ventilation_mode == cfg.VentilationMode.NATURAL: turbulent_component[zone_cells] = self.config.turbulent_diffusion_base * 0.3
                elif zone.ventilation_mode == cfg.VentilationMode.DISPLACEMENT: turbulent_component[zone_cells] = self.config.turbulent_diffusion_base * 0.7
                else: turbulent_component[zone_cells] = self.config.turbulent_diffusion_base
            coeffs[species] = base_diffusion + turbulent_component
        return coeffs
    
    def _initialize_velocity_field(self):
        self._create_natural_ventilation()
        for zone_idx, zone in enumerate(self.scenario.zones):
            zone_cells = np.where(self.zone_map == zone_idx + 1)
            if len(zone_cells[0]) == 0: continue
            if zone.ventilation_mode == cfg.VentilationMode.MECHANICAL:
                for inlet_x, inlet_y in zone.air_inlets:
                    cell_x = int(inlet_x * self.cells_x); cell_y = int(inlet_y * self.cells_y)
                    self._create_jet_flow(cell_x, cell_y, zone.target_ach, 'inlet')
                for outlet_x, outlet_y in zone.air_outlets:
                    cell_x = int(outlet_x * self.cells_x); cell_y = int(outlet_y * self.cells_y)
                    self._create_suction_flow(cell_x, cell_y, zone.target_ach)
            elif zone.ventilation_mode == cfg.VentilationMode.DISPLACEMENT:
                for inlet_x, inlet_y in zone.air_inlets:
                    cell_x = int(inlet_x * self.cells_x); cell_y = int(inlet_y * self.cells_y)
                    self._create_displacement_flow(cell_x, cell_y, zone.target_ach)
    
    def _create_natural_ventilation(self):
        for zone_idx, zone in enumerate(self.scenario.zones):
            if not zone.has_windows or zone.window_area_ratio <= 0: continue
            zone_cells = np.where(self.zone_map == zone_idx + 1)
            if len(zone_cells[0]) == 0: continue
            temp_diff = self.external_temperature - np.mean(self.grids['temperature'][zone_cells])
            if abs(temp_diff) > 0.5: self._create_stack_ventilation(zone_cells, temp_diff, zone.window_area_ratio)
            wind_speed = 1.0
            self._create_wind_ventilation(zone_cells, wind_speed, zone.window_area_ratio)
    
    def _create_stack_ventilation(self, zone_cells: Tuple, temp_diff: float, window_ratio: float):
        H = self.scenario.floor_height; g = 9.81
        T_avg = (self.external_temperature + np.mean(self.grids['temperature'][zone_cells])) / 2
        stack_velocity = np.sqrt(2 * g * H * abs(temp_diff) / T_avg) * 0.3
        y_indices, x_indices = zone_cells
        if temp_diff > 0:
            top_cells = y_indices == np.max(y_indices)
            self.velocity_field[y_indices[top_cells], x_indices[top_cells], 1] -= stack_velocity * window_ratio
        else:
            bottom_cells = y_indices == np.min(y_indices)
            self.velocity_field[y_indices[bottom_cells], x_indices[bottom_cells], 1] += stack_velocity * window_ratio
    
    def _create_wind_ventilation(self, zone_cells: Tuple, wind_speed: float, window_ratio: float):
        wind_direction = np.array([-1.0, 0.0])
        wind_vector = wind_direction * wind_speed * window_ratio
        y_indices, x_indices = zone_cells
        east_cells = x_indices == np.max(x_indices)
        self.velocity_field[y_indices[east_cells], x_indices[east_cells], 0] += wind_vector[0]
        self.velocity_field[y_indices[east_cells], x_indices[east_cells], 1] += wind_vector[1]
    
    def _create_jet_flow(self, center_x: int, center_y: int, ach: float, flow_type: str):
        characteristic_velocity = ach * self.scenario.floor_height / 3600.0
        y, x = np.mgrid[0:self.cells_y, 0:self.cells_x]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        influence_radius = min(self.cells_x, self.cells_y) // 4
        mask = distance < influence_radius
        if flow_type == 'inlet':
            self.velocity_field[mask, 1] -= characteristic_velocity * (1 - distance[mask]/influence_radius)
            radial_factor = 0.3
            self.velocity_field[mask, 0] += (x[mask] - center_x) * characteristic_velocity * radial_factor / influence_radius
            self.velocity_field[mask, 1] += (y[mask] - center_y) * characteristic_velocity * radial_factor / influence_radius
        for i in range(2): self.velocity_field[:, :, i] = ndimage.gaussian_filter(self.velocity_field[:, :, i], sigma=1.5)
    
    def _create_suction_flow(self, center_x: int, center_y: int, ach: float):
        characteristic_velocity = ach * self.scenario.floor_height / 3600.0
        y, x = np.mgrid[0:self.cells_y, 0:self.cells_x]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        influence_radius = min(self.cells_x, self.cells_y) // 6
        mask = distance < influence_radius
        self.velocity_field[mask, 1] += characteristic_velocity * (1 - distance[mask]/influence_radius)
    
    def _create_displacement_flow(self, inlet_x: int, inlet_y: int, ach: float):
        characteristic_velocity = ach * self.scenario.floor_height / 3600.0 * 0.5
        y, x = np.mgrid[0:self.cells_y, 0:self.cells_x]
        mask = (x >= inlet_x - 2) & (x <= inlet_x + 2)
        self.velocity_field[mask, 1] += characteristic_velocity
    
    def _calculate_heat_gains(self) -> Dict[int, Dict]:
        heat_gains = {}
        for zone_idx, zone in enumerate(self.scenario.zones):
            zone_cells = np.where(self.zone_map == zone_idx + 1)
            if len(zone_cells[0]) == 0: continue
            area = len(zone_cells[0]) * (self.config.cell_size ** 2)
            equipment_gain = zone.equipment_heat_gain * area
            lighting_gain = zone.lighting_density * area
            heat_gains[zone_idx] = {'equipment': equipment_gain, 'lighting': lighting_gain, 'occupants': 0.0, 'total': equipment_gain + lighting_gain, 'area': area}
        return heat_gains
    
    def _initialize_sources_sinks(self):
        self.sources = {k: np.zeros((self.cells_y, self.cells_x)) for k in ['co2', 'virus', 'hcho', 'voc', 'pm25', 'pm10', 'heat', 'moisture']}
        self.sinks = {k: np.zeros((self.cells_y, self.cells_x)) for k in ['deposition', 'ventilation', 'chemical_reaction', 'heat_loss', 'condensation']}
    
    def _initialize_kalman_filters(self) -> Dict[int, Any]:
        filters = {}
        for zone_idx, zone in enumerate(self.scenario.zones):
            zone_cells = np.where(self.zone_map == zone_idx + 1)
            zone_volume = len(zone_cells[0]) * (self.config.cell_size ** 2) * self.scenario.floor_height
            if zone_volume == 0: continue
            filters[zone_idx] = {
                'volume': zone_volume, 'ach_estimate': zone.target_ach,
                'temperature_estimate': self.scenario.temperature_setpoint + 273.15,
                'humidity_estimate': self.scenario.humidity_setpoint / 100.0,
                'error_covariance': np.eye(3) * 0.1,
                'process_noise': self.config.kalman_process_noise, 'measurement_noise': self.config.kalman_measurement_noise,
                'state': np.array([400 * cfg.CONVERSION_FACTORS['co2_ppm_to_kgm3'], self.scenario.temperature_setpoint + 273.15, self.scenario.humidity_setpoint / 100.0])
            }
        return filters
    
    def add_agent_emission(self, x: int, y: int, emissions: Dict[str, float], metabolic_heat: float = 0.0, moisture_production: float = 0.0):
        cell_volume = (self.config.cell_size ** 2) * self.scenario.floor_height
        for species, amount in emissions.items():
            if species in self.sources:
                concentration = amount / cell_volume
                self.sources[species][y, x] += concentration
        if metabolic_heat > 0: self.sources['heat'][y, x] += metabolic_heat
        if moisture_production > 0: self.sources['moisture'][y, x] += moisture_production
        
        if 0 <= x < self.cells_x and 0 <= y < self.cells_y:
            zone_idx = self.zone_map[y, x] - 1
            if zone_idx >= 0 and zone_idx in self.heat_gains:
                self.heat_gains[zone_idx]['occupants'] += metabolic_heat
                self.heat_gains[zone_idx]['total'] += metabolic_heat

    def apply_agent_sources(self):
        for species in ['co2', 'virus', 'hcho', 'voc', 'pm25', 'pm10']:
            if species in self.sources and species in self.grids:
                self.grids[species] += self.sources[species]

    def apply_material_emissions(self, dt: float):
        cell_volume = self.config.cell_size ** 2 * self.scenario.floor_height
        hcho_emission = self.material_grid['emission_rate_hcho'] * dt
        hcho_decay = self.grids['hcho'] * self.config.hcho_decay_rate * dt
        self.grids['hcho'] += hcho_emission / cell_volume - hcho_decay
        voc_emission = self.material_grid['emission_rate_voc'] * dt
        voc_oxidation = self.grids['voc'] * self.config.voc_oxidation_rate * dt
        self.grids['voc'] += voc_emission / cell_volume - voc_oxidation
        self.material_grid['age'] += dt / (24 * 3600)

    def apply_deposition(self, dt: float):
        deposition_species = {'virus': self.config.deposition_velocity_virus, 'pm25': self.config.deposition_velocity_pm25, 'pm10': self.config.deposition_velocity_pm10}
        cell_size = self.config.cell_size; height = self.scenario.floor_height
        A = 2 * (cell_size * cell_size) + 4 * (cell_size * height)
        V = cell_size * cell_size * height
        av_ratio = A / V
        for species, v_d in deposition_species.items():
            if species in self.grids:
                deposition_rate = v_d * av_ratio * dt
                self.grids[species] *= (1 - deposition_rate)
                self.sinks['deposition'] += self.grids[species] * deposition_rate

    def _get_surface_cells(self) -> Tuple[np.ndarray, np.ndarray]:
        zone_boundaries = np.zeros_like(self.zone_map, dtype=bool)
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            shifted = np.roll(self.zone_map, shift=(dy, dx), axis=(0, 1))
            zone_boundaries |= (self.zone_map != shifted) & (shifted != 0)
        zone_boundaries[0, :] = True; zone_boundaries[-1, :] = True
        zone_boundaries[:, 0] = True; zone_boundaries[:, -1] = True
        return np.where(zone_boundaries)

    def apply_ventilation(self, dt: float):
        for zone_idx, zone in enumerate(self.scenario.zones):
            zone_cells = np.where(self.zone_map == zone_idx + 1)
            if len(zone_cells[0]) == 0: continue
            ach = self.kalman_filters.get(zone_idx, {}).get('ach_estimate', zone.target_ach)
            decay_factor = np.exp(-ach * dt / 3600.0)
            for species in ['co2', 'hcho', 'virus', 'voc', 'pm25', 'pm10']:
                if species in self.grids:
                    external_concentration = self._get_external_concentration(species)
                    self.grids[species][zone_cells] = (self.grids[species][zone_cells] * decay_factor + external_concentration * (1 - decay_factor))
            self._apply_ventilation_thermal(zone_cells, ach, dt)

    def _get_external_concentration(self, species: str) -> float:
        if species == 'co2': return self.external_co2
        elif species == 'hcho': return 5 * cfg.CONVERSION_FACTORS['hcho_ppb_to_kgm3']
        elif species == 'voc': return 20 * cfg.CONVERSION_FACTORS['voc_ppb_to_kgm3']
        elif species == 'virus': return 1e-15
        elif species == 'pm25': return 10e-9
        elif species == 'pm10': return 20e-9
        else: return 0.0

    def _apply_ventilation_thermal(self, zone_cells: Tuple, ach: float, dt: float):
        mix_factor = 1 - np.exp(-ach * dt / 3600.0)
        current_temp = self.grids['temperature'][zone_cells]
        self.grids['temperature'][zone_cells] = (current_temp * (1 - mix_factor) + self.external_temperature * mix_factor)
        current_humidity = self.grids['humidity'][zone_cells]
        self.grids['humidity'][zone_cells] = (current_humidity * (1 - mix_factor) + self.external_humidity * mix_factor)

    def apply_advection(self, dt: float):
        for species in ['co2', 'hcho', 'virus', 'voc', 'pm25', 'pm10', 'temperature', 'humidity', 'air_age']:
            if species in self.grids:
                self._advect_species(species, dt, 0)
                self._advect_species(species, dt, 1)

    def _advect_species(self, species: str, dt: float, direction: int):
        grid = self.grids[species].copy()
        velocity = self.velocity_field[:, :, direction]
        if direction == 0:
            mask_right = velocity > 0
            if np.any(mask_right):
                shifted_right = np.roll(grid, -1, axis=1)
                dx = velocity[mask_right] * dt / self.config.cell_size
                grid[mask_right] = (1 - dx) * grid[mask_right] + dx * shifted_right[mask_right]
            mask_left = velocity < 0
            if np.any(mask_left):
                shifted_left = np.roll(grid, 1, axis=1)
                dx = -velocity[mask_left] * dt / self.config.cell_size
                grid[mask_left] = (1 - dx) * grid[mask_left] + dx * shifted_left[mask_left]
        else:
            mask_down = velocity > 0
            if np.any(mask_down):
                shifted_down = np.roll(grid, -1, axis=0)
                dy = velocity[mask_down] * dt / self.config.cell_size
                grid[mask_down] = (1 - dy) * grid[mask_down] + dy * shifted_down[mask_down]
            mask_up = velocity < 0
            if np.any(mask_up):
                shifted_up = np.roll(grid, 1, axis=0)
                dy = -velocity[mask_up] * dt / self.config.cell_size
                grid[mask_up] = (1 - dy) * grid[mask_up] + dy * shifted_up[mask_up]
        self.grids[species] = grid

    def apply_diffusion(self, dt: float):
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) / (self.config.cell_size ** 2)
        for species in ['co2', 'hcho', 'virus', 'voc', 'pm25', 'pm10', 'temperature', 'humidity']:
            if species in self.diffusion_coeffs:
                D = self.diffusion_coeffs[species]
                laplacian = convolve2d(self.grids[species], kernel, mode='same', boundary='fill', fillvalue=0)
                self.grids[species] += D * laplacian * dt

    def apply_zone_transfer(self, dt: float):
        for connection in self.scenario.connections:
            door_cells = self._find_door_cells(connection)
            if len(door_cells[0]) > 0:
                zone1_idx = self._find_zone_index(connection['from_zone'])
                zone2_idx = self._find_zone_index(connection['to_zone'])
                if zone1_idx is not None and zone2_idx is not None:
                    self._transfer_between_zones(zone1_idx, zone2_idx, door_cells, connection.get('open_probability', 0.8), connection.get('width', 1.0), dt)

    def _find_door_cells(self, connection: Dict) -> Tuple:
        zone1_idx = self._find_zone_index(connection['from_zone'])
        zone2_idx = self._find_zone_index(connection['to_zone'])
        if zone1_idx is None or zone2_idx is None: return (np.array([]), np.array([]))
        zone1_cells = np.where(self.zone_map == zone1_idx + 1)
        zone2_cells = np.where(self.zone_map == zone2_idx + 1)
        if len(zone1_cells[0]) == 0 or len(zone2_cells[0]) == 0: return (np.array([]), np.array([]))
        center1_x = np.mean(zone1_cells[1]); center1_y = np.mean(zone1_cells[0])
        center2_x = np.mean(zone2_cells[1]); center2_y = np.mean(zone2_cells[0])
        door_width_pixels = int(connection.get('width', 1.0) / self.config.cell_size)
        door_cells_y = []; door_cells_x = []
        for i in range(door_width_pixels):
            frac = i / max(door_width_pixels - 1, 1)
            x = int(center1_x * (1 - frac) + center2_x * frac)
            y = int(center1_y * (1 - frac) + center2_y * frac)
            if 0 <= x < self.cells_x and 0 <= y < self.cells_y:
                door_cells_x.append(x); door_cells_y.append(y)
        return (np.array(door_cells_y), np.array(door_cells_x))

    def _find_zone_index(self, zone_name: str) -> Optional[int]:
        for idx, zone in enumerate(self.scenario.zones):
            if zone.name == zone_name: return idx
        return None

    def _transfer_between_zones(self, zone1_idx: int, zone2_idx: int, door_cells: Tuple, open_prob: float, door_width: float, dt: float):
        if np.random.random() > open_prob: return
        y_indices, x_indices = door_cells
        if len(y_indices) == 0: return
        door_area = door_width * self.scenario.floor_height
        cell_area = self.config.cell_size ** 2
        cells_per_door = max(1, int(door_area / cell_area))
        if len(y_indices) > cells_per_door:
            indices = np.random.choice(len(y_indices), cells_per_door, replace=False)
            y_indices = y_indices[indices]; x_indices = x_indices[indices]
        transfer_coeff = 0.01 * dt
        for species in ['co2', 'hcho', 'virus', 'voc', 'temperature', 'humidity']:
            if species not in self.grids: continue
            zone1_cells = np.where(self.zone_map == zone1_idx + 1)
            zone2_cells = np.where(self.zone_map == zone2_idx + 1)
            if len(zone1_cells[0]) == 0 or len(zone2_cells[0]) == 0: continue
            c1_mean = np.mean(self.grids[species][zone1_cells])
            c2_mean = np.mean(self.grids[species][zone2_cells])
            flux = transfer_coeff * (c1_mean - c2_mean)
            for y, x in zip(y_indices, x_indices):
                if self.zone_map[y, x] == zone1_idx + 1: self.grids[species][y, x] -= flux
                elif self.zone_map[y, x] == zone2_idx + 1: self.grids[species][y, x] += flux

    def update_kalman_filters(self, dt: float, current_time: float):
        if not self.config.kalman_enabled: return
        if current_time % self.config.kalman_update_interval < dt:
            for zone_idx, kalman_data in self.kalman_filters.items():
                zone_cells = np.where(self.zone_map == zone_idx + 1)
                if len(zone_cells[0]) == 0: continue
                measured_co2 = np.mean(self.grids['co2'][zone_cells])
                measured_temp = np.mean(self.grids['temperature'][zone_cells])
                measured_humidity = np.mean(self.grids['humidity'][zone_cells])
                generation_rate = np.mean(self.sources['co2'][zone_cells])
                volume = kalman_data['volume']; ach = kalman_data['ach_estimate']
                if measured_co2 > 0 and volume > 0:
                    expected_decay = ach * (measured_co2 - self.external_co2) * volume / 3600.0
                    if expected_decay > 0:
                        adjustment = (generation_rate - expected_decay) / expected_decay * 0.1
                        kalman_data['ach_estimate'] = max(0.1, kalman_data['ach_estimate'] * (1 + adjustment))
                kalman_data['temperature_estimate'] = 0.9 * kalman_data['temperature_estimate'] + 0.1 * measured_temp
                kalman_data['humidity_estimate'] = 0.9 * kalman_data['humidity_estimate'] + 0.1 * measured_humidity

    def apply_plume_thermal_correction(self, agent_positions: List[Tuple], agent_activities: List[str]):
        if not self.config.pem_correction_active: return
        self.grids['puf'] = np.ones((self.cells_y, self.cells_x))
        for (x, y), activity in zip(agent_positions, agent_activities):
            if 0 <= x < self.cells_x and 0 <= y < self.cells_y:
                if activity in ['seated_quiet', 'seated_typing', 'talking']: plume_velocity = self.config.plume_velocity_seated
                elif activity in ['standing', 'walking']: plume_velocity = self.config.plume_velocity_standing
                elif activity in ['exercising_light', 'exercising_intense']: plume_velocity = self.config.plume_velocity_seated * 2.0
                else: plume_velocity = self.config.plume_velocity_seated
                local_velocity = np.sqrt(self.velocity_field[y, x, 0]**2 + self.velocity_field[y, x, 1]**2)
                if local_velocity < 0.05: puf_factor = 1.8
                elif local_velocity < plume_velocity: ratio = local_velocity / plume_velocity; puf_factor = 1.8 - 0.8 * ratio
                else: puf_factor = 1.0
                radius = max(2, min(int(plume_velocity * 5), 10))
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.cells_x and 0 <= ny < self.cells_y:
                            dist = np.sqrt(dx**2 + dy**2)
                            if dist <= radius:
                                weight = (radius - dist) / radius
                                current_puf = self.grids['puf'][ny, nx]
                                new_puf = 1.0 + (puf_factor - 1.0) * weight
                                self.grids['puf'][ny, nx] = max(current_puf, new_puf)
        if 'virus' in self.grids: self.grids['virus_exposure'] = self.grids['virus'] * self.grids['puf']

    def apply_heat_transfer(self, dt: float):
        for zone_idx, gains in self.heat_gains.items():
            zone_cells = np.where(self.zone_map == zone_idx + 1)
            if len(zone_cells[0]) == 0: continue
            total_gain = gains['total']; cell_count = len(zone_cells[0]); gain_per_cell = total_gain / max(cell_count, 1)
            cell_volume = (self.config.cell_size ** 2) * self.scenario.floor_height
            air_density = 1.204; specific_heat = 1005; heat_capacity = cell_volume * air_density * specific_heat
            delta_T = (gain_per_cell * dt) / heat_capacity
            self.grids['temperature'][zone_cells] += delta_T
        y_indices, x_indices = self._get_surface_cells()
        U = self.U_walls; cell_perimeter = 4 * self.config.cell_size; surface_area_per_cell = self.scenario.floor_height * cell_perimeter
        cell_volume = (self.config.cell_size ** 2) * self.scenario.floor_height; heat_capacity = cell_volume * 1.204 * 1005
        for y, x in zip(y_indices, x_indices):
            temp_diff = self.grids['temperature'][y, x] - self.external_temperature
            heat_loss = U * surface_area_per_cell * temp_diff * dt
            delta_T_loss = heat_loss / heat_capacity
            self.grids['temperature'][y, x] -= delta_T_loss
            self.sinks['heat_loss'][y, x] += heat_loss / dt

    def apply_humidity_transfer(self, dt: float):
        y_indices, x_indices = self._get_surface_cells()
        for y, x in zip(y_indices, x_indices):
            temp = self.grids['temperature'][y, x] - 273.15
            humidity = self.grids['humidity'][y, x]
            dew_point = temp - ((100 - humidity * 100) / 5)
            if temp < dew_point - 2:
                condensation_rate = 1e-6 * dt
                surface_area = self.scenario.floor_height * 4 * self.config.cell_size
                condensation_mass = condensation_rate * surface_area
                cell_volume = (self.config.cell_size ** 2) * self.scenario.floor_height
                humidity_reduction = condensation_mass / (cell_volume * 1.204)
                self.grids['humidity'][y, x] = max(0.0, self.grids['humidity'][y, x] - humidity_reduction)
                self.sinks['condensation'][y, x] += condensation_mass

    def update_air_age(self, dt: float):
        self.grids['air_age'] += dt
        for zone_idx, zone in enumerate(self.scenario.zones):
            zone_cells = np.where(self.zone_map == zone_idx + 1)
            if len(zone_cells[0]) == 0: continue
            ach = self.kalman_filters.get(zone_idx, {}).get('ach_estimate', zone.target_ach)
            renewal_probability = ach * dt / 3600.0
            renewal_mask = np.random.random(size=len(zone_cells[0])) < renewal_probability
            y_indices, x_indices = zone_cells
            if np.any(renewal_mask): self.grids['air_age'][y_indices[renewal_mask], x_indices[renewal_mask]] = 0.0

    def step(self, dt: float, current_time: float, agent_data: Optional[Dict] = None):
        self._clear_sources_sinks()
        self.apply_material_emissions(dt)
        if agent_data and 'emissions' in agent_data:
            for emission in agent_data['emissions']:
                if 'amounts' in emission:
                    amounts = emission['amounts']
                    heat = emission.get('metabolic_heat', 0.0)
                    moisture = emission.get('moisture_production', 0.0)
                else:
                    amounts = {}; heat = 0.0; moisture = 0.0
                    sp = emission.get('species'); val = emission.get('amount', 0.0)
                    if sp == 'heat': heat = val
                    elif sp == 'moisture': moisture = val
                    elif sp: amounts[sp] = val
                self.add_agent_emission(emission['x'], emission['y'], amounts, heat, moisture)
        
        self.apply_agent_sources() # Aplica poluição dos agentes
        self.apply_advection(dt)
        self.apply_diffusion(dt)
        self.apply_zone_transfer(dt)
        self.apply_heat_transfer(dt)
        self.apply_humidity_transfer(dt)
        self.apply_deposition(dt)
        self.apply_ventilation(dt)
        self.update_kalman_filters(dt, current_time)
        self.update_air_age(dt)
        if agent_data and 'positions' in agent_data and 'activities' in agent_data:
            self.apply_plume_thermal_correction(agent_data['positions'], agent_data['activities'])
        self._enforce_bounds()
        self._record_history(current_time)

    def _clear_sources_sinks(self):
        for key in self.sources: self.sources[key].fill(0.0)
        for key in self.sinks: self.sinks[key].fill(0.0)
        for zone_idx in self.heat_gains:
            self.heat_gains[zone_idx]['occupants'] = 0.0
            self.heat_gains[zone_idx]['total'] = (self.heat_gains[zone_idx]['equipment'] + self.heat_gains[zone_idx]['lighting'])

    def _enforce_bounds(self):
        self.grids['co2'] = np.maximum(self.grids['co2'], 400 * cfg.CONVERSION_FACTORS['co2_ppm_to_kgm3'])
        self.grids['hcho'] = np.maximum(self.grids['hcho'], 1e-15)
        self.grids['voc'] = np.maximum(self.grids['voc'], 1e-15)
        self.grids['virus'] = np.maximum(self.grids['virus'], 1e-20)
        self.grids['pm25'] = np.maximum(self.grids['pm25'], 1e-15)
        self.grids['pm10'] = np.maximum(self.grids['pm10'], 1e-15)
        self.grids['temperature'] = np.clip(self.grids['temperature'], 283.15, 313.15)
        self.grids['humidity'] = np.clip(self.grids['humidity'], 0.0, 1.0)
        self.grids['air_age'] = np.maximum(self.grids['air_age'], 0.0)
        self.grids['puf'] = np.maximum(self.grids['puf'], 1.0)

    def _record_history(self, current_time: float):
        if current_time % 300 < 0.1:
            self.history['time'].append(current_time)
            zone_concentrations = {}
            for zone_idx in range(len(self.scenario.zones)):
                zone_cells = np.where(self.zone_map == zone_idx + 1)
                if len(zone_cells[0]) > 0:
                    zone_concentrations[zone_idx] = {
                        'co2_ppm': np.mean(self.grids['co2'][zone_cells]) * cfg.CONVERSION_FACTORS['co2_kgm3_to_ppm'],
                        'temperature_c': np.mean(self.grids['temperature'][zone_cells]) - 273.15,
                        'humidity_percent': np.mean(self.grids['humidity'][zone_cells]) * 100,
                        'ach_actual': self.kalman_filters.get(zone_idx, {}).get('ach_estimate', self.scenario.zones[zone_idx].target_ach)
                    }
            self.history['zone_concentrations'].append(zone_concentrations)
            self.history['energy_consumption'].append(self._calculate_energy_consumption())

    def _calculate_energy_consumption(self) -> Dict[str, float]:
        total_energy = 0.0; fan_energy = 0.0; heating_cooling_energy = 0.0
        for zone_idx, zone in enumerate(self.scenario.zones):
            zone_cells = np.where(self.zone_map == zone_idx + 1)
            if len(zone_cells[0]) == 0: continue
            area = len(zone_cells[0]) * (self.config.cell_size ** 2)
            volume = area * self.scenario.floor_height
            ach = self.kalman_filters.get(zone_idx, {}).get('ach_estimate', zone.target_ach)
            airflow_m3s = ach * volume / 3600.0
            fan_power = airflow_m3s * 500 / 0.6
            fan_energy += fan_power
            temp_diff = np.mean(self.grids['temperature'][zone_cells]) - (self.scenario.temperature_setpoint + 273.15)
            heating_cooling_power = airflow_m3s * 1.204 * 1005 * abs(temp_diff)
            heating_cooling_energy += heating_cooling_power / 3.0
            total_energy += fan_power + heating_cooling_power / 3.0
        return {'total_power_w': total_energy, 'fan_power_w': fan_energy, 'hvac_power_w': heating_cooling_energy, 'total_energy_kwh': total_energy * 0.001 / 3600}

    def get_concentrations_at(self, x: int, y: int) -> Dict[str, float]:
        if 0 <= x < self.cells_x and 0 <= y < self.cells_y:
            return {
                'co2_ppm': self.grids['co2'][y, x] * cfg.CONVERSION_FACTORS['co2_kgm3_to_ppm'],
                'hcho_ppb': self.grids['hcho'][y, x] * cfg.CONVERSION_FACTORS['hcho_kgm3_to_ppb'],
                'voc_ppb': self.grids['voc'][y, x] * cfg.CONVERSION_FACTORS['voc_kgm3_to_ppb'],
                'virus_quanta_m3': self.grids['virus'][y, x],
                'virus_exposure_quanta_m3': self.grids.get('virus_exposure', self.grids['virus'])[y, x],
                'pm25_ugm3': self.grids['pm25'][y, x] * 1e9,
                'pm10_ugm3': self.grids['pm10'][y, x] * 1e9,
                'temperature_c': self.grids['temperature'][y, x] - 273.15,
                'humidity_percent': self.grids['humidity'][y, x] * 100,
                'air_velocity_ms': np.sqrt(self.velocity_field[y, x, 0]**2 + self.velocity_field[y, x, 1]**2),
                'air_age_minutes': self.grids['air_age'][y, x] / 60.0,
                'puf_factor': self.grids['puf'][y, x],
                'zone_id': int(self.zone_map[y, x])
            }
        return {}

    def get_zone_statistics(self) -> Dict[int, Dict]:
        stats = {}
        for zone_idx, zone in enumerate(self.scenario.zones):
            zone_cells = np.where(self.zone_map == zone_idx + 1)
            if len(zone_cells[0]) == 0: continue
            area = len(zone_cells[0]) * (self.config.cell_size ** 2)
            volume = area * self.scenario.floor_height
            stats[zone_idx] = {
                'name': zone.name, 'type': zone.zone_type.value, 'area_m2': area, 'volume_m3': volume,
                'occupancy_density': zone.occupancy_density, 'max_occupants': int(area * zone.occupancy_density),
                'concentrations': {
                    'co2_ppm_mean': np.mean(self.grids['co2'][zone_cells]) * cfg.CONVERSION_FACTORS['co2_kgm3_to_ppm'],
                    'co2_ppm_max': np.max(self.grids['co2'][zone_cells]) * cfg.CONVERSION_FACTORS['co2_kgm3_to_ppm'],
                    'hcho_ppb_mean': np.mean(self.grids['hcho'][zone_cells]) * cfg.CONVERSION_FACTORS['hcho_kgm3_to_ppb'],
                    'voc_ppb_mean': np.mean(self.grids['voc'][zone_cells]) * cfg.CONVERSION_FACTORS['voc_kgm3_to_ppb'],
                    'virus_mean': np.mean(self.grids['virus'][zone_cells]),
                    'pm25_ugm3_mean': np.mean(self.grids['pm25'][zone_cells]) * 1e9,
                    'pm10_ugm3_mean': np.mean(self.grids['pm10'][zone_cells]) * 1e9,
                    'temperature_c_mean': np.mean(self.grids['temperature'][zone_cells]) - 273.15,
                    'humidity_percent_mean': np.mean(self.grids['humidity'][zone_cells]) * 100,
                    'air_age_minutes_mean': np.mean(self.grids['air_age'][zone_cells]) / 60.0
                },
                'ach_actual': self.kalman_filters.get(zone_idx, {}).get('ach_estimate', zone.target_ach),
                'ach_target': zone.target_ach, 'ventilation_mode': zone.ventilation_mode.value,
                'has_windows': zone.has_windows, 'window_area_ratio': zone.window_area_ratio,
                'heat_gain_w': self.heat_gains.get(zone_idx, {}).get('total', 0.0),
                'puf_mean': np.mean(self.grids['puf'][zone_cells]) if 'puf' in self.grids else 1.0
            }
        return stats

    def get_visualization_data(self) -> Dict[str, Any]:
        return {
            'grids': {
                'co2_ppm': self.grids['co2'] * cfg.CONVERSION_FACTORS['co2_kgm3_to_ppm'],
                'hcho_ppb': self.grids['hcho'] * cfg.CONVERSION_FACTORS['hcho_kgm3_to_ppb'],
                'voc_ppb': self.grids['voc'] * cfg.CONVERSION_FACTORS['voc_kgm3_to_ppb'],
                'virus_log': np.log10(np.maximum(self.grids['virus'], 0) + 1e-20),
                'virus_exposure_log': np.log10(np.maximum(self.grids.get('virus_exposure', self.grids['virus']), 0) + 1e-20),
                'pm25_ugm3': self.grids['pm25'] * 1e9,
                'pm10_ugm3': self.grids['pm10'] * 1e9,
                'temperature_c': self.grids['temperature'] - 273.15,
                'humidity_percent': self.grids['humidity'] * 100,
                'velocity_magnitude': np.sqrt(self.velocity_field[:,:,0]**2 + self.velocity_field[:,:,1]**2),
                'air_age_minutes': self.grids['air_age'] / 60.0,
                'puf_factor': self.grids['puf'] if 'puf' in self.grids else np.ones_like(self.grids['co2']),
                'zone_map': self.zone_map,
                'material_map': self.material_grid['type'],
                'obstacle_map': self.obstacle_grid
            },
            'velocity_field': {
                'x': self.velocity_field[:,:,0],
                'y': self.velocity_field[:,:,1],
                'magnitude': np.sqrt(self.velocity_field[:,:,0]**2 + self.velocity_field[:,:,1]**2)
            },
            'zone_stats': self.get_zone_statistics(),
            'heat_gains': self.heat_gains,
            'sources_sinks': {'sources_total': {k: np.sum(v) for k, v in self.sources.items()}, 'sinks_total': {k: np.sum(v) for k, v in self.sinks.items()}},
            'metadata': {
                'cells_x': self.cells_x, 'cells_y': self.cells_y, 'cell_size': self.config.cell_size,
                'total_width': self.scenario.total_width, 'total_height': self.scenario.total_height,
                'floor_height': self.scenario.floor_height, 'total_zones': len(self.scenario.zones)
            }
        }

    def set_external_conditions(self, temperature_c: float, humidity_percent: float, co2_ppm: float = 400):
        self.external_temperature = temperature_c + 273.15
        self.external_humidity = humidity_percent / 100.0
        self.external_co2 = co2_ppm * cfg.CONVERSION_FACTORS['co2_ppm_to_kgm3']

    def get_energy_summary(self) -> Dict[str, float]:
        energy_data = self._calculate_energy_consumption()
        total_volume = 0.0; total_area = 0.0
        for zone_idx in range(len(self.scenario.zones)):
            zone_cells = np.where(self.zone_map == zone_idx + 1)
            if len(zone_cells[0]) > 0:
                area = len(zone_cells[0]) * (self.config.cell_size ** 2)
                total_area += area; total_volume += area * self.scenario.floor_height
        energy_data.update({
            'total_area_m2': total_area, 'total_volume_m3': total_volume,
            'energy_intensity_w_m2': energy_data['total_power_w'] / max(total_area, 0.1),
            'ventilation_intensity_lps_m2': sum(self.kalman_filters.get(i, {}).get('ach_estimate', zone.target_ach) * (zone.area_m2 if hasattr(zone, 'area_m2') else 0) * self.scenario.floor_height / 3.6 for i, zone in enumerate(self.scenario.zones)) / max(total_area, 0.1)
        })
        return energy_data

    def get_iaq_summary(self) -> Dict[str, Any]:
        zone_stats = self.get_zone_statistics()
        iaq_indices = {}
        for zone_idx, stats in zone_stats.items():
            concentrations = stats['concentrations']
            co2_index = max(0, min(100, 100 - (concentrations['co2_ppm_mean'] - 400) / 8))
            hcho_index = max(0, min(100, 100 - concentrations['hcho_ppb_mean'] / 0.818))
            temp_diff = abs(concentrations['temperature_c_mean'] - self.scenario.temperature_setpoint)
            temp_index = max(0, min(100, 100 - temp_diff * 10))
            hum_diff = abs(concentrations['humidity_percent_mean'] - self.scenario.humidity_setpoint)
            hum_index = max(0, min(100, 100 - hum_diff * 2))
            iaq_index = 0.4 * co2_index + 0.2 * hcho_index + 0.2 * temp_index + 0.2 * hum_index
            iaq_indices[zone_idx] = {
                'zone_name': stats['name'], 'iaq_index': iaq_index, 'co2_index': co2_index, 'hcho_index': hcho_index,
                'thermal_index': temp_index, 'humidity_index': hum_index,
                'ventilation_adequacy': min(100, stats['ach_actual'] / max(stats['ach_target'], 0.1) * 100),
                'air_freshness': max(0, min(100, 100 - concentrations['air_age_minutes_mean'] * 2))
            }
        total_volume = sum(stats['volume_m3'] for stats in zone_stats.values())
        global_iaq = 0.0
        if total_volume > 0:
            for zone_idx, stats in zone_stats.items():
                weight = stats['volume_m3'] / total_volume
                global_iaq += iaq_indices[zone_idx]['iaq_index'] * weight
        return {
            'global_iaq_index': global_iaq, 'zone_iaq_indices': iaq_indices,
            'worst_zone': min(iaq_indices.items(), key=lambda x: x[1]['iaq_index'])[0] if iaq_indices else None,
            'best_zone': max(iaq_indices.items(), key=lambda x: x[1]['iaq_index'])[0] if iaq_indices else None,
            'compliance': {
                'co2': all(s['concentrations']['co2_ppm_mean'] <= self.scenario.co2_setpoint for s in zone_stats.values()),
                'temperature': all(abs(s['concentrations']['temperature_c_mean'] - self.scenario.temperature_setpoint) <= 2.0 for s in zone_stats.values()),
                'ventilation': all(s['ach_actual'] >= s['ach_target'] * 0.8 for s in zone_stats.values())
            }
        }
