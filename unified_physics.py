"""
MOTOR FÍSICO
Integra toda a física do modelo
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
        self.U_walls = 2.0  # Coeficiente de transferência de calor paredes (W/m²K)
        self.U_windows = 3.0  # Coeficiente de transferência de calor janelas (W/m²K)
        
        # Histórico para análise
        self.history = {
            'time': [],
            'zone_concentrations': [],
            'energy_consumption': []
        }
    
    def _initialize_grids(self) -> Dict[str, np.ndarray]:
        """Inicializa todos os grids de concentração."""
        shape = (self.cells_y, self.cells_x)
        
        return {
            'co2': np.ones(shape) * self.external_co2,
            'hcho': np.ones(shape) * 10 * cfg.CONVERSION_FACTORS['hcho_ppb_to_kgm3'],
            'voc': np.ones(shape) * 50 * cfg.CONVERSION_FACTORS['voc_ppb_to_kgm3'],
            'virus': np.ones(shape) * 1e-12,  # quanta/m³
            'pm25': np.ones(shape) * 5e-9,    # kg/m³ (~5 µg/m³)
            'pm10': np.ones(shape) * 10e-9,   # kg/m³ (~10 µg/m³)
            'temperature': np.ones(shape) * self.scenario.temperature_setpoint + 273.15,  # K
            'humidity': np.ones(shape) * (self.scenario.humidity_setpoint / 100.0),  # fração (0-1)
            'air_age': np.zeros(shape),  # idade do ar em segundos
            'puf': np.ones(shape),  # fator de utilização da pluma
        }
    
    def _create_zone_map(self) -> np.ndarray:
        """Cria mapa de zonas baseado no cenário."""
        zone_map = np.zeros((self.cells_y, self.cells_x), dtype=np.int16)
        
        current_x = 0
        current_y = 0
        
        for zone_idx, zone in enumerate(self.scenario.zones):
            # Calcula limites em células
            x_start = int(current_x * self.scenario.total_width / self.config.cell_size)
            x_end = int((current_x + zone.width_ratio) * self.scenario.total_width / self.config.cell_size)
            y_start = int(current_y * self.scenario.total_height / self.config.cell_size)
            y_end = int((current_y + zone.height_ratio) * self.scenario.total_height / self.config.cell_size)
            
            # Garante limites dentro do grid
            x_start = max(0, min(x_start, self.cells_x - 1))
            x_end = max(0, min(x_end, self.cells_x))
            y_start = max(0, min(y_start, self.cells_y - 1))
            y_end = max(0, min(y_end, self.cells_y))
            
            # Aplica zona (índice começa em 1, 0 = fora de qualquer zona)
            zone_map[y_start:y_end, x_start:x_end] = zone_idx + 1
            
            # Atualiza posição atual para a próxima zona (layout em grade)
            current_x += zone.width_ratio
            if current_x >= 1.0:
                current_x = 0
                current_y += zone.height_ratio
        
        return zone_map
    
    def _create_material_grid(self) -> Dict[str, np.ndarray]:
        """Cria grids de propriedades dos materiais."""
        material_grid = {
            'type': np.zeros((self.cells_y, self.cells_x), dtype=np.int16),
            'age': np.zeros((self.cells_y, self.cells_x)),
            'emission_rate_hcho': np.zeros((self.cells_y, self.cells_x)),
            'emission_rate_voc': np.zeros((self.cells_y, self.cells_x)),
            'surface_factor': np.ones((self.cells_y, self.cells_x)),
            'moisture_coefficient': np.zeros((self.cells_y, self.cells_x))
        }
        
        # Aplica materiais das zonas
        for zone_idx, zone in enumerate(self.scenario.zones):
            zone_cells = np.where(self.zone_map == zone_idx + 1)
            
            if len(zone_cells[0]) == 0:
                continue
                
            for material_config in zone.materials:
                material_type = material_config['type']
                
                # Verifica se é uma string ou MaterialType
                if isinstance(material_type, str):
                    try:
                        material_type = cfg.MaterialType(material_type)
                    except ValueError:
                        continue
                
                material_props = self.scenario.default_materials.get(material_type)
                
                if material_props:
                    # Calcula área de superfície
                    surface_area = self._calculate_surface_area(
                        zone_cells, 
                        material_config.get('surface', 'walls'),
                        material_config.get('density', 1.0)
                    )
                    
                    # Aplica ao grid
                    material_idx = list(cfg.MaterialType).index(material_type)
                    material_grid['type'][zone_cells] = material_idx
                    material_grid['age'][zone_cells] = material_config.get('age_days', material_props.age_days)
                    
                    # Taxa de emissão considerando decaimento e condições ambientais
                    age_factor = np.exp(-material_props.decay_rate * material_grid['age'][zone_cells])
                    temp_factor = 1.0 + material_props.temperature_coefficient * (
                        self.grids['temperature'][zone_cells] - 293.15  # 20°C reference
                    )
                    
                    # Fator de umidade
                    humidity_factor = 1.0 + material_props.moisture_coefficient * (
                        self.grids['humidity'][zone_cells] - 0.5  # 50% de referência
                    )
                    
                    material_grid['emission_rate_hcho'][zone_cells] = (
                        material_props.hcho_emission_rate * age_factor * temp_factor *
                        humidity_factor *
                        material_props.surface_factor *
                        surface_area
                    )
                    
                    material_grid['emission_rate_voc'][zone_cells] = (
                        material_props.voc_emission_rate *
                        age_factor *
                        temp_factor *
                        humidity_factor *
                        material_props.surface_factor *
                        surface_area
                    )
                    
                    material_grid['moisture_coefficient'][zone_cells] = material_props.moisture_coefficient
        
        return material_grid
    
    def _calculate_surface_area(self, cells: Tuple, surface_type: str, density: float) -> float:
        """Calcula área superficial efetiva."""
        cell_area = self.config.cell_size ** 2
        
        if surface_type == 'walls':
            # Para paredes: altura da sala * perímetro da célula
            return self.scenario.floor_height * (4 * self.config.cell_size) * density
        elif surface_type == 'floor':
            return cell_area * density
        elif surface_type == 'ceiling':
            return cell_area * density
        elif surface_type == 'furniture':
            # Mobília tem maior área superficial
            return cell_area * 2.5 * density
        elif surface_type in ['windows', 'glass']:
            return cell_area * density * 0.7  # eficiência reduzida
        else:
            return cell_area * density
    
    def _calculate_diffusion_coeffs(self) -> Dict[str, np.ndarray]:
        """Calcula coeficientes de difusão por célula."""
        coeffs = {}
        
        # Difusão base por zona
        species_diffusion = {
            'co2': self.config.molecular_diffusion_co2,
            'hcho': self.config.molecular_diffusion_hcho,
            'virus': self.config.molecular_diffusion_virus,
            'voc': getattr(self.config, 'molecular_diffusion_voc', 7.0e-6),
            'pm25': 2.0e-6,
            'pm10': 3.0e-6,
            'temperature': 2.1e-5,  # difusividade térmica do ar
            'humidity': 2.4e-5  # difusividade de vapor de água
        }
        
        for species, base_diffusion in species_diffusion.items():
            # Adiciona componente turbulenta baseada na ventilação
            turbulent_component = np.zeros((self.cells_y, self.cells_x))
            
            for zone_idx, zone in enumerate(self.scenario.zones):
                zone_cells = np.where(self.zone_map == zone_idx + 1)
                
                if len(zone_cells[0]) == 0:
                    continue
                    
                if zone.ventilation_mode == cfg.VentilationMode.MECHANICAL:
                    turbulent_component[zone_cells] = self.config.turbulent_diffusion_high_vent
                elif zone.ventilation_mode == cfg.VentilationMode.NATURAL:
                    turbulent_component[zone_cells] = self.config.turbulent_diffusion_base * 0.3
                elif zone.ventilation_mode == cfg.VentilationMode.DISPLACEMENT:
                    turbulent_component[zone_cells] = self.config.turbulent_diffusion_base * 0.7
                else:
                    turbulent_component[zone_cells] = self.config.turbulent_diffusion_base
            
            coeffs[species] = base_diffusion + turbulent_component
        
        return coeffs
    
    def _initialize_velocity_field(self):
        """Inicializa campo de velocidade baseado nas entradas/saídas de ar."""
        # Primeiro, cria fluxos naturais baseados em janelas
        self._create_natural_ventilation()
        
        # Depois, adiciona fluxos mecânicos
        for zone_idx, zone in enumerate(self.scenario.zones):
            zone_cells = np.where(self.zone_map == zone_idx + 1)
            
            if len(zone_cells[0]) == 0:
                continue
                
            # Ventilação mecânica
            if zone.ventilation_mode == cfg.VentilationMode.MECHANICAL:
                for inlet_x, inlet_y in zone.air_inlets:
                    # Converte posições normalizadas para células
                    cell_x = int(inlet_x * self.cells_x)
                    cell_y = int(inlet_y * self.cells_y)
                    
                    # Cria jato radial do difusor
                    self._create_jet_flow(cell_x, cell_y, zone.target_ach, 'inlet')
                
                for outlet_x, outlet_y in zone.air_outlets:
                    cell_x = int(outlet_x * self.cells_x)
                    cell_y = int(outlet_y * self.cells_y)
                    
                    # Cria fluxo de sucção
                    self._create_suction_flow(cell_x, cell_y, zone.target_ach)
            
            # Ventilação por deslocamento
            elif zone.ventilation_mode == cfg.VentilationMode.DISPLACEMENT:
                # Entrada baixa, saída alta
                for inlet_x, inlet_y in zone.air_inlets:
                    cell_x = int(inlet_x * self.cells_x)
                    cell_y = int(inlet_y * self.cells_y)
                    self._create_displacement_flow(cell_x, cell_y, zone.target_ach)
    
    def _create_natural_ventilation(self):
        """Cria fluxos de ventilação natural baseados em janelas."""
        for zone_idx, zone in enumerate(self.scenario.zones):
            if not zone.has_windows or zone.window_area_ratio <= 0:
                continue
                
            zone_cells = np.where(self.zone_map == zone_idx + 1)
            if len(zone_cells[0]) == 0:
                continue
            
            # Cria fluxo cruzado se houver diferença de temperatura
            temp_diff = self.external_temperature - np.mean(self.grids['temperature'][zone_cells])
            
            if abs(temp_diff) > 0.5:  # diferença significativa
                # Ventilação por empuxo térmico
                self._create_stack_ventilation(zone_cells, temp_diff, zone.window_area_ratio)
            
            # Ventilação por vento (simplificado)
            wind_speed = 1.0  # m/s (valor padrão)
            self._create_wind_ventilation(zone_cells, wind_speed, zone.window_area_ratio)
    
    def _create_stack_ventilation(self, zone_cells: Tuple, temp_diff: float, window_ratio: float):
        """Cria ventilação por empuxo térmico (stack effect)."""
        # Velocidade do empuxo: v = sqrt(2*g*H*ΔT/T)
        H = self.scenario.floor_height
        g = 9.81
        T_avg = (self.external_temperature + np.mean(self.grids['temperature'][zone_cells])) / 2
        
        if temp_diff > 0:
            # Ar externo mais quente -> entra por cima
            stack_velocity = np.sqrt(2 * g * H * abs(temp_diff) / T_avg) * 0.3  # fator de eficiência
            y_indices, x_indices = zone_cells
            
            # Aplica no topo da zona
            top_cells = y_indices == np.max(y_indices)
            self.velocity_field[y_indices[top_cells], x_indices[top_cells], 1] -= stack_velocity * window_ratio
        else:
            # Ar externo mais frio -> entra por baixo
            stack_velocity = np.sqrt(2 * g * H * abs(temp_diff) / T_avg) * 0.3
            y_indices, x_indices = zone_cells
            
            # Aplica na base da zona
            bottom_cells = y_indices == np.min(y_indices)
            self.velocity_field[y_indices[bottom_cells], x_indices[bottom_cells], 1] += stack_velocity * window_ratio
    
    def _create_wind_ventilation(self, zone_cells: Tuple, wind_speed: float, window_ratio: float):
        """Cria ventilação por vento."""
        # Direção do vento (leste para oeste por padrão)
        wind_direction = np.array([-1.0, 0.0])  # vento vindo do leste
        wind_vector = wind_direction * wind_speed * window_ratio
        
        y_indices, x_indices = zone_cells
        
        # Aplica nas células de borda leste
        east_cells = x_indices == np.max(x_indices)
        self.velocity_field[y_indices[east_cells], x_indices[east_cells], 0] += wind_vector[0]
        self.velocity_field[y_indices[east_cells], x_indices[east_cells], 1] += wind_vector[1]
    
    def _create_jet_flow(self, center_x: int, center_y: int, ach: float, flow_type: str):
        """Cria fluxo em jato radial."""
        # Velocidade característica baseada no ACH
        # v ≈ ACH * altura / 3600
        characteristic_velocity = ach * self.scenario.floor_height / 3600.0
        
        # Cria grade de distâncias
        y, x = np.mgrid[0:self.cells_y, 0:self.cells_x]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Máscara para área de influência
        influence_radius = min(self.cells_x, self.cells_y) // 4
        mask = distance < influence_radius
        
        if flow_type == 'inlet':
            # Jato para baixo (típico de difusor de teto)
            self.velocity_field[mask, 1] -= characteristic_velocity * (1 - distance[mask]/influence_radius)
            
            # Componente radial
            radial_factor = 0.3
            self.velocity_field[mask, 0] += (x[mask] - center_x) * characteristic_velocity * radial_factor / influence_radius
            self.velocity_field[mask, 1] += (y[mask] - center_y) * characteristic_velocity * radial_factor / influence_radius
        
        # Suaviza o campo
        for i in range(2):
            self.velocity_field[:, :, i] = ndimage.gaussian_filter(
                self.velocity_field[:, :, i], sigma=1.5
            )
    
    def _create_suction_flow(self, center_x: int, center_y: int, ach: float):
        """Cria fluxo de sucção para exaustores."""
        characteristic_velocity = ach * self.scenario.floor_height / 3600.0
        
        y, x = np.mgrid[0:self.cells_y, 0:self.cells_x]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        influence_radius = min(self.cells_x, self.cells_y) // 6
        mask = distance < influence_radius
        
        # Sucção para cima (exaustor)
        self.velocity_field[mask, 1] += characteristic_velocity * (1 - distance[mask]/influence_radius)
    
    def _create_displacement_flow(self, inlet_x: int, inlet_y: int, ach: float):
        """Cria fluxo de ventilação por deslocamento."""
        characteristic_velocity = ach * self.scenario.floor_height / 3600.0 * 0.5  # mais lento
        
        y, x = np.mgrid[0:self.cells_y, 0:self.cells_x]
        
        # Fluxo vertical ascendente a partir da entrada
        mask = (x >= inlet_x - 2) & (x <= inlet_x + 2)
        self.velocity_field[mask, 1] += characteristic_velocity
    
    def _calculate_heat_gains(self) -> Dict[int, Dict]:
        """Calcula ganhos de calor por zona."""
        heat_gains = {}
        
        for zone_idx, zone in enumerate(self.scenario.zones):
            zone_cells = np.where(self.zone_map == zone_idx + 1)
            
            if len(zone_cells[0]) == 0:
                continue
                
            area = len(zone_cells[0]) * (self.config.cell_size ** 2)
            
            # Ganhos de calor por equipamentos e iluminação
            equipment_gain = zone.equipment_heat_gain * area  # W
            lighting_gain = zone.lighting_density * area  # W
            
            heat_gains[zone_idx] = {
                'equipment': equipment_gain,
                'lighting': lighting_gain,
                'occupants': 0.0,  # será atualizado com agentes
                'total': equipment_gain + lighting_gain,
                'area': area
            }
        
        return heat_gains
    
    def _initialize_sources_sinks(self):
        """Inicializa fontes e sumidouros fixos."""
        self.sources = {
            'co2': np.zeros((self.cells_y, self.cells_x)),
            'virus': np.zeros((self.cells_y, self.cells_x)),
            'hcho': np.zeros((self.cells_y, self.cells_x)),
            'voc': np.zeros((self.cells_y, self.cells_x)),
            'pm25': np.zeros((self.cells_y, self.cells_x)),
            'pm10': np.zeros((self.cells_y, self.cells_x)),
            'heat': np.zeros((self.cells_y, self.cells_x)),  # calor sensível (W)
            'moisture': np.zeros((self.cells_y, self.cells_x))  # vapor de água (kg/s)
        }
        
        self.sinks = {
            'deposition': np.zeros((self.cells_y, self.cells_x)),
            'ventilation': np.zeros((self.cells_y, self.cells_x)),
            'chemical_reaction': np.zeros((self.cells_y, self.cells_x)),
            'heat_loss': np.zeros((self.cells_y, self.cells_x)),
            'condensation': np.zeros((self.cells_y, self.cells_x))
        }
    
    def _initialize_kalman_filters(self) -> Dict[int, Any]:
        """Inicializa filtros de Kalman por zona."""
        filters = {}
        
        for zone_idx, zone in enumerate(self.scenario.zones):
            # Calcula volume da zona
            zone_cells = np.where(self.zone_map == zone_idx + 1)
            zone_volume = len(zone_cells[0]) * (self.config.cell_size ** 2) * self.scenario.floor_height
            
            if zone_volume == 0:
                continue
            
            # Cria filtro (estado: CO2, temperatura, umidade)
            filters[zone_idx] = {
                'volume': zone_volume,
                'ach_estimate': zone.target_ach,
                'temperature_estimate': self.scenario.temperature_setpoint + 273.15,
                'humidity_estimate': self.scenario.humidity_setpoint / 100.0,
                'error_covariance': np.eye(3) * 0.1,  # matriz 3x3
                'process_noise': self.config.kalman_process_noise,
                'measurement_noise': self.config.kalman_measurement_noise,
                'state': np.array([
                    400 * cfg.CONVERSION_FACTORS['co2_ppm_to_kgm3'],  # CO2
                    self.scenario.temperature_setpoint + 273.15,  # temperatura
                    self.scenario.humidity_setpoint / 100.0  # umidade
                ])
            }
        
        return filters
    
    def add_agent_emission(self, x: int, y: int, emissions: Dict[str, float], metabolic_heat: float = 0.0, 
                          moisture_production: float = 0.0):
        """Adiciona emissões de um agente."""
        cell_volume = (self.config.cell_size ** 2) * self.scenario.floor_height
        
        for species, amount in emissions.items():
            if species in self.sources:
                concentration = amount / cell_volume
                self.sources[species][y, x] += concentration
        
        # Adiciona calor sensível e umidade
        if metabolic_heat > 0:
            self.sources['heat'][y, x] += metabolic_heat
        
        if moisture_production > 0:
            self.sources['moisture'][y, x] += moisture_production
        
        # Atualiza ganhos de calor da zona
        if 0 <= x < self.cells_x and 0 <= y < self.cells_y:
            zone_idx = self.zone_map[y, x] - 1
            if zone_idx >= 0 and zone_idx in self.heat_gains:
                self.heat_gains[zone_idx]['occupants'] += metabolic_heat
                self.heat_gains[zone_idx]['total'] += metabolic_heat
    
    def apply_agent_sources(self):
        """Aplica as fontes acumuladas aos grids de concentração."""
        for species in ['co2', 'virus', 'hcho', 'voc', 'pm25', 'pm10']:
            if species in self.sources and species in self.grids:
                self.grids[species] += self.sources[species]
    
    def apply_material_emissions(self, dt: float):
        """Aplica emissões de materiais de construção."""
        cell_volume = self.config.cell_size ** 2 * self.scenario.floor_height
        
        # Emissão de HCHO com decaimento químico
        hcho_emission = self.material_grid['emission_rate_hcho'] * dt
        hcho_decay = self.grids['hcho'] * self.config.hcho_decay_rate * dt
        self.grids['hcho'] += hcho_emission / cell_volume - hcho_decay
        
        # Emissão de VOC com oxidação
        voc_emission = self.material_grid['emission_rate_voc'] * dt
        voc_oxidation = self.grids['voc'] * self.config.voc_oxidation_rate * dt
        self.grids['voc'] += voc_emission / cell_volume - voc_oxidation
        
        # Atualiza idade dos materiais
        self.material_grid['age'] += dt / (24 * 3600)  # converte segundos para dias
    
    def apply_deposition(self, dt: float):
        """Aplica deposição em superfícies."""
        # Para cada espécie que deposita
        deposition_species = {
            'virus': self.config.deposition_velocity_virus,
            'pm25': self.config.deposition_velocity_pm25,
            'pm10': self.config.deposition_velocity_pm10
        }
        
        for species, v_d in deposition_species.items():
            if species in self.grids:
                # Perda por deposição: dC/dt = -v_d * (A/V) * C
                # Área por volume considerando todas as superfícies
                cell_size = self.config.cell_size
                height = self.scenario.floor_height
                
                # Área das 6 faces da célula
                A = 2 * (cell_size * cell_size) + 4 * (cell_size * height)
                V = cell_size * cell_size * height
                av_ratio = A / V
                
                deposition_rate = v_d * av_ratio * dt
                
                # Aplica em todas as células (simplificado)
                self.grids[species] *= (1 - deposition_rate)
                
                # Registra no sink
                self.sinks['deposition'] += self.grids[species] * deposition_rate
    
    def _get_surface_cells(self) -> Tuple[np.ndarray, np.ndarray]:
        """Identifica células adjacentes a superfícies."""
        # Identifica fronteiras entre zonas
        zone_boundaries = np.zeros_like(self.zone_map, dtype=bool)
        
        # Verifica vizinhos em 4 direções
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            shifted = np.roll(self.zone_map, shift=(dy, dx), axis=(0, 1))
            zone_boundaries |= (self.zone_map != shifted) & (shifted != 0)
        
        # Também considera bordas externas
        zone_boundaries[0, :] = True  # borda superior
        zone_boundaries[-1, :] = True  # borda inferior
        zone_boundaries[:, 0] = True  # borda esquerda
        zone_boundaries[:, -1] = True  # borda direita
        
        return np.where(zone_boundaries)
    
    def apply_ventilation(self, dt: float):
        """Aplica ventilação por zona."""
        for zone_idx, zone in enumerate(self.scenario.zones):
            zone_cells = np.where(self.zone_map == zone_idx + 1)
            
            if len(zone_cells[0]) == 0:
                continue
            
            # Fator de decaimento exponencial
            ach = self.kalman_filters.get(zone_idx, {}).get('ach_estimate', zone.target_ach)
            decay_factor = np.exp(-ach * dt / 3600.0)
            
            # Aplica ventilação para poluentes
            for species in ['co2', 'hcho', 'virus', 'voc', 'pm25', 'pm10']:
                if species in self.grids:
                    # Ventilação traz ar externo
                    external_concentration = self._get_external_concentration(species)
                    self.grids[species][zone_cells] = (
                        self.grids[species][zone_cells] * decay_factor +
                        external_concentration * (1 - decay_factor)
                    )
            
            # Ventilação para temperatura e umidade (mistura com ar externo)
            self._apply_ventilation_thermal(zone_cells, ach, dt)
    
    def _get_external_concentration(self, species: str) -> float:
        """Retorna concentração externa para uma espécie."""
        if species == 'co2':
            return self.external_co2
        elif species == 'hcho':
            return 5 * cfg.CONVERSION_FACTORS['hcho_ppb_to_kgm3']  # 5 ppb externo
        elif species == 'voc':
            return 20 * cfg.CONVERSION_FACTORS['voc_ppb_to_kgm3']  # 20 ppb externo
        elif species == 'virus':
            return 1e-15  # muito baixo externo
        elif species == 'pm25':
            return 10e-9  # 10 µg/m³ externo
        elif species == 'pm10':
            return 20e-9  # 20 µg/m³ externo
        else:
            return 0.0
    
    def _apply_ventilation_thermal(self, zone_cells: Tuple, ach: float, dt: float):
        """Aplica ventilação para temperatura e umidade."""
        # Fator de mistura
        mix_factor = 1 - np.exp(-ach * dt / 3600.0)
        
        # Temperatura (mistura com ar externo)
        current_temp = self.grids['temperature'][zone_cells]
        target_temp = self.external_temperature
        self.grids['temperature'][zone_cells] = (
            current_temp * (1 - mix_factor) + target_temp * mix_factor
        )
        
        # Umidade (mistura com ar externo)
        current_humidity = self.grids['humidity'][zone_cells]
        target_humidity = self.external_humidity
        self.grids['humidity'][zone_cells] = (
            current_humidity * (1 - mix_factor) + target_humidity * mix_factor
        )
    
    def apply_advection(self, dt: float):
        """Aplica advecção pelo campo de velocidade usando método de passo fracionado."""
        # Primeiro componente x
        for species in ['co2', 'hcho', 'virus', 'voc', 'pm25', 'pm10', 'temperature', 'humidity', 'air_age']:
            if species in self.grids:
                self._advect_species(species, dt, 0)  # direção x
                self._advect_species(species, dt, 1)  # direção y
    
    def _advect_species(self, species: str, dt: float, direction: int):
        """Advecção para uma direção específica."""
        grid = self.grids[species].copy()
        velocity = self.velocity_field[:, :, direction]
        
        # Usa esquema upwind de primeira ordem para estabilidade
        if direction == 0:  # x direction
            # Para velocidade positiva (direita)
            mask_right = velocity > 0
            if np.any(mask_right):
                shifted_right = np.roll(grid, -1, axis=1)
                dx = velocity[mask_right] * dt / self.config.cell_size
                grid[mask_right] = (1 - dx) * grid[mask_right] + dx * shifted_right[mask_right]
            
            # Para velocidade negativa (esquerda)
            mask_left = velocity < 0
            if np.any(mask_left):
                shifted_left = np.roll(grid, 1, axis=1)
                dx = -velocity[mask_left] * dt / self.config.cell_size
                grid[mask_left] = (1 - dx) * grid[mask_left] + dx * shifted_left[mask_left]
        
        else:  # y direction
            # Para velocidade positiva (baixo)
            mask_down = velocity > 0
            if np.any(mask_down):
                shifted_down = np.roll(grid, -1, axis=0)
                dy = velocity[mask_down] * dt / self.config.cell_size
                grid[mask_down] = (1 - dy) * grid[mask_down] + dy * shifted_down[mask_down]
            
            # Para velocidade negativa (cima)
            mask_up = velocity < 0
            if np.any(mask_up):
                shifted_up = np.roll(grid, 1, axis=0)
                dy = -velocity[mask_up] * dt / self.config.cell_size
                grid[mask_up] = (1 - dy) * grid[mask_up] + dy * shifted_up[mask_up]
        
        self.grids[species] = grid
    
    def apply_diffusion(self, dt: float):
        """Aplica difusão molecular e turbulenta."""
        kernel = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]]) / (self.config.cell_size ** 2)
        
        for species in ['co2', 'hcho', 'virus', 'voc', 'pm25', 'pm10', 'temperature', 'humidity']:
            if species in self.diffusion_coeffs:
                D = self.diffusion_coeffs[species]
                grid = self.grids[species]
                
                # Calcula Laplaciano
                laplacian = convolve2d(grid, kernel, mode='same', boundary='fill', fillvalue=0)
                
                # Atualiza concentração (método explícito de Euler)
                self.grids[species] = grid + D * laplacian * dt
    
    def apply_zone_transfer(self, dt: float):
        """Aplica transferência de massa entre zonas conectadas."""
        for connection in self.scenario.connections:
            # Encontra células da porta/conexão
            door_cells = self._find_door_cells(connection)
            
            if len(door_cells[0]) > 0:
                # Calcula fluxo baseado na diferença de concentração
                zone1_idx = self._find_zone_index(connection['from_zone'])
                zone2_idx = self._find_zone_index(connection['to_zone'])
                
                if zone1_idx is not None and zone2_idx is not None:
                    self._transfer_between_zones(
                        zone1_idx, zone2_idx, door_cells, 
                        connection.get('open_probability', 0.8),
                        connection.get('width', 1.0),
                        dt
                    )
    
    def _find_door_cells(self, connection: Dict) -> Tuple:
        """Encontra células correspondentes a uma porta/conexão."""
        zone1_idx = self._find_zone_index(connection['from_zone'])
        zone2_idx = self._find_zone_index(connection['to_zone'])
        
        if zone1_idx is None or zone2_idx is None:
            return (np.array([]), np.array([]))
        
        # Encontra fronteira entre as duas zonas
        zone1_cells = np.where(self.zone_map == zone1_idx + 1)
        zone2_cells = np.where(self.zone_map == zone2_idx + 1)
        
        # Para simplificar, retorna células próximas ao centro da fronteira
        if len(zone1_cells[0]) == 0 or len(zone2_cells[0]) == 0:
            return (np.array([]), np.array([]))
        
        # Calcula centro médio de cada zona
        center1_x = np.mean(zone1_cells[1])
        center1_y = np.mean(zone1_cells[0])
        center2_x = np.mean(zone2_cells[1])
        center2_y = np.mean(zone2_cells[0])
        
        # Cria uma linha entre os centros
        door_width_pixels = int(connection.get('width', 1.0) / self.config.cell_size)
        
        # Encontra células próximas à linha
        door_cells_y = []
        door_cells_x = []
        
        for i in range(door_width_pixels):
            frac = i / max(door_width_pixels - 1, 1)
            x = int(center1_x * (1 - frac) + center2_x * frac)
            y = int(center1_y * (1 - frac) + center2_y * frac)
            
            if 0 <= x < self.cells_x and 0 <= y < self.cells_y:
                door_cells_x.append(x)
                door_cells_y.append(y)
        
        return (np.array(door_cells_y), np.array(door_cells_x))
    
    def _find_zone_index(self, zone_name: str) -> Optional[int]:
        """Encontra índice de uma zona pelo nome."""
        for idx, zone in enumerate(self.scenario.zones):
            if zone.name == zone_name:
                return idx
        return None
    
    def _transfer_between_zones(self, zone1_idx: int, zone2_idx: int, 
                               door_cells: Tuple, open_prob: float, door_width: float, dt: float):
        """Transfere massa entre duas zonas conectadas."""
        if np.random.random() > open_prob:
            return  # Porta fechada
        
        y_indices, x_indices = door_cells
        
        if len(y_indices) == 0:
            return
        
        # Área efetiva da porta
        door_area = door_width * self.scenario.floor_height  # m²
        cell_area = self.config.cell_size ** 2
        cells_per_door = max(1, int(door_area / cell_area))
        
        # Seleciona um subconjunto de células para representar a porta
        if len(y_indices) > cells_per_door:
            indices = np.random.choice(len(y_indices), cells_per_door, replace=False)
            y_indices = y_indices[indices]
            x_indices = x_indices[indices]
        
        # Coeficiente de transferência baseado na diferença de pressão
        transfer_coeff = 0.01 * dt
        
        for species in ['co2', 'hcho', 'virus', 'voc', 'temperature', 'humidity']:
            if species not in self.grids:
                continue
            
            # Médias nas zonas
            zone1_cells = np.where(self.zone_map == zone1_idx + 1)
            zone2_cells = np.where(self.zone_map == zone2_idx + 1)
            
            if len(zone1_cells[0]) == 0 or len(zone2_cells[0]) == 0:
                continue
            
            c1_mean = np.mean(self.grids[species][zone1_cells])
            c2_mean = np.mean(self.grids[species][zone2_cells])
            
            # Fluxo proporcional à diferença
            flux = transfer_coeff * (c1_mean - c2_mean)
            
            # Aplica fluxo através das células da porta
            for y, x in zip(y_indices, x_indices):
                if self.zone_map[y, x] == zone1_idx + 1:
                    self.grids[species][y, x] -= flux
                elif self.zone_map[y, x] == zone2_idx + 1:
                    self.grids[species][y, x] += flux
    
    def update_kalman_filters(self, dt: float, current_time: float):
        """Atualiza filtros de Kalman para estimar ACH real."""
        if not self.config.kalman_enabled:
            return
        
        update_interval = self.config.kalman_update_interval
        
        if current_time % update_interval < dt:
            for zone_idx, kalman_data in self.kalman_filters.items():
                zone_cells = np.where(self.zone_map == zone_idx + 1)
                
                if len(zone_cells[0]) == 0:
                    continue
                
                # Medições atuais
                measured_co2 = np.mean(self.grids['co2'][zone_cells])
                measured_temp = np.mean(self.grids['temperature'][zone_cells])
                measured_humidity = np.mean(self.grids['humidity'][zone_cells])
                
                # Predições (modelo simplificado)
                # Para CO2: dC/dt = (generation - ventilation*(C-Cext))/volume
                generation_rate = np.mean(self.sources['co2'][zone_cells])
                volume = kalman_data['volume']
                ach = kalman_data['ach_estimate']
                
                # Atualiza estimativa de ACH baseada no balanço de CO2
                if measured_co2 > 0 and volume > 0:
                    C_ext = self.external_co2
                    expected_decay = ach * (measured_co2 - C_ext) * volume / 3600.0
                    
                    if expected_decay > 0:
                        adjustment = (generation_rate - expected_decay) / expected_decay * 0.1
                        kalman_data['ach_estimate'] = max(0.1, kalman_data['ach_estimate'] * (1 + adjustment))
                
                # Atualiza estimativas de temperatura e umidade
                kalman_data['temperature_estimate'] = 0.9 * kalman_data['temperature_estimate'] + 0.1 * measured_temp
                kalman_data['humidity_estimate'] = 0.9 * kalman_data['humidity_estimate'] + 0.1 * measured_humidity
    
    def apply_plume_thermal_correction(self, agent_positions: List[Tuple], agent_activities: List[str]):
        """Aplica correção da pluma térmica para exposição pessoal."""
        if not self.config.pem_correction_active:
            return
        
        # Reinicia o grid PUF
        self.grids['puf'] = np.ones((self.cells_y, self.cells_x))
        
        # Para cada agente, calcula fator PEM
        for (x, y), activity in zip(agent_positions, agent_activities):
            if 0 <= x < self.cells_x and 0 <= y < self.cells_y:
                # Velocidade da pluma baseada na atividade
                if activity in ['seated_quiet', 'seated_typing', 'talking']:
                    plume_velocity = self.config.plume_velocity_seated
                elif activity in ['standing', 'walking']:
                    plume_velocity = self.config.plume_velocity_standing
                elif activity in ['exercising_light', 'exercising_intense']:
                    plume_velocity = self.config.plume_velocity_seated * 2.0
                else:
                    plume_velocity = self.config.plume_velocity_seated
                
                # Velocidade local do ar
                local_velocity = np.sqrt(
                    self.velocity_field[y, x, 0]**2 + 
                    self.velocity_field[y, x, 1]**2
                )
                
                # Calcula fator PUF (Personal Exposure Factor)
                if local_velocity < 0.05:  # ar quase estagnado
                    puf_factor = 1.8  # maior exposição pessoal
                elif local_velocity < plume_velocity:
                    # Mistura parcial
                    ratio = local_velocity / plume_velocity
                    puf_factor = 1.8 - 0.8 * ratio
                else:
                    puf_factor = 1.0  # bem misturado
                
                # Aplica em área ao redor do agente
                radius = int(plume_velocity * 5)  # células proporcional à velocidade
                radius = max(2, min(radius, 10))
                
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
        
        # Cria grid de exposição viral corrigida
        if 'virus' in self.grids:
            self.grids['virus_exposure'] = self.grids['virus'] * self.grids['puf']
    
    def apply_heat_transfer(self, dt: float):
        """Aplica transferência de calor."""
        # Aplica ganhos de calor internos
        for zone_idx, gains in self.heat_gains.items():
            zone_cells = np.where(self.zone_map == zone_idx + 1)
            
            if len(zone_cells[0]) == 0:
                continue
            
            # Ganho total de calor por célula
            total_gain = gains['total']
            cell_count = len(zone_cells[0])
            gain_per_cell = total_gain / max(cell_count, 1)
            
            # Capacidade térmica do ar por célula
            cell_volume = (self.config.cell_size ** 2) * self.scenario.floor_height
            air_density = 1.204  # kg/m³
            specific_heat = 1005  # J/kg·K
            heat_capacity = cell_volume * air_density * specific_heat  # J/K
            
            # Aumento de temperatura
            delta_T = (gain_per_cell * dt) / heat_capacity
            self.grids['temperature'][zone_cells] += delta_T
        
        # Perda de calor pelas superfícies
        surface_cells = self._get_surface_cells()
        if len(surface_cells[0]) > 0:
            y_indices, x_indices = surface_cells
            
            # Coeficiente global de transferência de calor
            U = self.U_walls  # W/m²K
            
            # Área por célula de superfície
            cell_perimeter = 4 * self.config.cell_size
            surface_area_per_cell = self.scenario.floor_height * cell_perimeter
            
            for y, x in zip(y_indices, x_indices):
                temp_diff = self.grids['temperature'][y, x] - self.external_temperature
                
                # Perda de calor
                heat_loss = U * surface_area_per_cell * temp_diff * dt  # J
                
                # Redução de temperatura
                cell_volume = (self.config.cell_size ** 2) * self.scenario.floor_height
                air_density = 1.204  # kg/m³
                specific_heat = 1005  # J/kg·K
                heat_capacity = cell_volume * air_density * specific_heat
                
                delta_T_loss = heat_loss / heat_capacity
                self.grids['temperature'][y, x] -= delta_T_loss
                
                # Registra perda de calor
                self.sinks['heat_loss'][y, x] += heat_loss / dt  # W
    
    def apply_humidity_transfer(self, dt: float):
        """Aplica transferência de umidade."""
        # Ganhos de umidade de agentes já aplicados via add_agent_emission
        
        # Perda de umidade por ventilação já aplicada
        # Condensação em superfícies frias
        y_indices, x_indices = self._get_surface_cells()
        
        # Usa zip para iterar sobre pares de coordenadas
        for y, x in zip(y_indices, x_indices):
            temp = self.grids['temperature'][y, x] - 273.15  # °C
            humidity = self.grids['humidity'][y, x]
            
            # Temperatura do ponto de orvalho
            # Fórmula simplificada: Td = T - ((100 - RH)/5)
            dew_point = temp - ((100 - humidity * 100) / 5)
            
            # Se temperatura da superfície está abaixo do ponto de orvalho
            if temp < dew_point - 2:  # margem de segurança
                # Condensação
                condensation_rate = 1e-6 * dt  # kg/m²·s (simplificado)
                surface_area = self.scenario.floor_height * 4 * self.config.cell_size
                condensation_mass = condensation_rate * surface_area
                
                # Reduz umidade no ar
                cell_volume = (self.config.cell_size ** 2) * self.scenario.floor_height
                air_density = 1.204  # kg/m³
                humidity_reduction = condensation_mass / (cell_volume * air_density)
                
                self.grids['humidity'][y, x] = max(0.0, self.grids['humidity'][y, x] - humidity_reduction)
                self.sinks['condensation'][y, x] += condensation_mass
    
    def update_air_age(self, dt: float):
        """Atualiza a idade do ar em cada célula."""
        # Incrementa idade do ar
        self.grids['air_age'] += dt
        
        # Quando há ventilação, renova o ar (reduz idade)
        for zone_idx, zone in enumerate(self.scenario.zones):
            zone_cells = np.where(self.zone_map == zone_idx + 1)
            
            if len(zone_cells[0]) == 0:
                continue
            
            ach = self.kalman_filters.get(zone_idx, {}).get('ach_estimate', zone.target_ach)
            renewal_probability = ach * dt / 3600.0
            
            # Células que recebem ar novo
            renewal_mask = np.random.random(size=len(zone_cells[0])) < renewal_probability
            y_indices, x_indices = zone_cells
            
            if np.any(renewal_mask):
                self.grids['air_age'][y_indices[renewal_mask], x_indices[renewal_mask]] = 0.0
    
    def step(self, dt: float, current_time: float, agent_data: Optional[Dict] = None):
        """
        Executa um passo completo da física.
        
        Args:
            dt: Passo de tempo (s)
            current_time: Tempo atual da simulação (s)
            agent_data: Dados dos agentes para correções
        """
        # 0. Limpa fontes/sumidouros do passo anterior
        self._clear_sources_sinks()
        
        # 1. Aplica emissões de materiais
        self.apply_material_emissions(dt)
        
        # 2. Aplica fontes dos agentes (se fornecidas)
        if agent_data and 'emissions' in agent_data:
            for emission in agent_data['emissions']:
                # Detecta formato dos dados (granular ou agregado) e normaliza
                if 'amounts' in emission:
                    # Formato agregado
                    amounts = emission['amounts']
                    heat = emission.get('metabolic_heat', 0.0)
                    moisture = emission.get('moisture_production', 0.0)
                else:
                    # Formato granular (flat) vindo de advanced_agents.py
                    amounts = {}
                    heat = 0.0
                    moisture = 0.0
                    
                    sp = emission.get('species')
                    val = emission.get('amount', 0.0)
                    
                    if sp == 'heat':
                        heat = val
                    elif sp == 'moisture':
                        moisture = val
                    elif sp:
                        amounts[sp] = val
                        
                self.add_agent_emission(
                    emission['x'], 
                    emission['y'], 
                    amounts,
                    heat,
                    moisture
                )
        
        # Aplica fontes dos agentes aos grids
        self.apply_agent_sources()
        
        # 3. Advecção
        self.apply_advection(dt)
        
        # 4. Difusão
        self.apply_diffusion(dt)
        
        # 5. Transferência entre zonas
        self.apply_zone_transfer(dt)
        
        # 6. Transferência de calor
        self.apply_heat_transfer(dt)
        
        # 7. Transferência de umidade
        self.apply_humidity_transfer(dt)
        
        # 8. Deposição
        self.apply_deposition(dt)
        
        # 9. Ventilação (deve vir depois das fontes)
        self.apply_ventilation(dt)
        
        # 10. Atualiza filtros de Kalman
        self.update_kalman_filters(dt, current_time)
        
        # 11. Atualiza idade do ar
        self.update_air_age(dt)
        
        # 12. Aplica correção da pluma térmica
        if agent_data and 'positions' in agent_data and 'activities' in agent_data:
            self.apply_plume_thermal_correction(
                agent_data['positions'], 
                agent_data['activities']
            )
        
        # 13. Garante valores mínimos/máximos
        self._enforce_bounds()
        
        # 14. Registra histórico
        self._record_history(current_time)
    
    def _clear_sources_sinks(self):
        """Limpa fontes e sumidouros para o próximo passo."""
        for key in self.sources:
            self.sources[key].fill(0.0)
        for key in self.sinks:
            self.sinks[key].fill(0.0)
        
        # Reinicia ganhos de calor dos ocupantes
        for zone_idx in self.heat_gains:
            self.heat_gains[zone_idx]['occupants'] = 0.0
            self.heat_gains[zone_idx]['total'] = (
                self.heat_gains[zone_idx]['equipment'] + 
                self.heat_gains[zone_idx]['lighting']
            )
    
    def _enforce_bounds(self):
        """Garante que variáveis físicas estejam dentro de limites razoáveis."""
        # CO2: 400 ppm mínimo (exterior)
        self.grids['co2'] = np.maximum(self.grids['co2'], 
                                     400 * cfg.CONVERSION_FACTORS['co2_ppm_to_kgm3'])
        
        # HCHO: não negativo
        self.grids['hcho'] = np.maximum(self.grids['hcho'], 1e-15)
        
        # VOCs: não negativo
        self.grids['voc'] = np.maximum(self.grids['voc'], 1e-15)
        
        # Vírus: não negativo
        self.grids['virus'] = np.maximum(self.grids['virus'], 1e-20)
        
        # PM: não negativo
        self.grids['pm25'] = np.maximum(self.grids['pm25'], 1e-15)
        self.grids['pm10'] = np.maximum(self.grids['pm10'], 1e-15)
        
        # Temperatura: limites razoáveis (10°C a 40°C)
        self.grids['temperature'] = np.clip(self.grids['temperature'], 283.15, 313.15)
        
        # Umidade: 0% a 100%
        self.grids['humidity'] = np.clip(self.grids['humidity'], 0.0, 1.0)
        
        # Idade do ar: não negativa
        self.grids['air_age'] = np.maximum(self.grids['air_age'], 0.0)
        
        # PUF: pelo menos 1.0
        self.grids['puf'] = np.maximum(self.grids['puf'], 1.0)
    
    def _record_history(self, current_time: float):
        """Registra dados para histórico."""
        if current_time % 300 < 0.1:  # A cada 5 minutos
            self.history['time'].append(current_time)
            
            # Registra concentrações médias por zona
            zone_concentrations = {}
            for zone_idx in range(len(self.scenario.zones)):
                zone_cells = np.where(self.zone_map == zone_idx + 1)
                if len(zone_cells[0]) > 0:
                    zone_concentrations[zone_idx] = {
                        'co2_ppm': np.mean(self.grids['co2'][zone_cells]) * cfg.CONVERSION_FACTORS['co2_kgm3_to_ppm'],
                        'temperature_c': np.mean(self.grids['temperature'][zone_cells]) - 273.15,
                        'humidity_percent': np.mean(self.grids['humidity'][zone_cells]) * 100,
                        'ach_actual': self.kalman_filters.get(zone_idx, {}).get('ach_estimate', 
                                                                               self.scenario.zones[zone_idx].target_ach)
                    }
            self.history['zone_concentrations'].append(zone_concentrations)
            
            # Calcula consumo energético
            energy = self._calculate_energy_consumption()
            self.history['energy_consumption'].append(energy)
    
    def _calculate_energy_consumption(self) -> Dict[str, float]:
        """Calcula consumo energético atual."""
        total_energy = 0.0
        fan_energy = 0.0
        heating_cooling_energy = 0.0
        
        for zone_idx, zone in enumerate(self.scenario.zones):
            zone_cells = np.where(self.zone_map == zone_idx + 1)
            
            if len(zone_cells[0]) == 0:
                continue
            
            # Área e volume da zona
            area = len(zone_cells[0]) * (self.config.cell_size ** 2)
            volume = area * self.scenario.floor_height
            
            # Energia do ventilador
            ach = self.kalman_filters.get(zone_idx, {}).get('ach_estimate', zone.target_ach)
            airflow_m3s = ach * volume / 3600.0
            pressure_pa = 500  # Pressão típica do sistema
            fan_efficiency = 0.6
            
            fan_power = airflow_m3s * pressure_pa / fan_efficiency  # W
            fan_energy += fan_power
            
            # Energia para aquecimento/resfriamento
            temp_diff = np.mean(self.grids['temperature'][zone_cells]) - (self.scenario.temperature_setpoint + 273.15)
            heating_cooling_power = airflow_m3s * 1.204 * 1005 * abs(temp_diff)  # ρ * cp * ΔT
            
            # Eficiência do sistema HVAC (COP ou EER)
            hvac_efficiency = 3.0  # COP típico
            heating_cooling_energy += heating_cooling_power / hvac_efficiency
            
            total_energy += fan_power + heating_cooling_power / hvac_efficiency
        
        return {
            'total_power_w': total_energy,
            'fan_power_w': fan_energy,
            'hvac_power_w': heating_cooling_energy,
            'total_energy_kwh': total_energy * 0.001 / 3600  # kWh por passo (assumindo 1s dt)
        }
    
    def get_concentrations_at(self, x: int, y: int) -> Dict[str, float]:
        """Retorna concentrações em uma posição específica."""
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
        """Retorna estatísticas por zona."""
        stats = {}
        
        for zone_idx, zone in enumerate(self.scenario.zones):
            zone_cells = np.where(self.zone_map == zone_idx + 1)
            
            if len(zone_cells[0]) == 0:
                continue
            
            area = len(zone_cells[0]) * (self.config.cell_size ** 2)
            volume = area * self.scenario.floor_height
            
            stats[zone_idx] = {
                'name': zone.name,
                'type': zone.zone_type.value,
                'area_m2': area,
                'volume_m3': volume,
                'occupancy_density': zone.occupancy_density,
                'max_occupants': int(area * zone.occupancy_density),
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
                'ach_target': zone.target_ach,
                'ventilation_mode': zone.ventilation_mode.value,
                'has_windows': zone.has_windows,
                'window_area_ratio': zone.window_area_ratio,
                'heat_gain_w': self.heat_gains.get(zone_idx, {}).get('total', 0.0),
                'puf_mean': np.mean(self.grids['puf'][zone_cells]) if 'puf' in self.grids else 1.0
            }
        
        return stats
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """Prepara dados para visualização."""
        return {
            'grids': {
                'co2_ppm': self.grids['co2'] * cfg.CONVERSION_FACTORS['co2_kgm3_to_ppm'],
                'hcho_ppb': self.grids['hcho'] * cfg.CONVERSION_FACTORS['hcho_kgm3_to_ppb'],
                'voc_ppb': self.grids['voc'] * cfg.CONVERSION_FACTORS['voc_kgm3_to_ppb'],
                'virus_log': np.log10(self.grids['virus'] + 1e-20),
                'virus_exposure_log': np.log10(self.grids.get('virus_exposure', self.grids['virus']) + 1e-20),
                'pm25_ugm3': self.grids['pm25'] * 1e9,
                'pm10_ugm3': self.grids['pm10'] * 1e9,
                'temperature_c': self.grids['temperature'] - 273.15,
                'humidity_percent': self.grids['humidity'] * 100,
                'velocity_magnitude': np.sqrt(self.velocity_field[:,:,0]**2 + self.velocity_field[:,:,1]**2),
                'air_age_minutes': self.grids['air_age'] / 60.0,
                'puf_factor': self.grids['puf'] if 'puf' in self.grids else np.ones_like(self.grids['co2']),
                'zone_map': self.zone_map,
                'material_map': self.material_grid['type']
            },
            'velocity_field': {
                'x': self.velocity_field[:,:,0],
                'y': self.velocity_field[:,:,1],
                'magnitude': np.sqrt(self.velocity_field[:,:,0]**2 + self.velocity_field[:,:,1]**2)
            },
            'zone_stats': self.get_zone_statistics(),
            'heat_gains': self.heat_gains,
            'sources_sinks': {
                'sources_total': {k: np.sum(v) for k, v in self.sources.items()},
                'sinks_total': {k: np.sum(v) for k, v in self.sinks.items()}
            },
            'metadata': {
                'cells_x': self.cells_x,
                'cells_y': self.cells_y,
                'cell_size': self.config.cell_size,
                'total_width': self.scenario.total_width,
                'total_height': self.scenario.total_height,
                'floor_height': self.scenario.floor_height,
                'total_zones': len(self.scenario.zones)
            }
        }
    
    def set_external_conditions(self, temperature_c: float, humidity_percent: float, 
                                co2_ppm: float = 400):
        """Define condições externas."""
        self.external_temperature = temperature_c + 273.15
        self.external_humidity = humidity_percent / 100.0
        self.external_co2 = co2_ppm * cfg.CONVERSION_FACTORS['co2_ppm_to_kgm3']
    
    def get_energy_summary(self) -> Dict[str, float]:
        """Retorna resumo de consumo energético."""
        energy_data = self._calculate_energy_consumption()
        
        # Adiciona estatísticas adicionais
        total_volume = 0.0
        total_area = 0.0
        
        for zone_idx in range(len(self.scenario.zones)):
            zone_cells = np.where(self.zone_map == zone_idx + 1)
            if len(zone_cells[0]) > 0:
                area = len(zone_cells[0]) * (self.config.cell_size ** 2)
                total_area += area
                total_volume += area * self.scenario.floor_height
        
        energy_data.update({
            'total_area_m2': total_area,
            'total_volume_m3': total_volume,
            'energy_intensity_w_m2': energy_data['total_power_w'] / max(total_area, 0.1),
            'ventilation_intensity_lps_m2': sum(
                self.kalman_filters.get(i, {}).get('ach_estimate', zone.target_ach) * (zone.area_m2 if hasattr(zone, 'area_m2') else 0) * self.scenario.floor_height / 3.6
                for i, zone in enumerate(self.scenario.zones)
            ) / max(total_area, 0.1)
        })
        
        return energy_data
    
    def get_iaq_summary(self) -> Dict[str, Any]:
        """Retorna resumo da qualidade do ar interno."""
        zone_stats = self.get_zone_statistics()
        
        # Calcula índices de qualidade do ar por zona
        iaq_indices = {}
        for zone_idx, stats in zone_stats.items():
            concentrations = stats['concentrations']
            
            # Índice CO2 (0-100, onde 100 é melhor)
            co2_index = max(0, min(100, 100 - (concentrations['co2_ppm_mean'] - 400) / 8))
            
            # Índice HCHO (0-100)
            hcho_index = max(0, min(100, 100 - concentrations['hcho_ppb_mean'] / 0.818))
            
            # Índice térmico (0-100)
            temp_diff = abs(concentrations['temperature_c_mean'] - self.scenario.temperature_setpoint)
            temp_index = max(0, min(100, 100 - temp_diff * 10))
            
            # Índice de umidade (0-100)
            hum_diff = abs(concentrations['humidity_percent_mean'] - self.scenario.humidity_setpoint)
            hum_index = max(0, min(100, 100 - hum_diff * 2))
            
            # Índice geral de IAQ (média ponderada)
            iaq_index = 0.4 * co2_index + 0.2 * hcho_index + 0.2 * temp_index + 0.2 * hum_index
            
            iaq_indices[zone_idx] = {
                'zone_name': stats['name'],
                'iaq_index': iaq_index,
                'co2_index': co2_index,
                'hcho_index': hcho_index,
                'thermal_index': temp_index,
                'humidity_index': hum_index,
                'ventilation_adequacy': min(100, stats['ach_actual'] / max(stats['ach_target'], 0.1) * 100),
                'air_freshness': max(0, min(100, 100 - concentrations['air_age_minutes_mean'] * 2))
            }
        
        # IAQ global (média ponderada por volume)
        total_volume = sum(stats['volume_m3'] for stats in zone_stats.values())
        global_iaq = 0.0
        
        if total_volume > 0:
            for zone_idx, stats in zone_stats.items():
                weight = stats['volume_m3'] / total_volume
                global_iaq += iaq_indices[zone_idx]['iaq_index'] * weight
        
        return {
            'global_iaq_index': global_iaq,
            'zone_iaq_indices': iaq_indices,
            'worst_zone': min(iaq_indices.items(), key=lambda x: x[1]['iaq_index'])[0] if iaq_indices else None,
            'best_zone': max(iaq_indices.items(), key=lambda x: x[1]['iaq_index'])[0] if iaq_indices else None,
            'compliance': {
                'co2': all(s['concentrations']['co2_ppm_mean'] <= self.scenario.co2_setpoint 
                          for s in zone_stats.values()),
                'temperature': all(abs(s['concentrations']['temperature_c_mean'] - self.scenario.temperature_setpoint) <= 2.0
                                 for s in zone_stats.values()),
                'ventilation': all(s['ach_actual'] >= s['ach_target'] * 0.8 
                                  for s in zone_stats.values())
            }
        }
