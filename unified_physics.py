"""
MOTOR FÍSICO
Integra toda a física do modelo com suporte a obstáculos que bloqueiam difusão.
"""

import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d
from typing import Dict, List, Tuple, Optional, Any, Union
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
    - **BLOQUEIO DE DIFUSÃO POR OBSTÁCULOS** (CORRIGIDO)
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
        
        # Garante pelo menos 1 célula
        self.cells_x = max(1, self.cells_x)
        self.cells_y = max(1, self.cells_y)
        
        # ====================================================================
        # INICIALIZAÇÃO DE MÁSCARA DE OBSTÁCULOS
        # ====================================================================
        # A máscara é criada antes dos grids porque será usada na difusão
        self.obstacle_mask = self._create_obstacle_mask()
        
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
    
    # ========================================================================
    # CRIAÇÃO DA MÁSCARA DE OBSTÁCULOS
    # ========================================================================
    def _create_obstacle_mask(self) -> np.ndarray:
        """
        Cria máscara de obstáculos para bloqueio de difusão.
        Aceita tanto objetos Obstacle quanto dicionários.
        
        Returns:
            np.ndarray: Matriz (cells_y, cells_x) onde:
                - 1.0 = Ar livre (difusão permitida)
                - 0.0 = Parede sólida (bloqueia difusão totalmente)
                - valor entre 0-1 = Porosidade (móveis, divisórias porosas)
        """
        # Inicializa tudo como ar livre
        mask = np.ones((self.cells_y, self.cells_x), dtype=np.float32)
        
        # Processa cada obstáculo
        for obstacle in self.scenario.obstacles:
            # ================================================================
            # SUPORTE HÍBRIDO: Detecta se é objeto ou dict
            # ================================================================
            if isinstance(obstacle, dict):
                # Formato legado (dict)
                x_start_m = obstacle.get('x', 0.0)
                y_start_m = obstacle.get('y', 0.0)
                width_m = obstacle.get('width', 1.0)
                height_m = obstacle.get('height', 1.0)
                obstacle_type_str = obstacle.get('type', 'furniture')
                porosity = obstacle.get('porosity', 0.7)
                
                # Converte string para enum se necessário
                try:
                    obstacle_type = cfg.ObstacleType(obstacle_type_str)
                except (ValueError, AttributeError):
                    obstacle_type = cfg.ObstacleType.FURNITURE
            else:
                # Formato novo (objeto Obstacle)
                x_start_m = obstacle.x
                y_start_m = obstacle.y
                width_m = obstacle.width
                height_m = obstacle.height
                obstacle_type = obstacle.obstacle_type
                porosity = obstacle.porosity
            
            # Calcula coordenadas finais
            x_end_m = x_start_m + width_m
            y_end_m = y_start_m + height_m
            
            # Converte para índices de células (arredondamento seguro)
            x_start = int(np.floor(x_start_m / self.config.cell_size))
            y_start = int(np.floor(y_start_m / self.config.cell_size))
            x_end = int(np.ceil(x_end_m / self.config.cell_size))
            y_end = int(np.ceil(y_end_m / self.config.cell_size))
            
            # ================================================================
            # GARANTE LIMITES DENTRO DO GRID (ROBUSTEZ CRÍTICA)
            # ================================================================
            x_start = max(0, min(x_start, self.cells_x - 1))
            x_end = max(0, min(x_end, self.cells_x))
            y_start = max(0, min(y_start, self.cells_y - 1))
            y_end = max(0, min(y_end, self.cells_y))
            
            # Verifica se há área válida
            if x_end <= x_start or y_end <= y_start:
                continue  # Obstáculo fora do grid ou área zero
            
            # ================================================================
            # APLICA POROSIDADE BASEADA NO TIPO
            # ================================================================
            if obstacle_type == cfg.ObstacleType.WALL:
                # WALL: Bloqueio total (0.0 = sem difusão)
                mask[y_start:y_end, x_start:x_end] = 0.0
            elif obstacle_type == cfg.ObstacleType.FURNITURE:
                # FURNITURE: Permite difusão parcial (70-90%)
                mask[y_start:y_end, x_start:x_end] = max(0.7, porosity)
            elif obstacle_type == cfg.ObstacleType.PARTITION:
                # PARTITION: Usa porosidade direta
                mask[y_start:y_end, x_start:x_end] = porosity
            elif obstacle_type == cfg.ObstacleType.EQUIPMENT:
                # EQUIPMENT: Moderadamente permeável
                mask[y_start:y_end, x_start:x_end] = max(0.5, porosity)
            else:
                # Tipo desconhecido: assume móvel
                mask[y_start:y_end, x_start:x_end] = max(0.7, porosity)
        
        return mask
    
    def _initialize_grids(self) -> Dict[str, np.ndarray]:
        """Inicializa todos os grids de concentração."""
        shape = (self.cells_y, self.cells_x)
        
        grids = {
            'co2': np.ones(shape) * self.external_co2,
            'hcho': np.ones(shape) * 10 * cfg.CONVERSION_FACTORS['hcho_ppb_to_kgm3'],
            'voc': np.ones(shape) * 50 * cfg.CONVERSION_FACTORS.get('voc_ppb_to_kgm3', 1.5e-9) * 50,
            'virus': np.ones(shape) * 1e-12,  # quanta/m³
            'pm25': np.ones(shape) * 5e-9,    # kg/m³ (~5 µg/m³)
            'pm10': np.ones(shape) * 10e-9,   # kg/m³ (~10 µg/m³)
            'temperature': np.ones(shape) * (self.scenario.temperature_setpoint + 273.15),  # K
            'humidity': np.ones(shape) * (self.scenario.humidity_setpoint / 100.0),  # fração (0-1)
            'air_age': np.zeros(shape),  # idade do ar em segundos
            'puf': np.ones(shape),  # fator de utilização da pluma
        }
        
        # ====================================================================
        # APLICA MÁSCARA DE OBSTÁCULOS NAS CONCENTRAÇÕES INICIAIS
        # ====================================================================
        # Células de parede começam com concentração zero (não há ar lá)
        for species in ['co2', 'hcho', 'voc', 'virus', 'pm25', 'pm10']:
            grids[species] *= self.obstacle_mask
        
        return grids
    
    def _create_zone_map(self) -> np.ndarray:
        """Cria mapa de zonas baseado no cenário."""
        zone_map = np.zeros((self.cells_y, self.cells_x), dtype=np.int16)
        
        for zone_idx, zone in enumerate(self.scenario.zones):
            # Calcula limites em células (normalizado para grid)
            x_start = int((zone.x_start / self.scenario.total_width) * self.cells_x)
            x_end = int((zone.x_end / self.scenario.total_width) * self.cells_x)
            y_start = int((zone.y_start / self.scenario.total_height) * self.cells_y)
            y_end = int((zone.y_end / self.scenario.total_height) * self.cells_y)
            
            # Garante limites dentro do grid
            x_start = max(0, min(x_start, self.cells_x - 1))
            x_end = max(0, min(x_end, self.cells_x))
            y_start = max(0, min(y_start, self.cells_y - 1))
            y_end = max(0, min(y_end, self.cells_y))
            
            # Verifica se há área válida
            if x_end <= x_start or y_end <= y_start:
                continue
            
            # Aplica zona (índice começa em 1, 0 = fora de qualquer zona)
            zone_map[y_start:y_end, x_start:x_end] = zone_idx + 1
        
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
                material_type = material_config.get('type')
                
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
        total_area = len(cells[0]) * cell_area
        
        # Fator por tipo de superfície
        surface_factors = {
            'walls': 2.5,      # paredes verticais
            'ceiling': 1.0,    # teto
            'floor': 1.0,      # piso
            'furniture': 3.0   # móveis (maior área por volume)
        }
        
        factor = surface_factors.get(surface_type, 1.0)
        return total_area * factor * density
    
    def _calculate_diffusion_coeffs(self) -> Dict[str, float]:
        """Calcula coeficientes de difusão efetiva (molecular + turbulenta)."""
        # Difusão turbulenta depende do ACH médio
        avg_ach = np.mean([zone.target_ach for zone in self.scenario.zones])
        
        if avg_ach < 2.0:
            turbulent_D = self.config.turbulent_diffusion_low_vent
        elif avg_ach < 6.0:
            turbulent_D = self.config.turbulent_diffusion_medium_vent
        else:
            turbulent_D = self.config.turbulent_diffusion_high_vent
        
        return {
            'co2': self.config.molecular_diffusion_co2 + turbulent_D,
            'hcho': self.config.molecular_diffusion_hcho + turbulent_D,
            'voc': self.config.molecular_diffusion_voc + turbulent_D,
            'virus': self.config.molecular_diffusion_virus + turbulent_D * 0.5,  # partículas maiores
            'pm25': self.config.molecular_diffusion_pm25 + turbulent_D * 0.3,
            'pm10': self.config.molecular_diffusion_pm10 + turbulent_D * 0.2,
            'temperature': self.config.thermal_diffusivity + turbulent_D,
            'humidity': self.config.moisture_diffusivity + turbulent_D
        }
    
    # ========================================================================
    # INICIALIZAÇÃO DO CAMPO DE VELOCIDADE
    # ========================================================================
    def _initialize_velocity_field(self):
        """
        Inicializa campo de velocidade baseado APENAS nas zonas definidas.
        """
        # Cria campo simplificado com padrão de circulação
        for zone_idx, zone in enumerate(self.scenario.zones):
            zone_cells = np.where(self.zone_map == zone_idx + 1)
            
            if len(zone_cells[0]) == 0:
                continue
            
            # Velocidade característica baseada no ACH
            ach = zone.target_ach
            room_height = self.scenario.ceiling_height
            characteristic_velocity = ach * room_height / 3600.0  # m/s
            
            # Padrão simplificado: entrada no topo, saída no fundo
            # Velocidade vertical descendente (mixing suave)
            self.velocity_field[zone_cells[0], zone_cells[1], 1] = -characteristic_velocity * 0.3
        
        # ====================================================================
        # CONDIÇÕES DE CONTORNO (sem fluxo nas bordas)
        # ====================================================================
        self.velocity_field[0, :, :] = 0
        self.velocity_field[-1, :, :] = 0
        self.velocity_field[:, 0, :] = 0
        self.velocity_field[:, -1, :] = 0
    
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
            zone_volume = len(zone_cells[0]) * (self.config.cell_size ** 2) * self.scenario.ceiling_height
            
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
    
    def add_agent_emission(self, x: int, y: int, emissions: Dict[str, float], 
                          metabolic_heat: float = 0.0, moisture_production: float = 0.0):
        """
        Adiciona emissões de um agente às fontes do motor físico.
        """
        # VALIDAÇÃO DE LIMITES
        if not (0 <= x < self.cells_x and 0 <= y < self.cells_y):
            # Fora dos limites do grid - ignora silenciosamente
            return
        
        # VERIFICAÇÃO DE OBSTÁCULOS
        # Não permite emissão em paredes sólidas (obstacle_mask < 0.5)
        if self.obstacle_mask[y, x] < 0.5:
            return
        
        # CÁLCULO DO VOLUME DA CÉLULA
        cell_area = self.config.cell_size ** 2  # m²
        cell_volume = cell_area * self.scenario.ceiling_height  # m³
        
        # PROCESSAMENTO DE EMISSÕES POR ESPÉCIE
        for species, amount in emissions.items():
            if amount <= 0:
                continue
            
            # Verifica se a espécie é suportada
            if species not in self.sources:
                continue
            
            concentration = amount / cell_volume
            
            # Adiciona à fonte (acumula se múltiplos agentes na mesma célula)
            self.sources[species][y, x] += concentration
            
            # DEBUG: Log para emissões virais significativas
            if species == 'virus' and concentration > 1e-9:
                pass
            
        # PROCESSAMENTO DE CALOR METABÓLICO E DE UMIDADE
        if metabolic_heat > 0:
            # Calor em Watts (W) - será usado para atualizar temperatura
            self.sources['heat'][y, x] += metabolic_heat
        if moisture_production > 0:
            # Umidade em kg/s - converte para concentração de vapor
            # Massa de vapor por volume de ar
            vapor_concentration = moisture_production / cell_volume  # kg/(m³·s)
            self.sources['moisture'][y, x] += vapor_concentration
    
    def apply_agent_sources(self):
            """
            Aplica as fontes acumuladas (emissions) aos grids de concentração.
            Chamado a cada passo de tempo antes da difusão.
            """
            dt = getattr(self, 'current_dt', 1.0)  # Usa dt atual ou default 1s
            
            for species in ['co2', 'virus', 'hcho', 'voc', 'pm25', 'pm10']:
                if species in self.sources and species in self.grids:
                    self.grids[species] += self.sources[species] * dt
                    self.grids[species] = np.maximum(self.grids[species], 0.0)
    
    def apply_material_emissions(self, dt: float):
        """Aplica emissões de materiais de construção."""
        cell_volume = self.config.cell_size ** 2 * self.scenario.ceiling_height
        
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
                height = self.scenario.ceiling_height
                
                # Área das 6 faces da célula
                A = 2 * (cell_size * cell_size) + 4 * (cell_size * height)
                V = cell_size * cell_size * height
                av_ratio = A / V
                
                deposition_rate = v_d * av_ratio * dt
                
                # Aplica em todas as células (simplificado)
                loss = self.grids[species] * deposition_rate
                self.grids[species] -= loss
                
                # Registra no sink
                self.sinks['deposition'] += loss
    
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
            return 20 * cfg.CONVERSION_FACTORS.get('voc_ppb_to_kgm3', 1.5e-9) * 20  # 20 ppb externo
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
        # Primeiro componente x, depois y
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
                dx = np.clip(velocity[mask_right] * dt / self.config.cell_size, 0, 1)
                grid[mask_right] = (1 - dx) * grid[mask_right] + dx * shifted_right[mask_right]
            
            # Para velocidade negativa (esquerda)
            mask_left = velocity < 0
            if np.any(mask_left):
                shifted_left = np.roll(grid, 1, axis=1)
                dx = np.clip(-velocity[mask_left] * dt / self.config.cell_size, 0, 1)
                grid[mask_left] = (1 - dx) * grid[mask_left] + dx * shifted_left[mask_left]
        
        else:  # y direction
            # Para velocidade positiva (baixo)
            mask_down = velocity > 0
            if np.any(mask_down):
                shifted_down = np.roll(grid, -1, axis=0)
                dy = np.clip(velocity[mask_down] * dt / self.config.cell_size, 0, 1)
                grid[mask_down] = (1 - dy) * grid[mask_down] + dy * shifted_down[mask_down]
            
            # Para velocidade negativa (cima)
            mask_up = velocity < 0
            if np.any(mask_up):
                shifted_up = np.roll(grid, 1, axis=0)
                dy = np.clip(-velocity[mask_up] * dt / self.config.cell_size, 0, 1)
                grid[mask_up] = (1 - dy) * grid[mask_up] + dy * shifted_up[mask_up]
        
        self.grids[species] = grid
    
    # ========================================================================
    # MÉTODO DE DIFUSÃO COM BLOQUEIO DE OBSTÁCULOS
    # ========================================================================
    def apply_diffusion(self, dt: float):
        """
        Aplica difusão molecular e turbulenta COM bloqueio por obstáculos.
        
        A máscara de obstáculos impede difusão através de paredes.
        - obstacle_mask = 0.0 → Parede sólida (bloqueia 100%)
        - obstacle_mask = 1.0 → Ar livre (difusão normal)
        - 0.0 < obstacle_mask < 1.0 → Difusão parcial (móveis, divisórias)
        """
        # Kernel Laplaciano padrão (5 pontos)
        kernel = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]], dtype=np.float32) / (self.config.cell_size ** 2)
        
        for species in ['co2', 'hcho', 'virus', 'voc', 'pm25', 'pm10', 'temperature', 'humidity']:
            if species in self.diffusion_coeffs:
                D = self.diffusion_coeffs[species]
                grid = self.grids[species]
                
                # ============================================================
                # PASSO 1: ZERA CONCENTRAÇÃO EM PAREDES
                # ============================================================
                # Paredes não contêm ar, logo concentração deve ser zero
                grid_masked = grid * self.obstacle_mask
                
                # ============================================================
                # PASSO 2: CALCULA LAPLACIANO (gradiente de concentração)
                # ============================================================
                laplacian = convolve2d(grid_masked, kernel, mode='same', boundary='fill', fillvalue=0)
                
                # ============================================================
                # PASSO 3: APLICA MÁSCARA NO LAPLACIANO
                # ============================================================
                # Isso impede fluxo difusivo entrando ou saindo de células bloqueadas
                laplacian_masked = laplacian * self.obstacle_mask
                
                # ============================================================
                # PASSO 4: ATUALIZA CONCENTRAÇÃO (Euler explícito)
                # ============================================================
                self.grids[species] = grid_masked + D * laplacian_masked * dt
                
                # ============================================================
                # PASSO 5: GARANTIA FINAL - ZERA CÉLULAS BLOQUEADAS
                # ============================================================
                # Segurança dupla: garante que paredes permanecem sem concentração
                self.grids[species] *= self.obstacle_mask
                
                # ============================================================
                # PASSO 6: EVITA VALORES NEGATIVOS (estabilidade numérica)
                # ============================================================
                self.grids[species] = np.maximum(self.grids[species], 0.0)
    
    def apply_zone_transfer(self, dt: float):
        """Aplica transferência de massa entre zonas conectadas."""
        for connection in self.scenario.connections:
            zone_a_idx = connection['zone_a_id'] - 1
            zone_b_idx = connection['zone_b_id'] - 1
            flow_rate = connection['flow_rate']  # m³/s
            
            # Identifica células das duas zonas
            cells_a = np.where(self.zone_map == zone_a_idx + 1)
            cells_b = np.where(self.zone_map == zone_b_idx + 1)
            
            if len(cells_a[0]) == 0 or len(cells_b[0]) == 0:
                continue
            
            # Calcula volumes
            volume_a = len(cells_a[0]) * (self.config.cell_size ** 2) * self.scenario.ceiling_height
            volume_b = len(cells_b[0]) * (self.config.cell_size ** 2) * self.scenario.ceiling_height
            
            # Fração trocada neste dt
            frac_a = min(flow_rate * dt / volume_a, 0.5)
            frac_b = min(flow_rate * dt / volume_b, 0.5)
            
            # Troca de massa para cada espécie
            for species in ['co2', 'hcho', 'virus', 'voc', 'pm25', 'pm10']:
                if species in self.grids:
                    mean_a = np.mean(self.grids[species][cells_a])
                    mean_b = np.mean(self.grids[species][cells_b])
                    
                    # Transferência de A para B
                    delta_ab = (mean_a - mean_b) * frac_a
                    self.grids[species][cells_a] -= delta_ab
                    self.grids[species][cells_b] += delta_ab * (volume_a / volume_b)
    
    def apply_thermal_transfer(self, dt: float):
        """Aplica transferência de calor entre zonas e com ambiente externo."""
        for zone_idx, zone in enumerate(self.scenario.zones):
            zone_cells = np.where(self.zone_map == zone_idx + 1)
            
            if len(zone_cells[0]) == 0:
                continue
            
            # Calcula temperatura média da zona
            temp_zone = np.mean(self.grids['temperature'][zone_cells])
            
            # Perda de calor pelas paredes (simplificado)
            zone_area = len(zone_cells[0]) * (self.config.cell_size ** 2)
            wall_area = 2 * np.sqrt(zone_area) * self.scenario.ceiling_height
            
            heat_loss = self.U_walls * wall_area * (temp_zone - self.external_temperature)  # W
            
            # Atualiza temperatura (considerando capacidade térmica do ar)
            zone_volume = zone_area * self.scenario.ceiling_height
            air_mass = zone_volume * cfg.PHYSICAL_CONSTANTS['air_density']
            specific_heat = cfg.PHYSICAL_CONSTANTS['air_specific_heat']
            
            dT = -heat_loss * dt / (air_mass * specific_heat)
            self.grids['temperature'][zone_cells] += dT
            
            # Registra perda
            self.sinks['heat_loss'][zone_cells] += heat_loss * dt
    
    def apply_human_plume(self, agent_positions: List[Tuple[int, int]], dt: float):
        """Aplica pluma térmica e de respiração humana."""
        for x, y in agent_positions:
            if not (0 <= x < self.cells_x and 0 <= y < self.cells_y):
                continue
            
            # Define região de pluma (3x3 células ao redor do agente)
            x_min = max(0, x - 1)
            x_max = min(self.cells_x, x + 2)
            y_min = max(0, y - 1)
            y_max = min(self.cells_y, y + 2)
            
            # Cria arrays para cálculo gaussiano
            x_range = np.arange(x_min, x_max)
            y_range = np.arange(y_min, y_max)
            
            if len(x_range) == 0 or len(y_range) == 0:
                continue
            
            # Fator de pluma gaussiano simplificado
            xx, yy = np.meshgrid(x_range - x, y_range - y)
            plume_factor = np.exp(-(xx**2 + yy**2) / 2.0)
            
            # Aplica pluma ao PUF
            self.grids['puf'][y_min:y_max, x_min:x_max] = np.maximum(
                self.grids['puf'][y_min:y_max, x_min:x_max],
                plume_factor
            )
    
    def update_kalman_filters(self, dt: float):
        """Atualiza filtros de Kalman para estimativa de estado."""
        if not self.config.kalman_enabled:
            return
        
        for zone_idx, kf in self.kalman_filters.items():
            zone_cells = np.where(self.zone_map == zone_idx + 1)
            
            if len(zone_cells[0]) == 0:
                continue
            
            # Medidas atuais
            measured_co2 = np.mean(self.grids['co2'][zone_cells])
            measured_temp = np.mean(self.grids['temperature'][zone_cells])
            measured_hum = np.mean(self.grids['humidity'][zone_cells])
            
            measurement = np.array([measured_co2, measured_temp, measured_hum])
            
            # Predição
            # Modelo simples: estado permanece (sem dinâmica forte)
            predicted_state = kf['state']
            predicted_cov = kf['error_covariance'] + np.eye(3) * kf['process_noise']
            
            # Atualização (correção de Kalman)
            innovation = measurement - predicted_state
            S = predicted_cov + np.eye(3) * kf['measurement_noise']
            K = predicted_cov @ np.linalg.inv(S)  # Ganho de Kalman
            
            # Estado atualizado
            kf['state'] = predicted_state + K @ innovation
            kf['error_covariance'] = (np.eye(3) - K) @ predicted_cov
            
            # Extrai estimativas
            kf['co2_estimate'] = kf['state'][0]
            kf['temperature_estimate'] = kf['state'][1]
            kf['humidity_estimate'] = kf['state'][2]
            
    def get_concentrations_at(self, x: int, y: int) -> Dict[str, Any]:
        """
        Retorna concentrações de todas as espécies em uma célula específica.
        
        Args:
            x: Coordenada X (célula)
            y: Coordenada Y (célula)
            
        Returns:
            Dicionário com concentrações ou dict vazio se fora dos limites
        """
        # Verifica limites
        if not (0 <= x < self.cells_x and 0 <= y < self.cells_y):
            return {}
        
        # Verifica se é obstáculo sólido
        if self.obstacle_mask[y, x] < 0.5:
            return {
                'co2_ppm': 400.0,
                'hcho_ppb': 0.0,
                'virus_quanta_m3': 0.0,
                'temperature_c': 20.0,
                'humidity_percent': 50.0,
                'air_velocity_ms': 0.0,
                'air_age_minutes': 0.0,
                'puf_factor': 1.0,
                'is_obstacle': True
            }
        
        # Coleta concentrações do grid
        result = {
            'co2_ppm': float(self.grids['co2'][y, x] * cfg.CONVERSION_FACTORS['co2_kgm3_to_ppm']),
            'hcho_ppb': float(self.grids['hcho'][y, x] * cfg.CONVERSION_FACTORS['hcho_kgm3_to_ppb']),
            'voc_ppb': float(self.grids['voc'][y, x] * cfg.CONVERSION_FACTORS.get('voc_kgm3_to_ppb', 6.67e8)),
            'virus_quanta_m3': float(self.grids['virus'][y, x]),
            'pm25_ug_m3': float(self.grids['pm25'][y, x] * 1e9),
            'pm10_ug_m3': float(self.grids['pm10'][y, x] * 1e9),
            'temperature_c': float(self.grids['temperature'][y, x] - 273.15),
            'humidity_percent': float(self.grids['humidity'][y, x] * 100),
            'air_age_minutes': float(self.grids['air_age'][y, x] / 60.0),
            'puf_factor': float(self.grids['puf'][y, x]),
            'is_obstacle': False
        }
        
        # Campos adicionais esperados pelos agentes
        result['co2_kgm3'] = float(self.grids['co2'][y, x])
        result['hcho_kgm3'] = float(self.grids['hcho'][y, x])
        result['virus_exposure_quanta_m3'] = float(self.grids['virus'][y, x])
        
        return result
    
    def step(self, dt: float, current_time: float, agent_data: Dict[str, Any]):
        """
        Execução de um passo de tempo da física.
        
        Args:
            dt: Passo de tempo (segundos)
            current_time: Tempo simulado atual (segundos)
            agent_data: Dicionário com:
                - 'emissions': lista de {'x': int, 'y': int, 'species': {...}, 'heat': float, 'moisture': float}
                - 'positions': lista de (x, y)
                - 'activities': lista de strings de atividade
        """
        # Guarda dt para uso em outros métodos
        self.current_dt = dt
        
        # ================================================================
        # 1. REINICIALIZA FONTES
        # ================================================================
        for key in self.sources:
            self.sources[key].fill(0.0)
        
        # ================================================================
        # 2. PROCESSA EMISSÕES DOS AGENTES
        # ================================================================
        emissions_list = agent_data.get('emissions', [])
        
        for emission in emissions_list:
            # Extração segura com valores padrão
            x = emission.get('x')
            y = emission.get('y')
            species_dict = emission.get('species', {})
            heat = emission.get('heat', 0.0)
            moisture = emission.get('moisture', 0.0)
            
            # Validação de tipos
            if x is None or y is None:
                continue
            
            try:
                x = int(x)
                y = int(y)
                heat = float(heat) if heat is not None else 0.0
                moisture = float(moisture) if moisture is not None else 0.0
            except (ValueError, TypeError):
                continue  # Pula emissões com coordenadas inválidas
            
            # Valida espécies - garante que são números, não strings
            validated_species = {}
            for sp, val in species_dict.items():
                if sp not in ['co2', 'virus', 'hcho', 'voc', 'pm25', 'pm10']:
                    continue  # Ignora espécies desconhecidas
                try:
                    val_float = float(val) if val is not None else 0.0
                    if val_float > 0:
                        validated_species[sp] = val_float
                except (ValueError, TypeError):
                    continue  # Ignora valores não-numéricos
            
            # Só processa se houver algo válido
            if validated_species or heat > 0 or moisture > 0:
                self.add_agent_emission(x, y, validated_species, heat, moisture)
        
        # ================================================================
        # 3. APLICA FONTES AOS GRIDS
        # ================================================================
        self.apply_agent_sources()
        
        # ================================================================
        # 4. PROCESSOS FÍSICOS RESTANTES
        # ================================================================
        # Emissões de materiais
        self.apply_material_emissions(dt)
        
        # Transporte
        self.apply_diffusion(dt)
        self.apply_advection(dt)
        
        # Ventilação e transferência
        self.apply_ventilation(dt)
        self.apply_zone_transfer(dt)
        
        # Remoção
        self.apply_deposition(dt)
        
        # Térmico
        self.apply_thermal_transfer(dt)
        
        # Pluma humana
        positions = agent_data.get('positions', [])
        valid_positions = [p for p in positions if p is not None and len(p) == 2]
        if valid_positions:
            # Converte para índices de grid se necessário
            grid_positions = []
            for pos in valid_positions:
                try:
                    px, py = int(pos[0]), int(pos[1])
                    if 0 <= px < self.cells_x and 0 <= py < self.cells_y:
                        grid_positions.append((px, py))
                except (ValueError, TypeError, IndexError):
                    continue
            self.apply_human_plume(grid_positions, dt)
        
        # Filtros de Kalman
        self.update_kalman_filters(dt)
        
        # ================================================================
        # 5. REGISTRO DE HISTÓRICO
        # ================================================================
        if int(current_time) % 60 == 0:  # A cada minuto
            self.history['time'].append(current_time)
            self.history['zone_concentrations'].append(self.get_zone_statistics())
    
    def get_zone_statistics(self) -> Dict[int, Dict[str, Any]]:
        """Retorna estatísticas por zona."""
        stats = {}
        
        for zone_idx, zone in enumerate(self.scenario.zones):
            zone_cells = np.where(self.zone_map == zone_idx + 1)
            
            if len(zone_cells[0]) == 0:
                continue
            
            stats[zone_idx] = {
                'zone_id': zone_idx + 1,
                'zone_name': zone.name,
                'concentrations': {
                    'co2_ppm_mean': float(np.mean(self.grids['co2'][zone_cells]) * cfg.CONVERSION_FACTORS['co2_kgm3_to_ppm']),
                    'co2_ppm_max': float(np.max(self.grids['co2'][zone_cells]) * cfg.CONVERSION_FACTORS['co2_kgm3_to_ppm']),
                    'hcho_ppb_mean': float(np.mean(self.grids['hcho'][zone_cells]) * cfg.CONVERSION_FACTORS['hcho_kgm3_to_ppb']),
                    'hcho_ppb_max': float(np.max(self.grids['hcho'][zone_cells]) * cfg.CONVERSION_FACTORS['hcho_kgm3_to_ppb']),
                    'virus_quanta_mean': float(np.mean(self.grids['virus'][zone_cells])),
                    'virus_quanta_max': float(np.max(self.grids['virus'][zone_cells])),
                    'temperature_c_mean': float(np.mean(self.grids['temperature'][zone_cells]) - 273.15),
                    'humidity_percent_mean': float(np.mean(self.grids['humidity'][zone_cells]) * 100)
                },
                'heat_gain_total': self.heat_gains.get(zone_idx, {}).get('total', 0.0)
            }
        
        return stats
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """Prepara dados para visualização."""
        return {
            'grids': {
                'co2_ppm': (self.grids['co2'] * cfg.CONVERSION_FACTORS['co2_kgm3_to_ppm']).tolist(),
                'hcho_ppb': (self.grids['hcho'] * cfg.CONVERSION_FACTORS['hcho_kgm3_to_ppb']).tolist(),
                'virus_quanta': self.grids['virus'].tolist(),
                'temperature_c': (self.grids['temperature'] - 273.15).tolist(),
                'humidity_percent': (self.grids['humidity'] * 100).tolist(),
                'obstacle_mask': self.obstacle_mask.tolist(),
            },
            'zone_map': self.zone_map.tolist(),
            'velocity_field': {
                'u': self.velocity_field[:, :, 0].tolist(),
                'v': self.velocity_field[:, :, 1].tolist()
            }
        }
