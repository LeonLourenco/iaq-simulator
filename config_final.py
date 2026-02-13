"""
Configuração Científica para Simulador de Qualidade do Ar Interno (IAQ)
Baseado em ASHRAE 62.1, Harvard Healthy Buildings e Buonanno et al.

Este módulo define cenários precisos para simulação de ambientes indoor com
parâmetros validados por pesquisa bibliográfica, incluindo suporte completo
a obstáculos físicos e múltiplas zonas.

Referências:
- ASHRAE 62.1-2019: Ventilation for Acceptable Indoor Air Quality
- Harvard Healthy Buildings (2021): Indoor Air Quality Guidelines
- Buonanno et al. (2020): Estimation of airborne viral emission
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum


# ============================================================================
# ENUMERAÇÕES
# ============================================================================

class BuildingType(str, Enum):
    """Tipos de edificação suportados."""
    SCHOOL = "school"
    OFFICE = "office"
    GYM = "gym"
    HOSPITAL = "hospital"
    RESIDENTIAL = "residential"
    CUSTOM = "custom"


class ActivityLevel(Enum):
    """
    Níveis de atividade metabólica.
    Baseado em ASHRAE Fundamentals.
    """
    SEDENTARY = "sedentary"      # 1.0 met (escritório)
    LIGHT = "light"               # 1.6 met (escola, caminhando)
    MODERATE = "moderate"         # 3.0 met (exercício leve)
    HEAVY = "heavy"               # 6.0 met (crossfit, pesos)
    VERY_HEAVY = "very_heavy"     # 8.0+ met (sprint, HIIT)


class VentilationType(Enum):
    """Tipos de ventilação."""
    NATURAL = "natural"
    MECHANICAL = "mechanical"
    MIXED_MODE = "mixed_mode"
    DISPLACEMENT = "displacement"


class ObstacleType(str, Enum):
    """Tipos de obstáculos físicos."""
    WALL = "wall"
    FURNITURE = "furniture"
    PARTITION = "partition"
    EQUIPMENT = "equipment"


class MaterialType(str, Enum):
    """Tipos de materiais de construção."""
    MDF = "mdf"
    LATEX_PAINT = "latex_paint"
    PLYWOOD = "plywood"
    CARPET = "carpet"
    VINYL_FLOORING = "vinyl_flooring"
    CONCRETE = "concrete"
    DRYWALL = "drywall"
    GLASS = "glass"
    METAL = "metal"
    FABRIC = "fabric"
    WOOD = "wood"
    CERAMIC = "ceramic"
    MARBLE = "marble"
    PLASTIC = "plastic"
    RUBBER = "rubber"


# ============================================================================
# CONFIGURAÇÕES DE FÍSICA
# ============================================================================

@dataclass
class PhysicsConfig:
    """
    Configuração de parâmetros físicos da simulação.
    
    Attributes:
        cell_size: Tamanho da célula do grid (metros)
        dt_max: Passo de tempo máximo (segundos)
        stability_safety_factor: Fator de segurança para estabilidade CFL
        molecular_diffusion_co2: Difusividade molecular CO2 (m²/s)
        molecular_diffusion_hcho: Difusividade molecular HCHO (m²/s)
        molecular_diffusion_voc: Difusividade molecular VOC (m²/s)
        molecular_diffusion_virus: Difusividade molecular vírus (m²/s)
        molecular_diffusion_pm25: Difusividade molecular PM2.5 (m²/s)
        molecular_diffusion_pm10: Difusividade molecular PM10 (m²/s)
        turbulent_diffusion_low_vent: Difusividade turbulenta baixa ventilação (m²/s)
        turbulent_diffusion_medium_vent: Difusividade turbulenta média ventilação (m²/s)
        turbulent_diffusion_high_vent: Difusividade turbulenta alta ventilação (m²/s)
        thermal_diffusivity: Difusividade térmica (m²/s)
        moisture_diffusivity: Difusividade de umidade (m²/s)
        deposition_velocity_virus: Velocidade de deposição vírus (m/s)
        deposition_velocity_pm25: Velocidade de deposição PM2.5 (m/s)
        deposition_velocity_pm10: Velocidade de deposição PM10 (m/s)
        hcho_decay_rate: Taxa de decaimento químico HCHO (1/s)
        voc_oxidation_rate: Taxa de oxidação VOC (1/s)
        kalman_enabled: Habilitar filtros de Kalman
        kalman_process_noise: Ruído de processo Kalman
        kalman_measurement_noise: Ruído de medição Kalman
        diffusion_coefficient_co2: Coeficiente total de difusão CO2 (compatibilidade)
    """
    cell_size: float = 0.5  # metros
    dt_max: float = 1.0  # segundos
    stability_safety_factor: float = 0.9
    
    # Difusividades moleculares (m²/s)
    molecular_diffusion_co2: float = 1.6e-5
    molecular_diffusion_hcho: float = 1.8e-5
    molecular_diffusion_voc: float = 1.5e-5
    molecular_diffusion_virus: float = 1.0e-6  # partículas maiores
    molecular_diffusion_pm25: float = 5.0e-7
    molecular_diffusion_pm10: float = 2.0e-7
    
    # Difusividades turbulentas (m²/s) - dependem da ventilação
    turbulent_diffusion_low_vent: float = 1.0e-4   # ACH < 2
    turbulent_diffusion_medium_vent: float = 5.0e-4  # 2 < ACH < 6
    turbulent_diffusion_high_vent: float = 1.0e-3   # ACH > 6
    
    # Outras difusividades
    thermal_diffusivity: float = 2.0e-5  # ar (m²/s)
    moisture_diffusivity: float = 2.5e-5  # vapor d'água (m²/s)
    
    # Velocidades de deposição (m/s)
    deposition_velocity_virus: float = 0.0001  # 0.1 mm/s
    deposition_velocity_pm25: float = 0.00005  # 0.05 mm/s
    deposition_velocity_pm10: float = 0.0001   # 0.1 mm/s
    
    # Taxas de reação química (1/s)
    hcho_decay_rate: float = 1.0e-5  # decaimento lento
    voc_oxidation_rate: float = 5.0e-6  # oxidação por ozônio/radicais
    
    # Filtro de Kalman
    kalman_enabled: bool = False
    kalman_process_noise: float = 0.01
    kalman_measurement_noise: float = 0.05
    
    # Compatibilidade reversa
    @property
    def diffusion_coefficient_co2(self) -> float:
        """Retorna difusividade total CO2 (molecular + turbulenta média)."""
        return self.molecular_diffusion_co2 + self.turbulent_diffusion_medium_vent


# ============================================================================
# CONFIGURAÇÕES DE ZONA
# ============================================================================

@dataclass
class Zone:
    """
    Representa uma zona/compartimento dentro do ambiente.
    
    Attributes:
        name: Nome da zona
        zone_type: Tipo da zona (general, isolation, cleanroom, etc.)
        x_start: Coordenada X inicial (metros)
        y_start: Coordenada Y inicial (metros)
        x_end: Coordenada X final (metros)
        y_end: Coordenada Y final (metros)
        z_start: Coordenada Z inicial (metros)
        z_end: Coordenada Z final (metros)
        target_ach: Taxa de troca de ar alvo (ACH)
        occupancy_density: Densidade de ocupação (m²/pessoa)
        equipment_heat_gain: Ganho de calor por equipamentos (W/m²)
        lighting_density: Densidade de iluminação (W/m²)
        outdoor_air_fraction: Fração de ar externo
        materials: Lista de materiais presentes na zona
    """
    name: str
    zone_type: str
    x_start: float
    y_start: float
    x_end: float
    y_end: float
    z_start: float
    z_end: float
    target_ach: float = 4.0
    occupancy_density: float = 5.0  # m²/pessoa
    equipment_heat_gain: float = 10.0  # W/m²
    lighting_density: float = 12.0  # W/m²
    outdoor_air_fraction: float = 0.3
    materials: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def width_ratio(self) -> float:
        """Retorna razão de largura (usado em zone_map)."""
        # Simplificação: assume grid normalizado 0-1
        return self.x_end - self.x_start
    
    @property
    def height_ratio(self) -> float:
        """Retorna razão de altura (usado em zone_map)."""
        return self.y_end - self.y_start
    
    @property
    def volume(self) -> float:
        """Calcula volume da zona (m³)."""
        return (self.x_end - self.x_start) * (self.y_end - self.y_start) * (self.z_end - self.z_start)
    
    @property
    def area(self) -> float:
        """Calcula área de piso (m²)."""
        return (self.x_end - self.x_start) * (self.y_end - self.y_start)


# ============================================================================
# CONFIGURAÇÕES DE AGENTE
# ============================================================================

@dataclass
class AgentConfig:
    """
    Configuração para agentes (ocupantes do ambiente).
    
    Baseado em Buonanno et al. (2020) para emissão viral e 
    ASHRAE Fundamentals para taxas metabólicas.
    
    Attributes:
        activity_level: Nível de atividade metabólica
        base_quanta_emission: Taxa basal de emissão viral (quanta/h)
        activity_multiplier: Multiplicador de emissão por atividade
        respiration_rate: Taxa de respiração (m³/h)
        mask_efficiency: Eficiência de máscara (0.0-1.0)
        vaccination_factor: Fator de proteção por vacinação (0.0-1.0)
    """
    activity_level: ActivityLevel
    base_quanta_emission: float  # quanta/h
    activity_multiplier: float = 1.0
    respiration_rate: float = 0.54  # m³/h (valor médio adulto)
    mask_efficiency: float = 0.0
    vaccination_factor: float = 0.0
    
    @property
    def effective_quanta_emission(self) -> float:
        """
        Calcula emissão efetiva de quanta considerando atividade.
        Baseado em Buonanno et al. (2020).
        """
        return self.base_quanta_emission * self.activity_multiplier
    
    @property
    def metabolic_rate(self) -> float:
        """
        Retorna taxa metabólica em met (1 met = 58.2 W/m²).
        Baseado em ASHRAE Fundamentals.
        """
        rates = {
            ActivityLevel.SEDENTARY: 1.0,
            ActivityLevel.LIGHT: 1.6,
            ActivityLevel.MODERATE: 3.0,
            ActivityLevel.HEAVY: 6.0,
            ActivityLevel.VERY_HEAVY: 8.0
        }
        return rates[self.activity_level]


# ============================================================================
# CONFIGURAÇÕES DE VENTILAÇÃO
# ============================================================================

@dataclass
class VentilationConfig:
    """
    Configuração de ventilação do ambiente.
    
    Baseado em ASHRAE 62.1-2019.
    
    Attributes:
        ach: Trocas de ar por hora (Air Changes per Hour)
        ventilation_type: Tipo de sistema de ventilação
        outdoor_air_fraction: Fração de ar externo (0.0-1.0)
        filtration_efficiency: Eficiência de filtração MERV (0.0-1.0)
        co2_outdoor: Concentração de CO₂ externa (ppm)
    """
    ach: float  # Air Changes per Hour
    ventilation_type: VentilationType
    outdoor_air_fraction: float = 0.3
    filtration_efficiency: float = 0.0  # 0 = sem filtro, 0.8 = MERV 13
    co2_outdoor: float = 420.0  # ppm (concentração atmosférica atual)
    
    def __post_init__(self):
        """Valida parâmetros."""
        if self.ach < 0:
            raise ValueError("ACH deve ser não-negativo")
        if not 0 <= self.outdoor_air_fraction <= 1:
            raise ValueError("outdoor_air_fraction deve estar entre 0 e 1")
        if not 0 <= self.filtration_efficiency <= 1:
            raise ValueError("filtration_efficiency deve estar entre 0 e 1")


# ============================================================================
# CONFIGURAÇÕES DE MATERIAIS
# ============================================================================

@dataclass
class MaterialProperties:
    """
    Propriedades de um material de construção.
    
    Attributes:
        name: Nome do material
        material_type: Tipo de material
        hcho_emission_rate: Taxa de emissão de formaldeído (kg/m²/s)
        voc_emission_rate: Taxa de emissão de VOC (kg/m²/s)
        decay_rate: Taxa de decaimento (1/dia)
        surface_factor: Fator de superfície
        age_days: Idade do material em dias
    """
    name: str
    material_type: MaterialType
    hcho_emission_rate: float  # kg/m²/s
    voc_emission_rate: float   # kg/m²/s
    decay_rate: float          # 1/day
    surface_factor: float      # fator de superfície
    age_days: float = 0.0
    temperature_coefficient: float = 0.02  # %/°C
    moisture_coefficient: float = 0.01  # %/%RH


# ============================================================================
# CONFIGURAÇÕES DE OBSTÁCULOS
# ============================================================================

@dataclass
class Obstacle:
    """
    Representa um obstáculo físico no ambiente.
    
    Attributes:
        id: Identificador único
        x: Posição X (metros)
        y: Posição Y (metros)
        width: Largura (metros)
        height: Altura (metros)
        obstacle_type: Tipo de obstáculo
        porosity: Porosidade para fluxo de ar (0.0 = sólido, 1.0 = totalmente permeável)
    """
    id: str
    x: float
    y: float
    width: float
    height: float
    obstacle_type: ObstacleType
    porosity: float = 0.0  # 0 = impermeável
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'id': self.id,
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'type': self.obstacle_type.value,
            'porosity': self.porosity
        }


# ============================================================================
# CENÁRIO DE AMBIENTE
# ============================================================================

@dataclass
class BuildingScenario:
    """
    Define um cenário completo de ambiente interno.
    
    Attributes:
        building_type: Tipo de edificação
        name: Nome do cenário
        description: Descrição
        room_volume: Volume do ambiente (m³)
        floor_area: Área do piso (m²)
        ceiling_height: Pé-direito (metros)
        occupancy_density: Densidade de ocupação (m²/pessoa)
        max_occupants: Número máximo de ocupantes
        ventilation: Configuração de ventilação
        agent_config: Configuração padrão de agentes
        obstacles: Lista de obstáculos físicos
        temperature: Temperatura ambiente (°C)
        relative_humidity: Umidade relativa (%)
        default_materials: Materiais padrão do ambiente
        total_width: Largura total do ambiente (metros)
        total_height: Profundidade total do ambiente (metros)
        floor_height: Altura do piso (metros, geralmente 0)
        zones: Lista de zonas
        connections: Conexões entre zonas
        total_occupants: Número total de ocupantes
        initial_infected_ratio: Fração inicial de infectados
        temperature_setpoint: Setpoint de temperatura (°C)
        humidity_setpoint: Setpoint de umidade (%)
        co2_setpoint: Setpoint de CO2 (ppm)
        mask_usage_rate: Taxa de uso de máscaras (0.0 a 1.0)
        external_temperature: Temperatura externa em Kelvin (20°C)
    """
    building_type: BuildingType
    name: str
    description: str
    room_volume: float  # m³
    floor_area: float  # m²
    ceiling_height: float  # m
    occupancy_density: float  # m²/pessoa
    max_occupants: int
    ventilation: VentilationConfig
    agent_config: AgentConfig
    obstacles: List[Any] = field(default_factory=list)  # Aceita Obstacle ou dict
    temperature: float = 22.0  # °C
    relative_humidity: float = 50.0  # %
    default_materials: Dict[MaterialType, MaterialProperties] = field(default_factory=dict)
    total_width: float = 0.0
    total_height: float = 0.0
    floor_height: float = 0.0
    zones: List[Zone] = field(default_factory=list)
    connections: List[Dict[str, Any]] = field(default_factory=list)
    total_occupants: int = 0
    initial_infected_ratio: float = 0.0
    temperature_setpoint: float = 22.0
    humidity_setpoint: float = 50.0
    co2_setpoint: float = 800.0
    mask_usage_rate: float = 0.0  
    external_temperature: float = 293.15
    
    def __post_init__(self):
        """Valida consistência dos parâmetros e cria zona padrão se necessário."""
        # Calcula dimensões se não fornecidas
        if self.total_width == 0.0:
            self.total_width = np.sqrt(self.floor_area)
        if self.total_height == 0.0:
            self.total_height = self.floor_area / self.total_width
        
        # Valida volume
        calculated_volume = self.floor_area * self.ceiling_height
        if abs(calculated_volume - self.room_volume) > 0.1:
            raise ValueError(
                f"Volume inconsistente: {self.room_volume} m³ declarado vs "
                f"{calculated_volume} m³ calculado"
            )
        
        # Valida ocupação (com margem para arredondamento)
        max_capacity_float = self.floor_area / self.occupancy_density
        max_capacity = int(np.ceil(max_capacity_float))  # Arredonda para CIMA

        # Permite margem de 10% ou +1 pessoa (o que for maior)
        tolerance = max(1, int(max_capacity * 0.1))

        if self.max_occupants > max_capacity + tolerance:
            raise ValueError(
                f"max_occupants ({self.max_occupants}) excede significativamente "
                f"a capacidade baseada em densidade ({max_capacity}, "
                f"tolerância: +{tolerance})"
            )
        
        # Define total_occupants se não definido
        if self.total_occupants == 0:
            self.total_occupants = self.max_occupants
        
        # Cria zona padrão se nenhuma foi fornecida
        if not self.zones:
            default_zone = Zone(
                name="Main Zone",
                zone_type="general",
                x_start=0.0,
                y_start=0.0,
                x_end=self.total_width,
                y_end=self.total_height,
                z_start=self.floor_height,
                z_end=self.floor_height + self.ceiling_height,
                target_ach=self.ventilation.ach,
                occupancy_density=self.occupancy_density,
                materials=[]
            )
            self.zones = [default_zone]
        
        # Converte obstáculos de dict para Obstacle se necessário
        converted_obstacles = []
        for obs in self.obstacles:
            if isinstance(obs, dict):
                converted_obstacles.append(Obstacle(
                    id=obs.get('id', f'obs_{len(converted_obstacles)}'),
                    x=obs.get('x', 0.0),
                    y=obs.get('y', 0.0),
                    width=obs.get('width', 1.0),
                    height=obs.get('height', 1.0),
                    obstacle_type=ObstacleType(obs.get('type', 'furniture')),
                    porosity=obs.get('porosity', 0.0)
                ))
            else:
                converted_obstacles.append(obs)
        self.obstacles = converted_obstacles
    
    @property
    def obstacles_list(self) -> List[Any]:
        """Retorna lista de obstáculos (compatibilidade)."""
        return self.obstacles


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def calculate_zone_parameters(
    zone: Zone, 
    total_width: float, 
    total_height: float,
    floor_height: float
) -> Dict[str, Any]:
    """
    Calcula parâmetros derivados de uma zona.
    
    Args:
        zone: Zona a calcular
        total_width: Largura total do ambiente
        total_height: Profundidade total do ambiente
        floor_height: Altura do piso
    
    Returns:
        Dicionário com parâmetros calculados
    """
    area = zone.area
    volume = zone.volume
    max_occupants = int(area / zone.occupancy_density)
    
    return {
        'area': area,
        'volume': volume,
        'max_occupants': max_occupants,
        'width': zone.x_end - zone.x_start,
        'height': zone.y_end - zone.y_start,
        'ceiling_height': zone.z_end - zone.z_start
    }


# ============================================================================
# FUNÇÕES DE CRIAÇÃO DE CENÁRIOS PRÉ-DEFINIDOS
# ============================================================================

def create_school_scenario(
    room_length: float = 10.0,
    room_width: float = 8.0,
    ceiling_height: float = 3.0
) -> BuildingScenario:
    """
    Cria cenário de sala de aula baseado em ASHRAE 62.1 e Harvard Healthy Buildings.
    
    Parâmetros Científicos Validados:
    - Densidade: 1.9 m²/pessoa (ASHRAE 62.1 Table 6-1)
    - Ventilação: 4.5 ACH (Harvard Healthy Buildings, 2021)
    - Emissão Viral: 5.0 quanta/h (crianças falando - Buonanno et al., 2020)
    - Layout: Fileiras de carteiras organizadas
    
    Args:
        room_length: Comprimento da sala (metros)
        room_width: Largura da sala (metros)
        ceiling_height: Pé-direito (metros)
    
    Returns:
        BuildingScenario configurado para escola
        
    References:
        - ASHRAE 62.1-2019: Ventilation for Acceptable Indoor Air Quality
        - Harvard Healthy Buildings (2021): Schools For Health
        - Buonanno et al. (2020): Estimation of airborne viral emission
    """
    floor_area = room_length * room_width
    room_volume = floor_area * ceiling_height
    
    # ====== PARÂMETROS CIENTÍFICOS VALIDADOS ======
    DENSITY_M2_PER_PERSON = 1.9  # ASHRAE 62.1
    ACH = 4.5  # Harvard Healthy Buildings
    QUANTA_EMISSION = 5.0  # Buonanno et al. - crianças falando
    
    # Configuração de ventilação
    ventilation = VentilationConfig(
        ach=ACH,
        ventilation_type=VentilationType.MECHANICAL,
        outdoor_air_fraction=0.35,
        filtration_efficiency=0.6,  # MERV 11
        co2_outdoor=420.0
    )
    
    # Configuração de agentes (estudantes)
    agent_config = AgentConfig(
        activity_level=ActivityLevel.LIGHT,
        base_quanta_emission=QUANTA_EMISSION,
        activity_multiplier=1.0,  # Fala contínua em aula
        respiration_rate=0.48,  # m³/h (crianças 8-12 anos)
        mask_efficiency=0.0,
        vaccination_factor=0.0
    )
    
    # ====== LAYOUT: OBSTÁCULOS FÍSICOS ======
    obstacles = []
    
    # Mesa do professor (frente da sala)
    obstacles.append(Obstacle(
        id='teacher_desk',
        x=room_length / 2,
        y=0.8,
        width=1.6,
        height=0.8,
        obstacle_type=ObstacleType.FURNITURE,
        porosity=0.7
    ))
    
    # Fileiras de carteiras (3 fileiras x 4 colunas = 12 carteiras)
    desk_spacing_x = room_length / 5
    desk_spacing_y = (room_width - 2.0) / 4
    desk_start_y = 2.5
    
    desk_count = 0
    for row in range(3):
        for col in range(4):
            obstacles.append(Obstacle(
                id=f'desk_{desk_count}',
                x=1.5 + col * desk_spacing_x,
                y=desk_start_y + row * desk_spacing_y,
                width=0.6,
                height=0.5,
                obstacle_type=ObstacleType.FURNITURE,
                porosity=0.7
            ))
            desk_count += 1
    
    # Materiais padrão para escolas
    materials = {
        MaterialType.MDF: MaterialProperties(
            name="MDF Classroom Furniture",
            material_type=MaterialType.MDF,
            hcho_emission_rate=1.7e-11,
            voc_emission_rate=3.0e-11,
            decay_rate=0.01,
            surface_factor=2.5,
            age_days=180
        ),
        MaterialType.LATEX_PAINT: MaterialProperties(
            name="Latex Paint",
            material_type=MaterialType.LATEX_PAINT,
            hcho_emission_rate=5.6e-11,
            voc_emission_rate=8.0e-11,
            decay_rate=0.05,
            surface_factor=1.0,
            age_days=30
        ),
        MaterialType.CARPET: MaterialProperties(
            name="Classroom Carpet",
            material_type=MaterialType.CARPET,
            hcho_emission_rate=2.5e-11,
            voc_emission_rate=5.0e-11,
            decay_rate=0.02,
            surface_factor=1.2,
            age_days=365
        )
    }
    
    # Cenário completo
    return BuildingScenario(
        building_type=BuildingType.SCHOOL,
        name="Modern School Classroom",
        description="Sala de aula moderna com ventilação mecânica e 12 carteiras",
        room_volume=room_volume,
        floor_area=floor_area,
        ceiling_height=ceiling_height,
        occupancy_density=DENSITY_M2_PER_PERSON,
        max_occupants=int(floor_area / DENSITY_M2_PER_PERSON),
        ventilation=ventilation,
        agent_config=agent_config,
        obstacles=obstacles,
        temperature=22.0,
        relative_humidity=45.0,
        default_materials=materials,
        total_width=room_length,
        total_height=room_width,
        floor_height=0.0,
        total_occupants=int(floor_area / DENSITY_M2_PER_PERSON),
        initial_infected_ratio=0.0,
        temperature_setpoint=22.0,
        humidity_setpoint=45.0,
        co2_setpoint=800.0
    )


def create_gym_scenario(
    room_length: float = 15.0,
    room_width: float = 12.0,
    ceiling_height: float = 4.0
) -> BuildingScenario:
    """
    Cria cenário de academia (Crossfit/Pesos) baseado em pesquisa validada.
    
    Parâmetros Científicos Validados:
    - Densidade: 8.0 m²/pessoa (alta performance, espaçamento adequado)
    - Ventilação: 10.0 ACH (necessário para diluição de aerossóis de exercício)
    - Emissão Viral: 2.5 quanta/h base × 4.0 multiplicador = 10.0 quanta/h efetivo
    - Layout: Área aberta com zonas de equipamentos
    
    Args:
        room_length: Comprimento da academia (metros)
        room_width: Largura da academia (metros)
        ceiling_height: Pé-direito (metros)
    
    Returns:
        BuildingScenario configurado para academia
        
    References:
        - ASHRAE 62.1-2019: Fitness centers require high ventilation
        - Buonanno et al. (2020): High emission during intense exercise
    """
    floor_area = room_length * room_width
    room_volume = floor_area * ceiling_height
    
    # ====== PARÂMETROS CIENTÍFICOS VALIDADOS ======
    DENSITY_M2_PER_PERSON = 8.0  # Alta performance
    ACH = 10.0  # Alto para diluição de aerossóis
    BASE_QUANTA = 2.5  # Baseline
    ACTIVITY_MULTIPLIER = 4.0  # Exercício intenso (10.0 quanta/h efetivo)
    
    # Configuração de ventilação (alta taxa necessária)
    ventilation = VentilationConfig(
        ach=ACH,
        ventilation_type=VentilationType.MECHANICAL,
        outdoor_air_fraction=0.5,  # 50% ar fresco
        filtration_efficiency=0.75,  # MERV 13
        co2_outdoor=420.0
    )
    
    # Configuração de agentes (atletas)
    agent_config = AgentConfig(
        activity_level=ActivityLevel.HEAVY,
        base_quanta_emission=BASE_QUANTA,
        activity_multiplier=ACTIVITY_MULTIPLIER,  # Exercício intenso
        respiration_rate=1.38,  # m³/h (exercício pesado)
        mask_efficiency=0.0,  # Impossível usar máscara em exercício
        vaccination_factor=0.0
    )
    
    # ====== LAYOUT: EQUIPAMENTOS DE ACADEMIA ======
    obstacles = []
    
    # Zona de Pesos Livres (racks de agachamento)
    obstacles.append(Obstacle(
        id='squat_rack_1',
        x=3.0,
        y=2.0,
        width=2.0,
        height=1.5,
        obstacle_type=ObstacleType.EQUIPMENT,
        porosity=0.5
    ))
    
    obstacles.append(Obstacle(
        id='squat_rack_2',
        x=3.0,
        y=5.5,
        width=2.0,
        height=1.5,
        obstacle_type=ObstacleType.EQUIPMENT,
        porosity=0.5
    ))
    
    # Zona de Cardio (esteiras)
    for i in range(3):
        obstacles.append(Obstacle(
            id=f'treadmill_{i}',
            x=10.5,
            y=2.0 + i * 2.2,
            width=1.8,
            height=0.8,
            obstacle_type=ObstacleType.EQUIPMENT,
            porosity=0.6
        ))
    
    # Área de Crossfit (rig central)
    obstacles.append(Obstacle(
        id='crossfit_rig',
        x=room_length / 2,
        y=room_width / 2,
        width=2.5,
        height=1.8,
        obstacle_type=ObstacleType.EQUIPMENT,
        porosity=0.4
    ))
    
    # Banco de musculação
    obstacles.append(Obstacle(
        id='bench_press',
        x=6.5,
        y=9.0,
        width=2.2,
        height=1.2,
        obstacle_type=ObstacleType.EQUIPMENT,
        porosity=0.5
    ))
    
    # Materiais para academia
    materials = {
        MaterialType.RUBBER: MaterialProperties(
            name="Rubber Flooring",
            material_type=MaterialType.RUBBER,
            hcho_emission_rate=8.0e-12,
            voc_emission_rate=3.0e-11,
            decay_rate=0.01,
            surface_factor=1.0,
            age_days=180
        ),
        MaterialType.METAL: MaterialProperties(
            name="Gym Equipment",
            material_type=MaterialType.METAL,
            hcho_emission_rate=1.0e-13,
            voc_emission_rate=1.0e-13,
            decay_rate=0.0,
            surface_factor=1.5,
            age_days=0
        ),
        MaterialType.CONCRETE: MaterialProperties(
            name="Concrete Walls",
            material_type=MaterialType.CONCRETE,
            hcho_emission_rate=1.0e-12,
            voc_emission_rate=1.0e-12,
            decay_rate=0.0,
            surface_factor=1.0,
            age_days=0
        )
    }
    
    # Cenário completo
    return BuildingScenario(
        building_type=BuildingType.GYM,
        name="Crossfit & Weight Training Gym",
        description="Academia com área de pesos livres, cardio e crossfit",
        room_volume=room_volume,
        floor_area=floor_area,
        ceiling_height=ceiling_height,
        occupancy_density=DENSITY_M2_PER_PERSON,
        max_occupants=int(floor_area / DENSITY_M2_PER_PERSON),
        ventilation=ventilation,
        agent_config=agent_config,
        obstacles=obstacles,
        temperature=20.0,  # Mais frio para exercício
        relative_humidity=40.0,
        default_materials=materials,
        total_width=room_length,
        total_height=room_width,
        floor_height=0.0,
        total_occupants=int(floor_area / DENSITY_M2_PER_PERSON),
        initial_infected_ratio=0.0,
        temperature_setpoint=20.0,
        humidity_setpoint=40.0,
        co2_setpoint=1000.0
    )


def create_office_scenario(
    room_length: float = 12.0,
    room_width: float = 10.0,
    ceiling_height: float = 2.7
) -> BuildingScenario:
    """
    Cria cenário de escritório (Open Space) baseado em ASHRAE 62.1.
    
    Parâmetros Científicos Validados:
    - Densidade: 9.3 m²/pessoa (ASHRAE 62.1 para escritórios)
    - Ventilação: 3.0 ACH (padrão para escritórios comerciais)
    - Emissão Viral: 1.5 quanta/h (fala ocasional - Buonanno et al., 2020)
    - Layout: Ilhas de trabalho (baias/estações)
    
    Args:
        room_length: Comprimento do escritório (metros)
        room_width: Largura do escritório (metros)
        ceiling_height: Pé-direito (metros)
    
    Returns:
        BuildingScenario configurado para escritório
        
    References:
        - ASHRAE 62.1-2019: Office space requirements
        - Buonanno et al. (2020): Low emission during sedentary work
    """
    floor_area = room_length * room_width
    room_volume = floor_area * ceiling_height
    
    # ====== PARÂMETROS CIENTÍFICOS VALIDADOS ======
    DENSITY_M2_PER_PERSON = 9.3  # ASHRAE 62.1
    ACH = 3.0  # Padrão escritórios
    QUANTA_EMISSION = 1.5  # Fala ocasional
    
    # Configuração de ventilação
    ventilation = VentilationConfig(
        ach=ACH,
        ventilation_type=VentilationType.MECHANICAL,
        outdoor_air_fraction=0.25,
        filtration_efficiency=0.65,  # MERV 11-12
        co2_outdoor=420.0
    )
    
    # Configuração de agentes (trabalhadores de escritório)
    agent_config = AgentConfig(
        activity_level=ActivityLevel.SEDENTARY,
        base_quanta_emission=QUANTA_EMISSION,
        activity_multiplier=1.0,  # Fala ocasional
        respiration_rate=0.54,  # m³/h (atividade sedentária)
        mask_efficiency=0.0,
        vaccination_factor=0.0
    )
    
    # ====== LAYOUT: ESTAÇÕES DE TRABALHO EM ILHAS ======
    obstacles = []
    
    # Ilha de trabalho 1 (4 estações)
    island_1_x = 2.5
    island_1_y = 2.0
    obstacles.append(Obstacle(
        id='workstation_island_1',
        x=island_1_x,
        y=island_1_y,
        width=3.0,
        height=2.5,
        obstacle_type=ObstacleType.FURNITURE,
        porosity=0.7
    ))
    
    # Ilha de trabalho 2 (4 estações)
    island_2_x = 2.5
    island_2_y = 6.0
    obstacles.append(Obstacle(
        id='workstation_island_2',
        x=island_2_x,
        y=island_2_y,
        width=3.0,
        height=2.5,
        obstacle_type=ObstacleType.FURNITURE,
        porosity=0.7
    ))
    
    # Ilha de trabalho 3 (lateral direita)
    island_3_x = 8.0
    island_3_y = 2.0
    obstacles.append(Obstacle(
        id='workstation_island_3',
        x=island_3_x,
        y=island_3_y,
        width=3.0,
        height=2.5,
        obstacle_type=ObstacleType.FURNITURE,
        porosity=0.7
    ))
    
    # Sala de reunião (canto superior direito) - paredes
    obstacles.append(Obstacle(
        id='meeting_room_wall',
        x=8.5,
        y=6.5,
        width=2.5,
        height=2.5,
        obstacle_type=ObstacleType.WALL,
        porosity=0.0
    ))
    
    # Mesa de reunião (posicionada adjacente)
    obstacles.append(Obstacle(
        id='meeting_table',
        x=6.5,
        y=7.0,
        width=1.5,
        height=1.0,
        obstacle_type=ObstacleType.FURNITURE,
        porosity=0.7
    ))
    
    # Materiais para escritório
    materials = {
        MaterialType.MDF: MaterialProperties(
            name="Office Furniture MDF",
            material_type=MaterialType.MDF,
            hcho_emission_rate=1.5e-11,
            voc_emission_rate=2.8e-11,
            decay_rate=0.008,
            surface_factor=2.0,
            age_days=365
        ),
        MaterialType.CARPET: MaterialProperties(
            name="Office Carpet",
            material_type=MaterialType.CARPET,
            hcho_emission_rate=2.2e-11,
            voc_emission_rate=4.5e-11,
            decay_rate=0.015,
            surface_factor=1.2,
            age_days=180
        ),
        MaterialType.GLASS: MaterialProperties(
            name="Glass Partitions",
            material_type=MaterialType.GLASS,
            hcho_emission_rate=1.0e-12,
            voc_emission_rate=1.0e-12,
            decay_rate=0.0,
            surface_factor=1.0,
            age_days=0
        )
    }
    
    # Cenário completo
    return BuildingScenario(
        building_type=BuildingType.OFFICE,
        name="Modern Open Office",
        description="Escritório aberto com ilhas de trabalho e sala de reunião",
        room_volume=room_volume,
        floor_area=floor_area,
        ceiling_height=ceiling_height,
        occupancy_density=DENSITY_M2_PER_PERSON,
        max_occupants=int(floor_area / DENSITY_M2_PER_PERSON),
        ventilation=ventilation,
        agent_config=agent_config,
        obstacles=obstacles,
        temperature=22.0,
        relative_humidity=50.0,
        default_materials=materials,
        total_width=room_length,
        total_height=room_width,
        floor_height=0.0,
        total_occupants=int(floor_area / DENSITY_M2_PER_PERSON),
        initial_infected_ratio=0.0,
        temperature_setpoint=22.0,
        humidity_setpoint=50.0,
        co2_setpoint=800.0
    )


# ============================================================================
# CONSTANTES FÍSICAS VALIDADAS
# ============================================================================

# Emissões humanas baseadas em Buonanno et al. (2020) e ASHRAE
HUMAN_EMISSION_RATES = {
    'co2': {
        'sleeping': 4.7e-6,      # kg/s (0.7 MET)
        'seated_quiet': 7.4e-6,  # kg/s (1.1 MET)
        'seated_typing': 8.2e-6, # kg/s (1.2 MET)
        'standing': 8.9e-6,      # kg/s (1.3 MET)
        'walking': 1.3e-5,       # kg/s (2.0 MET)
        'exercising_light': 2.2e-5,   # kg/s (3.5 MET)
        'exercising_intense': 4.0e-5,  # kg/s (6.0 MET)
    },
    'quanta': {
        'seated_quiet': 2.0 / 3600.0,      # quanta/s
        'talking': 10.0 / 3600.0,       # quanta/s
        'exercising_light': 5.0 / 3600.0,   # quanta/s
        'exercising_intense': 15.0 / 3600.0, # quanta/s
    },
    'heat': {
        'sleeping': 70,           # W
        'seated_quiet': 100,      # W
        'standing': 130,          # W
        'walking': 200,           # W
        'exercising_light': 350,  # W
        'exercising_intense': 600, # W
    }
}

# Limites de exposição (ASHRAE, WHO, EPA)
EXPOSURE_LIMITS = {
    'co2': {
        'excellent': 600,    # ppm
        'good': 800,         # ppm
        'moderate': 1000,    # ppm
        'poor': 1500,        # ppm
        'unhealthy': 2000    # ppm
    },
    'hcho': {
        'who_30min': 81.8,   # ppb
        'who_annual': 24.5,  # ppb
        'epa_chronic': 13.1, # ppb
    },
    'temperature': {
        'comfort_min': 20.0,  # °C
        'comfort_max': 24.0,  # °C
    },
    'humidity': {
        'comfort_min': 40.0,  # %
        'comfort_max': 60.0,  # %
    }
}

# Fatores de conversão
CONVERSION_FACTORS = {
    'co2_ppm_to_kgm3': 1.8e-6,
    'co2_kgm3_to_ppm': 5.56e5,
    'hcho_ppb_to_kgm3': 1.23e-9,
    'hcho_kgm3_to_ppb': 8.13e8,
    'voc_ppb_to_kgm3': 1.5e-9,
    'voc_kgm3_to_ppb': 6.67e8,
}

# Constantes físicas
PHYSICAL_CONSTANTS = {
    'air_density': 1.204,           # kg/m³ @ 20°C
    'air_specific_heat': 1005,      # J/kg·K
    'gravity': 9.81,                # m/s²
    'gas_constant': 287.05,         # J/kg·K para ar seco
}

# Eficiências de filtragem (ASHRAE 52.2)
FILTRATION_EFFICIENCIES = {
    'merv_8': {
        'pm25': 0.30,
        'pm10': 0.70,
        'virus': 0.10
    },
    'merv_13': {
        'pm25': 0.85,
        'pm10': 0.90,
        'virus': 0.50
    },
    'hepa': {
        'pm25': 0.995,
        'pm10': 0.997,
        'virus': 0.999
    }
}


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def get_scenario_by_type(
    building_type: BuildingType,
    **kwargs
) -> BuildingScenario:
    """
    Factory function para criar cenários baseados no tipo.
    
    Args:
        building_type: Tipo de edificação
        **kwargs: Argumentos adicionais para dimensões
    
    Returns:
        BuildingScenario configurado
    
    Raises:
        ValueError: Se tipo de edificação não for suportado
    """
    scenarios = {
        BuildingType.SCHOOL: create_school_scenario,
        BuildingType.GYM: create_gym_scenario,
        BuildingType.OFFICE: create_office_scenario
    }
    
    if building_type not in scenarios:
        raise ValueError(f"Tipo de edificação '{building_type}' não suportado")
    
    return scenarios[building_type](**kwargs)


def validate_scenario(scenario: BuildingScenario) -> List[str]:
    """
    Valida parâmetros do cenário e retorna lista de avisos.
    
    Args:
        scenario: Cenário a ser validado
    
    Returns:
        Lista de strings com avisos (vazia se tudo OK)
    """
    warnings = []
    
    # Validação de ACH
    if scenario.ventilation.ach < 2.0:
        warnings.append(
            f"ACH muito baixo ({scenario.ventilation.ach}). "
            "ASHRAE recomenda mínimo 2.0 para ambientes ocupados."
        )
    
    # Validação de densidade
    if scenario.occupancy_density < 1.0:
        warnings.append(
            f"Densidade muito alta ({scenario.occupancy_density} m²/pessoa). "
            "Risco elevado de transmissão."
        )
    
    # Validação de fração de ar externo
    if scenario.ventilation.outdoor_air_fraction < 0.15:
        warnings.append(
            f"Fração de ar externo muito baixa "
            f"({scenario.ventilation.outdoor_air_fraction}). "
            "Pode causar acúmulo de CO₂."
        )
    
    return warnings


def print_scenario_summary(scenario: BuildingScenario) -> None:
    """
    Imprime resumo formatado do cenário.
    
    Args:
        scenario: Cenário a ser resumido
    """
    print("=" * 70)
    print(f"CENÁRIO: {scenario.name}")
    print("=" * 70)
    print(f"Tipo: {scenario.building_type.value}")
    print(f"Descrição: {scenario.description}")
    print(f"\nDimensões:")
    print(f"  - Área: {scenario.floor_area:.1f} m²")
    print(f"  - Volume: {scenario.room_volume:.1f} m³")
    print(f"  - Pé-direito: {scenario.ceiling_height:.1f} m")
    print(f"\nOcupação:")
    print(f"  - Densidade: {scenario.occupancy_density:.1f} m²/pessoa")
    print(f"  - Capacidade máxima: {scenario.max_occupants} pessoas")
    print(f"\nVentilação:")
    print(f"  - ACH: {scenario.ventilation.ach:.1f} trocas/hora")
    print(f"  - Tipo: {scenario.ventilation.ventilation_type.value}")
    print(f"  - Ar externo: {scenario.ventilation.outdoor_air_fraction*100:.0f}%")
    print(f"  - Eficiência filtração: {scenario.ventilation.filtration_efficiency*100:.0f}%")
    print(f"\nAgentes:")
    print(f"  - Atividade: {scenario.agent_config.activity_level.value}")
    print(f"  - Taxa metabólica: {scenario.agent_config.metabolic_rate:.1f} met")
    print(f"  - Emissão viral base: {scenario.agent_config.base_quanta_emission:.1f} quanta/h")
    print(f"  - Emissão efetiva: {scenario.agent_config.effective_quanta_emission:.1f} quanta/h")
    print(f"\nObstáculos: {len(scenario.obstacles)} elementos")
    
    # Validação
    warnings = validate_scenario(scenario)
    if warnings:
        print(f"\n⚠️  AVISOS ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\n✓ Cenário validado com sucesso")
    print("=" * 70)
