import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

# ============================================================================
# 1. ENUMS PARA CONFIGURAÇÃO TIPADA
# ============================================================================

class BuildingType(str, Enum):
    """Tipos de edificação suportados."""
    SCHOOL = "school"
    OFFICE = "office"
    GYM = "gym"
    HOSPITAL = "hospital"
    RESIDENTIAL = "residential"
    CUSTOM = "custom"

class ZoneType(str, Enum):
    """Tipos de zonas internas."""
    CLASSROOM = "classroom"
    CORRIDOR = "corridor"
    OFFICE_SPACE = "office_space"
    MEETING_ROOM = "meeting_room"
    GYM_AREA = "gym_area"
    CAFETERIA = "cafeteria"
    LIBRARY = "library"
    RESTROOM = "restroom"
    LABORATORY = "laboratory"
    AUDITORIUM = "auditorium"
    WAITING_ROOM = "waiting_room"
    PATIENT_ROOM = "patient_room"
    OPERATING_ROOM = "operating_room"
    KITCHEN = "kitchen"
    BEDROOM = "bedroom"
    LIVING_ROOM = "living_room"

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
    LEATHER = "leather"
    PAPER = "paper"

class AgentActivity(str, Enum):
    """Atividades dos agentes."""
    SLEEPING = "sleeping"
    SEATED_QUIET = "seated_quiet"
    SEATED_TYPING = "seated_typing"
    STANDING = "standing"
    WALKING = "walking"
    EXERCISING_LIGHT = "exercising_light"
    EXERCISING_INTENSE = "exercising_intense"
    TALKING = "talking"
    SINGING = "singing"
    COUGHING = "coughing"
    SNEEZING = "sneezing"
    EATING = "eating"
    DRINKING = "drinking"
    READING = "reading"
    PRESENTING = "presenting"

class VentilationMode(str, Enum):
    """Modos de ventilação."""
    NATURAL = "natural"
    MECHANICAL = "mechanical"
    MIXED = "mixed"
    DISPLACEMENT = "displacement"
    PERSONALIZED = "personalized"
    CROSS_VENTILATION = "cross_ventilation"
    STACK_VENTILATION = "stack_ventilation"

# ============================================================================
# 2. CLASSES DE CONFIGURAÇÃO
# ============================================================================

@dataclass
class MaterialProperties:
    """Propriedades de um material de construção."""
    name: str
    material_type: MaterialType
    hcho_emission_rate: float  # kg/m²/s
    voc_emission_rate: float   # kg/m²/s
    decay_rate: float          # 1/day
    surface_factor: float      # fator de superfície
    age_days: float = 0.0
    temperature_coefficient: float = 0.02  # %/°C
    moisture_coefficient: float = 0.01  # %/%RH

@dataclass
class ZoneConfig:
    """Configuração de uma zona específica."""
    zone_type: ZoneType
    name: str
    width_ratio: float  # proporção da largura total
    height_ratio: float  # proporção da altura total
    materials: List[Dict[str, Any]]  # configurações de materiais
    occupancy_density: float  # pessoas/m²
    target_ach: float  # ACH desejado
    ventilation_mode: VentilationMode
    has_windows: bool = False
    window_area_ratio: float = 0.0  # % de área de janela
    furniture_density: float = 0.3  # densidade de mobília (0-1)
    air_inlets: List[Tuple[float, float]] = field(default_factory=list)  # posições normalizadas
    air_outlets: List[Tuple[float, float]] = field(default_factory=list)
    equipment_heat_gain: float = 0.0  # W/m²
    lighting_density: float = 10.0  # W/m²

@dataclass
class AgentConfig:
    """Configuração dos agentes humanos."""
    activity_distribution: Dict[AgentActivity, float]  # probabilidade de cada atividade
    age_distribution: Dict[str, float]  # faixas etárias
    metabolic_rates: Dict[AgentActivity, float]  # MET por atividade
    mask_wearing_prob: float = 0.3
    vaccination_rate: float = 0.7
    compliance_to_rules: float = 0.8  # aderência a protocolos
    avg_stay_duration_hours: float = 8.0  # duração média de permanência

@dataclass
class PhysicsConfig:
    """Configuração da física da simulação."""
    # Grid
    cell_size: float = 0.1  # m
    dt_max: float = 0.1  # s
    stability_safety_factor: float = 0.8
    
    # Difusão
    molecular_diffusion_co2: float = 1.6e-5  # m²/s
    molecular_diffusion_hcho: float = 1.5e-5
    molecular_diffusion_virus: float = 1.0e-5
    molecular_diffusion_voc: float = 7.0e-6
    turbulent_diffusion_base: float = 5.0e-3
    turbulent_diffusion_high_vent: float = 1.0e-2
    
    # Deposição
    deposition_velocity_pm25: float = 2.0e-4  # m/s
    deposition_velocity_pm10: float = 2.0e-3
    deposition_velocity_virus: float = 1.0e-4
    
    # Pluma térmica
    plume_velocity_seated: float = 0.23  # m/s
    plume_velocity_standing: float = 0.28
    plume_temperature_excess: float = 1.5  # °C acima da ambiente
    pem_correction_active: bool = True
    
    # Filtro de Kalman
    kalman_enabled: bool = True
    kalman_process_noise: float = 0.001
    kalman_measurement_noise: float = 0.1
    kalman_update_interval: int = 60  # s
    
    # Reações químicas
    hcho_decay_rate: float = 4.6e-6  # 1/s
    voc_oxidation_rate: float = 2.3e-6  # 1/s
    
    # Conforto térmico
    convection_coefficient: float = 3.0  # W/m²K
    radiation_coefficient: float = 4.7  # W/m²K

@dataclass
class BuildingScenario:
    """Configuração completa de um cenário de edificação."""
    name: str
    building_type: BuildingType
    description: str
    
    # Dimensões
    total_width: float  # m
    total_height: float  # m
    
    # Zonas
    zones: List[ZoneConfig]
    connections: List[Dict[str, Any]]  # conexões entre zonas
    
    # Agentes
    agent_config: AgentConfig
    total_occupants: int
    initial_infected_ratio: float = 0.05
    
    floor_height: float = 2.7  # m
    
    # Ventilação
    overall_ventilation_strategy: str = "demand_controlled"
    co2_setpoint: float = 800  # ppm
    temperature_setpoint: float = 22.0  # °C
    humidity_setpoint: float = 50.0  # %
    
    # Materiais padrão
    default_materials: Dict[MaterialType, MaterialProperties] = field(default_factory=dict)
    
    # Horário de operação
    operation_hours: Tuple[float, float] = (8.0, 18.0)  # início, fim
    occupancy_schedule: Dict[str, float] = field(default_factory=lambda: {
        "morning_peak": 0.9,
        "afternoon": 0.7,
        "evening": 0.3
    })

# ============================================================================
# 3. CENÁRIOS PRÉ-DEFINIDOS
# ============================================================================

def create_school_scenario() -> BuildingScenario:
    """Cria cenário de escola com múltiplas salas e corredores."""
    
    # Materiais padrão para escolas
    materials = {
        MaterialType.MDF: MaterialProperties(
            name="MDF Classroom Furniture",
            material_type=MaterialType.MDF,
            hcho_emission_rate=1.7e-11,  # ~60 µg/m²/h
            voc_emission_rate=3.0e-11,
            decay_rate=0.01,
            surface_factor=2.5,
            age_days=180
        ),
        MaterialType.LATEX_PAINT: MaterialProperties(
            name="Latex Paint",
            material_type=MaterialType.LATEX_PAINT,
            hcho_emission_rate=5.6e-11,  # ~200 µg/m²/h
            voc_emission_rate=8.0e-11,
            decay_rate=0.05,
            surface_factor=1.0,
            age_days=30
        ),
        MaterialType.CARPET: MaterialProperties(
            name="Classroom Carpet",
            material_type=MaterialType.CARPET,
            hcho_emission_rate=2.5e-11,  # ~90 µg/m²/h
            voc_emission_rate=5.0e-11,
            decay_rate=0.02,
            surface_factor=1.2,
            age_days=365
        ),
        MaterialType.VINYL_FLOORING: MaterialProperties(
            name="Vinyl Flooring",
            material_type=MaterialType.VINYL_FLOORING,
            hcho_emission_rate=3.2e-11,  # ~115 µg/m²/h
            voc_emission_rate=6.0e-11,
            decay_rate=0.03,
            surface_factor=1.0,
            age_days=90
        ),
        MaterialType.WOOD: MaterialProperties(
            name="Solid Wood",
            material_type=MaterialType.WOOD,
            hcho_emission_rate=1.1e-11,  # ~40 µg/m²/h
            voc_emission_rate=4.0e-11,
            decay_rate=0.005,
            surface_factor=1.5,
            age_days=365
        )
    }
    
    # Configuração de zonas
    zones = [
        ZoneConfig(
            zone_type=ZoneType.CLASSROOM,
            name="Classroom A",
            width_ratio=0.2,
            height_ratio=0.5,
            materials=[
                {"type": MaterialType.LATEX_PAINT, "surface": "walls", "age_days": 30},
                {"type": MaterialType.MDF, "surface": "furniture", "density": 0.4, "age_days": 180},
                {"type": MaterialType.CARPET, "surface": "floor", "age_days": 365}
            ],
            occupancy_density=2.7,  # pessoas/m²
            target_ach=5.0,
            ventilation_mode=VentilationMode.MECHANICAL,
            has_windows=True,
            window_area_ratio=0.3,
            air_inlets=[(0.5, 0.9)],  # teto
            air_outlets=[(0.2, 0.1), (0.8, 0.1)],  # perto do chão
            equipment_heat_gain=15.0,
            lighting_density=12.0
        ),
        ZoneConfig(
            zone_type=ZoneType.CLASSROOM,
            name="Classroom B",
            width_ratio=0.2,
            height_ratio=0.5,
            materials=[
                {"type": MaterialType.LATEX_PAINT, "surface": "walls", "age_days": 30},
                {"type": MaterialType.MDF, "surface": "furniture", "density": 0.4, "age_days": 180},
                {"type": MaterialType.CARPET, "surface": "floor", "age_days": 365}
            ],
            occupancy_density=2.7,
            target_ach=5.0,
            ventilation_mode=VentilationMode.MECHANICAL,
            has_windows=True,
            window_area_ratio=0.3
        ),
        ZoneConfig(
            zone_type=ZoneType.CORRIDOR,
            name="Main Corridor",
            width_ratio=0.1,
            height_ratio=1.0,
            materials=[
                {"type": MaterialType.LATEX_PAINT, "surface": "walls", "age_days": 7},
                {"type": MaterialType.VINYL_FLOORING, "surface": "floor", "age_days": 90},
                {"type": MaterialType.CARPET, "surface": "side_areas", "age_days": 30}
            ],
            occupancy_density=0.1,  # baixa densidade
            target_ach=3.0,
            ventilation_mode=VentilationMode.MIXED,
            furniture_density=0.1
        ),
        ZoneConfig(
            zone_type=ZoneType.LIBRARY,
            name="Library",
            width_ratio=0.2,
            height_ratio=0.5,
            materials=[
                {"type": MaterialType.LATEX_PAINT, "surface": "walls", "age_days": 90},
                {"type": MaterialType.WOOD, "surface": "shelves", "density": 0.6, "age_days": 365},
                {"type": MaterialType.CARPET, "surface": "floor", "age_days": 730}
            ],
            occupancy_density=0.5,
            target_ach=4.0,
            ventilation_mode=VentilationMode.MECHANICAL,
            has_windows=True,
            window_area_ratio=0.4
        ),
        ZoneConfig(
            zone_type=ZoneType.CAFETERIA,
            name="Cafeteria",
            width_ratio=0.3,
            height_ratio=0.5,
            materials=[
                {"type": MaterialType.LATEX_PAINT, "surface": "walls", "age_days": 60},
                {"type": MaterialType.VINYL_FLOORING, "surface": "floor", "age_days": 180},
                {"type": MaterialType.METAL, "surface": "tables", "density": 0.3, "age_days": 365}
            ],
            occupancy_density=1.5,
            target_ach=8.0,  # alta ventilação para odores
            ventilation_mode=VentilationMode.MECHANICAL,
            has_windows=True,
            window_area_ratio=0.5,
            equipment_heat_gain=25.0  # cozinha
        )
    ]
    
    # Conexões entre zonas
    connections = [
        {
            "from_zone": "Classroom A",
            "to_zone": "Main Corridor",
            "type": "door",
            "width": 0.9,  # m
            "open_probability": 0.7,
            "height": 2.1
        },
        {
            "from_zone": "Classroom B",
            "to_zone": "Main Corridor",
            "type": "door",
            "width": 0.9,
            "open_probability": 0.7,
            "height": 2.1
        },
        {
            "from_zone": "Library",
            "to_zone": "Main Corridor",
            "type": "double_door",
            "width": 1.8,
            "open_probability": 0.9,
            "height": 2.1
        },
        {
            "from_zone": "Cafeteria",
            "to_zone": "Main Corridor",
            "type": "wide_opening",
            "width": 3.0,
            "open_probability": 1.0,
            "height": 2.4
        }
    ]
    
    # Configuração dos agentes
    agent_config = AgentConfig(
        activity_distribution={
            AgentActivity.SEATED_QUIET: 0.4,
            AgentActivity.SEATED_TYPING: 0.2,
            AgentActivity.STANDING: 0.1,
            AgentActivity.WALKING: 0.2,
            AgentActivity.TALKING: 0.1
        },
        age_distribution={
            "child": 0.3,
            "teen": 0.5,
            "adult": 0.2
        },
        metabolic_rates={
            AgentActivity.SEATED_QUIET: 1.0,
            AgentActivity.SEATED_TYPING: 1.2,
            AgentActivity.STANDING: 1.4,
            AgentActivity.WALKING: 2.0,
            AgentActivity.TALKING: 1.3
        },
        mask_wearing_prob=0.4,
        vaccination_rate=0.8,
        compliance_to_rules=0.6,
        avg_stay_duration_hours=6.0
    )
    
    return BuildingScenario(
        name="Modern School Complex",
        building_type=BuildingType.SCHOOL,
        description="A modern school with multiple classrooms, library, and cafeteria",
        total_width=50.0,  # 50 metros
        total_height=30.0,  # 30 metros
        floor_height=3.0,
        zones=zones,
        connections=connections,
        agent_config=agent_config,
        total_occupants=200,
        initial_infected_ratio=0.03,
        overall_ventilation_strategy="demand_controlled_vav",
        co2_setpoint=800,
        temperature_setpoint=21.0,
        humidity_setpoint=45.0,
        default_materials=materials,
        operation_hours=(7.0, 17.0),
        occupancy_schedule={
            "morning_classes": 0.9,
            "lunch_break": 0.5,
            "afternoon_classes": 0.8,
            "after_school": 0.2
        }
    )

def create_office_scenario() -> BuildingScenario:
    """Cria cenário de escritório moderno."""
    
    # Materiais padrão para escritórios
    materials = {
        MaterialType.MDF: MaterialProperties(
            name="Office Furniture MDF",
            material_type=MaterialType.MDF,
            hcho_emission_rate=1.5e-11,  # ~54 µg/m²/h
            voc_emission_rate=2.8e-11,
            decay_rate=0.008,
            surface_factor=2.0,
            age_days=365
        ),
        MaterialType.CARPET: MaterialProperties(
            name="Office Carpet",
            material_type=MaterialType.CARPET,
            hcho_emission_rate=2.2e-11,  # ~79 µg/m²/h
            voc_emission_rate=4.5e-11,
            decay_rate=0.015,
            surface_factor=1.2,
            age_days=180
        ),
        MaterialType.GLASS: MaterialProperties(
            name="Glass Partitions",
            material_type=MaterialType.GLASS,
            hcho_emission_rate=1.0e-12,  # baixa emissão
            voc_emission_rate=1.0e-12,
            decay_rate=0.0,
            surface_factor=1.0,
            age_days=0
        ),
        MaterialType.METAL: MaterialProperties(
            name="Metal Furniture",
            material_type=MaterialType.METAL,
            hcho_emission_rate=1.0e-13,  # muito baixa
            voc_emission_rate=1.0e-13,
            decay_rate=0.0,
            surface_factor=1.0,
            age_days=0
        ),
        MaterialType.PLASTIC: MaterialProperties(
            name="Plastic Components",
            material_type=MaterialType.PLASTIC,
            hcho_emission_rate=8.0e-12,  # ~29 µg/m²/h
            voc_emission_rate=2.0e-11,
            decay_rate=0.02,
            surface_factor=1.5,
            age_days=90
        )
    }
    
    # Configuração de zonas
    zones = [
        ZoneConfig(
            zone_type=ZoneType.OFFICE_SPACE,
            name="Open Office Area",
            width_ratio=0.4,
            height_ratio=0.6,
            materials=[
                {"type": MaterialType.LATEX_PAINT, "surface": "walls", "age_days": 60},
                {"type": MaterialType.CARPET, "surface": "floor", "age_days": 180},
                {"type": MaterialType.MDF, "surface": "desks", "density": 0.5, "age_days": 365},
                {"type": MaterialType.GLASS, "surface": "partitions", "density": 0.3, "age_days": 0}
            ],
            occupancy_density=0.8,  # pessoas/m² (espaço mais aberto)
            target_ach=4.0,
            ventilation_mode=VentilationMode.MECHANICAL,
            has_windows=True,
            window_area_ratio=0.4,
            air_inlets=[(0.3, 0.9), (0.7, 0.9)],
            air_outlets=[(0.5, 0.1)],
            equipment_heat_gain=20.0,  # computadores
            lighting_density=15.0
        ),
        ZoneConfig(
            zone_type=ZoneType.MEETING_ROOM,
            name="Meeting Room A",
            width_ratio=0.15,
            height_ratio=0.3,
            materials=[
                {"type": MaterialType.LATEX_PAINT, "surface": "walls", "age_days": 30},
                {"type": MaterialType.CARPET, "surface": "floor", "age_days": 90},
                {"type": MaterialType.MDF, "surface": "table", "density": 0.6, "age_days": 180},
                {"type": MaterialType.FABRIC, "surface": "chairs", "density": 0.4, "age_days": 365}
            ],
            occupancy_density=1.2,  # mais densa durante reuniões
            target_ach=6.0,  # maior ventilação para maior ocupação
            ventilation_mode=VentilationMode.MECHANICAL,
            has_windows=False,
            window_area_ratio=0.0
        ),
        ZoneConfig(
            zone_type=ZoneType.MEETING_ROOM,
            name="Meeting Room B",
            width_ratio=0.15,
            height_ratio=0.3,
            materials=[
                {"type": MaterialType.LATEX_PAINT, "surface": "walls", "age_days": 30},
                {"type": MaterialType.CARPET, "surface": "floor", "age_days": 90},
                {"type": MaterialType.MDF, "surface": "table", "density": 0.6, "age_days": 180}
            ],
            occupancy_density=1.2,
            target_ach=6.0,
            ventilation_mode=VentilationMode.MECHANICAL
        ),
        ZoneConfig(
            zone_type=ZoneType.CORRIDOR,
            name="Office Corridor",
            width_ratio=0.1,
            height_ratio=1.0,
            materials=[
                {"type": MaterialType.LATEX_PAINT, "surface": "walls", "age_days": 60},
                {"type": MaterialType.VINYL_FLOORING, "surface": "floor", "age_days": 365}
            ],
            occupancy_density=0.05,
            target_ach=3.0,
            ventilation_mode=VentilationMode.MIXED,
            furniture_density=0.05
        ),
        ZoneConfig(
            zone_type=ZoneType.CAFETERIA,
            name="Office Kitchen",
            width_ratio=0.2,
            height_ratio=0.4,
            materials=[
                {"type": MaterialType.CERAMIC, "surface": "walls", "age_days": 180},
                {"type": MaterialType.CERAMIC, "surface": "floor", "age_days": 180},
                {"type": MaterialType.METAL, "surface": "appliances", "density": 0.4, "age_days": 365}
            ],
            occupancy_density=0.3,
            target_ach=10.0,  # alta ventilação para cozinha
            ventilation_mode=VentilationMode.MECHANICAL,
            equipment_heat_gain=30.0  # equipamentos de cozinha
        )
    ]
    
    # Conexões entre zonas
    connections = [
        {
            "from_zone": "Open Office Area",
            "to_zone": "Office Corridor",
            "type": "open_plan",
            "width": 5.0,
            "open_probability": 1.0,
            "height": 2.4
        },
        {
            "from_zone": "Meeting Room A",
            "to_zone": "Office Corridor",
            "type": "door",
            "width": 0.9,
            "open_probability": 0.5,
            "height": 2.1
        },
        {
            "from_zone": "Meeting Room B",
            "to_zone": "Office Corridor",
            "type": "door",
            "width": 0.9,
            "open_probability": 0.5,
            "height": 2.1
        },
        {
            "from_zone": "Office Kitchen",
            "to_zone": "Office Corridor",
            "type": "door",
            "width": 1.2,
            "open_probability": 0.8,
            "height": 2.1
        }
    ]
    
    # Configuração dos agentes
    agent_config = AgentConfig(
        activity_distribution={
            AgentActivity.SEATED_TYPING: 0.5,
            AgentActivity.SEATED_QUIET: 0.2,
            AgentActivity.STANDING: 0.1,
            AgentActivity.WALKING: 0.1,
            AgentActivity.TALKING: 0.05,
            AgentActivity.EATING: 0.05
        },
        age_distribution={
            "young_adult": 0.2,
            "adult": 0.6,
            "senior": 0.2
        },
        metabolic_rates={
            AgentActivity.SEATED_TYPING: 1.2,
            AgentActivity.SEATED_QUIET: 1.0,
            AgentActivity.STANDING: 1.4,
            AgentActivity.WALKING: 2.0,
            AgentActivity.TALKING: 1.3,
            AgentActivity.EATING: 1.1
        },
        mask_wearing_prob=0.2,
        vaccination_rate=0.85,
        compliance_to_rules=0.9,
        avg_stay_duration_hours=9.0
    )
    
    return BuildingScenario(
        name="Modern Office Building",
        building_type=BuildingType.OFFICE,
        description="A modern open-plan office with meeting rooms and kitchen",
        total_width=40.0,  # 40 metros
        total_height=25.0,  # 25 metros
        floor_height=2.8,
        zones=zones,
        connections=connections,
        agent_config=agent_config,
        total_occupants=80,
        initial_infected_ratio=0.02,
        overall_ventilation_strategy="demand_controlled_vav",
        co2_setpoint=700,  # mais rigoroso para escritório
        temperature_setpoint=22.5,
        humidity_setpoint=50.0,
        default_materials=materials,
        operation_hours=(8.0, 19.0),
        occupancy_schedule={
            "morning": 0.6,
            "midday": 0.8,
            "afternoon": 0.9,
            "evening": 0.3
        }
    )

def create_gym_scenario() -> BuildingScenario:
    """Cria cenário de academia/ginásio."""
    
    # Materiais padrão para academias
    materials = {
        MaterialType.RUBBER: MaterialProperties(
            name="Rubber Flooring",
            material_type=MaterialType.RUBBER,
            hcho_emission_rate=8.0e-12,  # ~29 µg/m²/h
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
        MaterialType.PLASTIC: MaterialProperties(
            name="Plastic Equipment",
            material_type=MaterialType.PLASTIC,
            hcho_emission_rate=5.0e-12,  # ~18 µg/m²/h
            voc_emission_rate=1.5e-11,
            decay_rate=0.015,
            surface_factor=1.8,
            age_days=90
        ),
        MaterialType.CONCRETE: MaterialProperties(
            name="Concrete Walls",
            material_type=MaterialType.CONCRETE,
            hcho_emission_rate=1.0e-12,
            voc_emission_rate=1.0e-12,
            decay_rate=0.0,
            surface_factor=1.0,
            age_days=0
        ),
        MaterialType.FABRIC: MaterialProperties(
            name="Fabric Mats",
            material_type=MaterialType.FABRIC,
            hcho_emission_rate=2.0e-11,  # ~72 µg/m²/h
            voc_emission_rate=4.0e-11,
            decay_rate=0.03,
            surface_factor=1.3,
            age_days=30
        )
    }
    
    # Configuração de zonas
    zones = [
        ZoneConfig(
            zone_type=ZoneType.GYM_AREA,
            name="Main Gym Floor",
            width_ratio=0.5,
            height_ratio=0.7,
            materials=[
                {"type": MaterialType.RUBBER, "surface": "floor", "age_days": 180},
                {"type": MaterialType.CONCRETE, "surface": "walls", "age_days": 0},
                {"type": MaterialType.METAL, "surface": "equipment", "density": 0.4, "age_days": 0},
                {"type": MaterialType.PLASTIC, "surface": "equipment", "density": 0.3, "age_days": 90}
            ],
            occupancy_density=0.2,  # pessoas/m² (espaço para exercício)
            target_ach=10.0,  # alta ventilação para atividade física
            ventilation_mode=VentilationMode.MECHANICAL,
            has_windows=False,
            window_area_ratio=0.0,
            air_inlets=[(0.2, 0.9), (0.5, 0.9), (0.8, 0.9)],
            air_outlets=[(0.3, 0.1), (0.7, 0.1)],
            equipment_heat_gain=15.0,  # equipamentos ligados
            lighting_density=20.0  # iluminação forte
        ),
        ZoneConfig(
            zone_type=ZoneType.CORRIDOR,
            name="Entrance and Corridor",
            width_ratio=0.2,
            height_ratio=1.0,
            materials=[
                {"type": MaterialType.RUBBER, "surface": "floor", "age_days": 365},
                {"type": MaterialType.LATEX_PAINT, "surface": "walls", "age_days": 90}
            ],
            occupancy_density=0.1,
            target_ach=6.0,
            ventilation_mode=VentilationMode.MIXED,
            furniture_density=0.1
        ),
        ZoneConfig(
            zone_type=ZoneType.RESTROOM,
            name="Locker Room",
            width_ratio=0.15,
            height_ratio=0.3,
            materials=[
                {"type": MaterialType.CERAMIC, "surface": "walls", "age_days": 180},
                {"type": MaterialType.CERAMIC, "surface": "floor", "age_days": 180},
                {"type": MaterialType.METAL, "surface": "lockers", "density": 0.5, "age_days": 365}
            ],
            occupancy_density=0.3,
            target_ach=12.0,  # ventilação muito alta para umidade/odores
            ventilation_mode=VentilationMode.MECHANICAL,
            has_windows=False
        ),
        ZoneConfig(
            zone_type=ZoneType.RESTROOM,
            name="Showers and Toilets",
            width_ratio=0.15,
            height_ratio=0.3,
            materials=[
                {"type": MaterialType.CERAMIC, "surface": "walls", "age_days": 90},
                {"type": MaterialType.CERAMIC, "surface": "floor", "age_days": 90},
                {"type": MaterialType.PLASTIC, "surface": "partitions", "density": 0.4, "age_days": 180}
            ],
            occupancy_density=0.2,
            target_ach=15.0,  # ventilação extremamente alta
            ventilation_mode=VentilationMode.MECHANICAL
        )
    ]
    
    # Conexões entre zonas
    connections = [
        {
            "from_zone": "Main Gym Floor",
            "to_zone": "Entrance and Corridor",
            "type": "wide_opening",
            "width": 4.0,
            "open_probability": 1.0,
            "height": 2.8
        },
        {
            "from_zone": "Locker Room",
            "to_zone": "Entrance and Corridor",
            "type": "door",
            "width": 1.2,
            "open_probability": 0.8,
            "height": 2.1
        },
        {
            "from_zone": "Showers and Toilets",
            "to_zone": "Locker Room",
            "type": "door",
            "width": 0.9,
            "open_probability": 0.9,
            "height": 2.1
        }
    ]
    
    # Configuração dos agentes
    agent_config = AgentConfig(
        activity_distribution={
            AgentActivity.EXERCISING_LIGHT: 0.3,
            AgentActivity.EXERCISING_INTENSE: 0.4,
            AgentActivity.WALKING: 0.1,
            AgentActivity.STANDING: 0.1,
            AgentActivity.TALKING: 0.05,
            AgentActivity.SEATED_QUIET: 0.05
        },
        age_distribution={
            "young_adult": 0.5,
            "adult": 0.4,
            "senior": 0.1
        },
        metabolic_rates={
            AgentActivity.EXERCISING_LIGHT: 3.5,
            AgentActivity.EXERCISING_INTENSE: 6.0,
            AgentActivity.WALKING: 2.0,
            AgentActivity.STANDING: 1.4,
            AgentActivity.TALKING: 1.3,
            AgentActivity.SEATED_QUIET: 1.0
        },
        mask_wearing_prob=0.1,  # baixo uso de máscara em academia
        vaccination_rate=0.75,
        compliance_to_rules=0.7,
        avg_stay_duration_hours=1.5  # estadia mais curta
    )
    
    return BuildingScenario(
        name="Fitness Center",
        building_type=BuildingType.GYM,
        description="A modern fitness center with gym floor and locker rooms",
        total_width=30.0,  # 30 metros
        total_height=20.0,  # 20 metros
        floor_height=4.0,  # pé-direito mais alto
        zones=zones,
        connections=connections,
        agent_config=agent_config,
        total_occupants=50,
        initial_infected_ratio=0.01,
        overall_ventilation_strategy="constant_volume",
        co2_setpoint=1000,  # mais alto devido à atividade física
        temperature_setpoint=20.0,  # mais frio para conforto durante exercício
        humidity_setpoint=50.0,
        default_materials=materials,
        operation_hours=(6.0, 23.0),  # horários estendidos
        occupancy_schedule={
            "morning_peak": 0.4,
            "afternoon": 0.6,
            "evening_peak": 0.9,
            "late_evening": 0.3
        }
    )

def create_hospital_scenario() -> BuildingScenario:
    """Cria cenário de hospital."""
    
    # Materiais para hospital (baixa emissão, fácil limpeza)
    materials = {
        MaterialType.METAL: MaterialProperties(
            name="Stainless Steel",
            material_type=MaterialType.METAL,
            hcho_emission_rate=1.0e-14,  # muito baixa
            voc_emission_rate=1.0e-14,
            decay_rate=0.0,
            surface_factor=1.0,
            age_days=0
        ),
        MaterialType.GLASS: MaterialProperties(
            name="Hospital Glass",
            material_type=MaterialType.GLASS,
            hcho_emission_rate=1.0e-14,
            voc_emission_rate=1.0e-14,
            decay_rate=0.0,
            surface_factor=1.0,
            age_days=0
        ),
        MaterialType.CERAMIC: MaterialProperties(
            name="Ceramic Tiles",
            material_type=MaterialType.CERAMIC,
            hcho_emission_rate=1.0e-13,
            voc_emission_rate=1.0e-13,
            decay_rate=0.0,
            surface_factor=1.0,
            age_days=0
        ),
        MaterialType.PLASTIC: MaterialProperties(
            name="Medical Grade Plastic",
            material_type=MaterialType.PLASTIC,
            hcho_emission_rate=1.0e-12,
            voc_emission_rate=1.0e-12,
            decay_rate=0.005,
            surface_factor=1.2,
            age_days=30
        ),
        MaterialType.FABRIC: MaterialProperties(
            name="Hospital Curtains",
            material_type=MaterialType.FABRIC,
            hcho_emission_rate=5.0e-12,
            voc_emission_rate=2.0e-11,
            decay_rate=0.02,
            surface_factor=1.5,
            age_days=7  # lavagem frequente
        )
    }
    
    # Configuração de zonas
    zones = [
        ZoneConfig(
            zone_type=ZoneType.WAITING_ROOM,
            name="Main Waiting Room",
            width_ratio=0.3,
            height_ratio=0.4,
            materials=[
                {"type": MaterialType.CERAMIC, "surface": "floor", "age_days": 90},
                {"type": MaterialType.LATEX_PAINT, "surface": "walls", "age_days": 30},
                {"type": MaterialType.PLASTIC, "surface": "chairs", "density": 0.5, "age_days": 180},
                {"type": MaterialType.FABRIC, "surface": "curtains", "density": 0.2, "age_days": 7}
            ],
            occupancy_density=0.8,
            target_ach=8.0,  # alta ventilação para reduzir infecções
            ventilation_mode=VentilationMode.MECHANICAL,
            has_windows=True,
            window_area_ratio=0.3,
            air_inlets=[(0.2, 0.9), (0.8, 0.9)],
            air_outlets=[(0.5, 0.1)],
            equipment_heat_gain=10.0,
            lighting_density=15.0
        ),
        ZoneConfig(
            zone_type=ZoneType.CORRIDOR,
            name="Hospital Corridor",
            width_ratio=0.1,
            height_ratio=1.0,
            materials=[
                {"type": MaterialType.CERAMIC, "surface": "floor", "age_days": 180},
                {"type": MaterialType.LATEX_PAINT, "surface": "walls", "age_days": 60}
            ],
            occupancy_density=0.2,
            target_ach=6.0,
            ventilation_mode=VentilationMode.MECHANICAL,
            furniture_density=0.1
        ),
        ZoneConfig(
            zone_type=ZoneType.PATIENT_ROOM,
            name="Patient Room A",
            width_ratio=0.15,
            height_ratio=0.3,
            materials=[
                {"type": MaterialType.CERAMIC, "surface": "floor", "age_days": 30},
                {"type": MaterialType.LATEX_PAINT, "surface": "walls", "age_days": 7},
                {"type": MaterialType.METAL, "surface": "bed", "density": 0.3, "age_days": 0},
                {"type": MaterialType.PLASTIC, "surface": "furniture", "density": 0.3, "age_days": 30}
            ],
            occupancy_density=0.1,  # paciente + visitantes
            target_ach=12.0,  # ventilação muito alta
            ventilation_mode=VentilationMode.DISPLACEMENT,  # ventilação por deslocamento
            has_windows=True,
            window_area_ratio=0.4,
            air_inlets=[(0.5, 0.1)],  # entrada baixa
            air_outlets=[(0.5, 0.9)]  # saída alta
        ),
        ZoneConfig(
            zone_type=ZoneType.PATIENT_ROOM,
            name="Patient Room B",
            width_ratio=0.15,
            height_ratio=0.3,
            materials=[
                {"type": MaterialType.CERAMIC, "surface": "floor", "age_days": 30},
                {"type": MaterialType.LATEX_PAINT, "surface": "walls", "age_days": 7},
                {"type": MaterialType.METAL, "surface": "bed", "density": 0.3, "age_days": 0}
            ],
            occupancy_density=0.1,
            target_ach=12.0,
            ventilation_mode=VentilationMode.DISPLACEMENT,
            has_windows=True,
            window_area_ratio=0.4
        ),
        ZoneConfig(
            zone_type=ZoneType.OPERATING_ROOM,
            name="Operating Room",
            width_ratio=0.2,
            height_ratio=0.4,
            materials=[
                {"type": MaterialType.CERAMIC, "surface": "walls", "age_days": 0},
                {"type": MaterialType.CERAMIC, "surface": "floor", "age_days": 0},
                {"type": MaterialType.METAL, "surface": "equipment", "density": 0.4, "age_days": 0},
                {"type": MaterialType.GLASS, "surface": "windows", "age_days": 0}
            ],
            occupancy_density=0.15,  # equipe cirúrgica
            target_ach=20.0,  # ventilação extremamente alta
            ventilation_mode=VentilationMode.MECHANICAL,
            has_windows=False,
            window_area_ratio=0.0,
            air_inlets=[(0.5, 0.9)],  # fluxo laminar do teto
            air_outlets=[(0.1, 0.1), (0.9, 0.1)],  # múltiplas saídas
            equipment_heat_gain=30.0,  # equipamentos cirúrgicos
            lighting_density=30.0  # iluminação cirúrgica
        )
    ]
    
    # Conexões entre zonas
    connections = [
        {
            "from_zone": "Main Waiting Room",
            "to_zone": "Hospital Corridor",
            "type": "double_door",
            "width": 2.4,
            "open_probability": 0.9,
            "height": 2.4
        },
        {
            "from_zone": "Patient Room A",
            "to_zone": "Hospital Corridor",
            "type": "door",
            "width": 1.2,
            "open_probability": 0.5,
            "height": 2.1
        },
        {
            "from_zone": "Patient Room B",
            "to_zone": "Hospital Corridor",
            "type": "door",
            "width": 1.2,
            "open_probability": 0.5,
            "height": 2.1
        },
        {
            "from_zone": "Operating Room",
            "to_zone": "Hospital Corridor",
            "type": "airlock",
            "width": 1.8,
            "open_probability": 0.3,  # mantido fechado frequentemente
            "height": 2.4
        }
    ]
    
    # Configuração dos agentes
    agent_config = AgentConfig(
        activity_distribution={
            AgentActivity.SEATED_QUIET: 0.3,
            AgentActivity.STANDING: 0.2,
            AgentActivity.WALKING: 0.2,
            AgentActivity.TALKING: 0.1,
            AgentActivity.COUGHING: 0.05,
            AgentActivity.SNEEZING: 0.05,
            AgentActivity.SLEEPING: 0.1
        },
        age_distribution={
            "child": 0.1,
            "adult": 0.6,
            "senior": 0.3
        },
        metabolic_rates={
            AgentActivity.SEATED_QUIET: 1.0,
            AgentActivity.STANDING: 1.4,
            AgentActivity.WALKING: 2.0,
            AgentActivity.TALKING: 1.3,
            AgentActivity.COUGHING: 1.5,
            AgentActivity.SNEEZING: 1.5,
            AgentActivity.SLEEPING: 0.7
        },
        mask_wearing_prob=0.95,  # uso quase universal de máscaras
        vaccination_rate=0.9,
        compliance_to_rules=0.95,
        avg_stay_duration_hours=4.0
    )
    
    return BuildingScenario(
        name="General Hospital",
        building_type=BuildingType.HOSPITAL,
        description="A general hospital with waiting room, patient rooms and operating room",
        total_width=50.0,
        total_height=35.0,
        floor_height=3.0,
        zones=zones,
        connections=connections,
        agent_config=agent_config,
        total_occupants=80,
        initial_infected_ratio=0.1,  # maior proporção inicial
        overall_ventilation_strategy="constant_volume_hepa",
        co2_setpoint=600,  # muito rigoroso
        temperature_setpoint=23.0,
        humidity_setpoint=40.0,  # mais seco para controle microbiano
        default_materials=materials,
        operation_hours=(0.0, 24.0),  # 24 horas
        occupancy_schedule={
            "morning_rounds": 0.8,
            "afternoon": 0.7,
            "evening": 0.5,
            "night": 0.3
        }
    )

def create_residential_scenario() -> BuildingScenario:
    """Cria cenário residencial."""
    
    # Materiais para residência
    materials = {
        MaterialType.WOOD: MaterialProperties(
            name="Wood Flooring",
            material_type=MaterialType.WOOD,
            hcho_emission_rate=2.5e-11,  # ~90 µg/m²/h
            voc_emission_rate=5.0e-11,
            decay_rate=0.01,
            surface_factor=1.0,
            age_days=365
        ),
        MaterialType.CARPET: MaterialProperties(
            name="Residential Carpet",
            material_type=MaterialType.CARPET,
            hcho_emission_rate=3.0e-11,  # ~108 µg/m²/h
            voc_emission_rate=6.0e-11,
            decay_rate=0.02,
            surface_factor=1.2,
            age_days=180
        ),
        MaterialType.LATEX_PAINT: MaterialProperties(
            name="Interior Paint",
            material_type=MaterialType.LATEX_PAINT,
            hcho_emission_rate=4.0e-11,  # ~144 µg/m²/h
            voc_emission_rate=7.0e-11,
            decay_rate=0.04,
            surface_factor=1.0,
            age_days=60
        ),
        MaterialType.FABRIC: MaterialProperties(
            name="Fabric Furniture",
            material_type=MaterialType.FABRIC,
            hcho_emission_rate=2.0e-11,
            voc_emission_rate=4.0e-11,
            decay_rate=0.03,
            surface_factor=2.0,
            age_days=90
        ),
        MaterialType.PLASTIC: MaterialProperties(
            name="Household Plastic",
            material_type=MaterialType.PLASTIC,
            hcho_emission_rate=3.0e-12,
            voc_emission_rate=1.0e-11,
            decay_rate=0.01,
            surface_factor=1.3,
            age_days=365
        )
    }
    
    # Configuração de zonas
    zones = [
        ZoneConfig(
            zone_type=ZoneType.LIVING_ROOM,
            name="Living Room",
            width_ratio=0.4,
            height_ratio=0.5,
            materials=[
                {"type": MaterialType.LATEX_PAINT, "surface": "walls", "age_days": 365},
                {"type": MaterialType.WOOD, "surface": "floor", "age_days": 730},
                {"type": MaterialType.FABRIC, "surface": "sofa", "density": 0.4, "age_days": 180},
                {"type": MaterialType.WOOD, "surface": "furniture", "density": 0.3, "age_days": 365}
            ],
            occupancy_density=0.1,
            target_ach=1.5,  # ventilação baixa (residencial)
            ventilation_mode=VentilationMode.NATURAL,
            has_windows=True,
            window_area_ratio=0.25,
            air_inlets=[(0.3, 0.8), (0.7, 0.8)],  # janelas
            air_outlets=[(0.2, 0.2)],  # portas/outras aberturas
            equipment_heat_gain=8.0,  # TV, eletrônicos
            lighting_density=8.0
        ),
        ZoneConfig(
            zone_type=ZoneType.KITCHEN,
            name="Kitchen",
            width_ratio=0.2,
            height_ratio=0.3,
            materials=[
                {"type": MaterialType.CERAMIC, "surface": "walls", "age_days": 180},
                {"type": MaterialType.CERAMIC, "surface": "floor", "age_days": 180},
                {"type": MaterialType.METAL, "surface": "appliances", "density": 0.4, "age_days": 365},
                {"type": MaterialType.PLASTIC, "surface": "cabinets", "density": 0.5, "age_days": 730}
            ],
            occupancy_density=0.15,
            target_ach=5.0,  # ventilação mais alta para cozinha
            ventilation_mode=VentilationMode.MIXED,
            has_windows=True,
            window_area_ratio=0.2,
            equipment_heat_gain=20.0  # fogão, forno
        ),
        ZoneConfig(
            zone_type=ZoneType.BEDROOM,
            name="Master Bedroom",
            width_ratio=0.2,
            height_ratio=0.4,
            materials=[
                {"type": MaterialType.LATEX_PAINT, "surface": "walls", "age_days": 730},
                {"type": MaterialType.CARPET, "surface": "floor", "age_days": 365},
                {"type": MaterialType.WOOD, "surface": "furniture", "density": 0.4, "age_days": 1095},
                {"type": MaterialType.FABRIC, "surface": "bed", "density": 0.5, "age_days": 180}
            ],
            occupancy_density=0.05,
            target_ach=1.0,  # ventilação muito baixa durante a noite
            ventilation_mode=VentilationMode.NATURAL,
            has_windows=True,
            window_area_ratio=0.3
        ),
        ZoneConfig(
            zone_type=ZoneType.BEDROOM,
            name="Children's Bedroom",
            width_ratio=0.2,
            height_ratio=0.4,
            materials=[
                {"type": MaterialType.LATEX_PAINT, "surface": "walls", "age_days": 365},
                {"type": MaterialType.CARPET, "surface": "floor", "age_days": 180},
                {"type": MaterialType.PLASTIC, "surface": "toys", "density": 0.3, "age_days": 30},
                {"type": MaterialType.WOOD, "surface": "furniture", "density": 0.3, "age_days": 365}
            ],
            occupancy_density=0.08,
            target_ach=1.0,
            ventilation_mode=VentilationMode.NATURAL,
            has_windows=True,
            window_area_ratio=0.3
        ),
        ZoneConfig(
            zone_type=ZoneType.RESTROOM,
            name="Bathroom",
            width_ratio=0.1,
            height_ratio=0.3,
            materials=[
                {"type": MaterialType.CERAMIC, "surface": "walls", "age_days": 90},
                {"type": MaterialType.CERAMIC, "surface": "floor", "age_days": 90},
                {"type": MaterialType.PLASTIC, "surface": "fixtures", "density": 0.4, "age_days": 180}
            ],
            occupancy_density=0.02,
            target_ach=3.0,  # ventilação média para banheiro
            ventilation_mode=VentilationMode.MIXED,
            has_windows=False
        )
    ]
    
    # Conexões entre zonas
    connections = [
        {
            "from_zone": "Living Room",
            "to_zone": "Kitchen",
            "type": "open_plan",
            "width": 3.0,
            "open_probability": 1.0,
            "height": 2.4
        },
        {
            "from_zone": "Living Room",
            "to_zone": "Master Bedroom",
            "type": "door",
            "width": 0.9,
            "open_probability": 0.7,
            "height": 2.1
        },
        {
            "from_zone": "Living Room",
            "to_zone": "Children's Bedroom",
            "type": "door",
            "width": 0.9,
            "open_probability": 0.8,
            "height": 2.1
        },
        {
            "from_zone": "Master Bedroom",
            "to_zone": "Bathroom",
            "type": "door",
            "width": 0.7,
            "open_probability": 0.6,
            "height": 2.0
        }
    ]
    
    # Configuração dos agentes
    agent_config = AgentConfig(
        activity_distribution={
            AgentActivity.SLEEPING: 0.3,
            AgentActivity.SEATED_QUIET: 0.2,
            AgentActivity.STANDING: 0.1,
            AgentActivity.WALKING: 0.1,
            AgentActivity.TALKING: 0.1,
            AgentActivity.EATING: 0.1,
            AgentActivity.SEATED_TYPING: 0.05,
            AgentActivity.EXERCISING_LIGHT: 0.05
        },
        age_distribution={
            "child": 0.2,
            "teen": 0.2,
            "adult": 0.4,
            "senior": 0.2
        },
        metabolic_rates={
            AgentActivity.SLEEPING: 0.7,
            AgentActivity.SEATED_QUIET: 1.0,
            AgentActivity.STANDING: 1.4,
            AgentActivity.WALKING: 2.0,
            AgentActivity.TALKING: 1.3,
            AgentActivity.EATING: 1.1,
            AgentActivity.SEATED_TYPING: 1.2,
            AgentActivity.EXERCISING_LIGHT: 3.5
        },
        mask_wearing_prob=0.05,  # muito baixo em casa
        vaccination_rate=0.7,
        compliance_to_rules=0.8,
        avg_stay_duration_hours=16.0  # maioria do tempo em casa
    )
    
    return BuildingScenario(
        name="Family Residence",
        building_type=BuildingType.RESIDENTIAL,
        description="A typical family home with living room, kitchen, bedrooms and bathroom",
        total_width=25.0,
        total_height=15.0,
        floor_height=2.6,
        zones=zones,
        connections=connections,
        agent_config=agent_config,
        total_occupants=4,
        initial_infected_ratio=0.25,  # maior chance em família
        overall_ventilation_strategy="natural_with_mechanical_boost",
        co2_setpoint=1000,  # menos rigoroso em residências
        temperature_setpoint=22.0,
        humidity_setpoint=55.0,
        default_materials=materials,
        operation_hours=(0.0, 24.0),  # 24 horas
        occupancy_schedule={
            "morning": 0.8,
            "daytime": 0.3,
            "evening": 0.9,
            "night": 1.0
        }
    )

# ============================================================================
# 4. CONSTANTES FÍSICAS VALIDADAS
# ============================================================================

# Emissões humanas (kg/s)
HUMAN_EMISSION_RATES = {
    'co2': {
        'sleeping': 4.7e-6,      # 0.7 MET
        'seated_quiet': 7.4e-6,  # 1.1 MET
        'seated_typing': 8.2e-6, # 1.2 MET
        'standing': 8.9e-6,      # 1.3 MET
        'walking': 1.3e-5,       # 2.0 MET
        'exercising_light': 2.2e-5,  # 3.5 MET
        'exercising_intense': 4.0e-5,  # 6.0 MET
        'talking': 9.5e-6,       # 1.4 MET
        'singing': 1.2e-5,       # 1.8 MET
        'eating': 7.4e-6,        # 1.1 MET
        'reading': 7.4e-6        # 1.1 MET
    },
    'vocs': {
        'baseline': 2.0e-9,  # kg/s-pessoa
        'active': 3.0e-9,
        'exercising': 5.0e-9
    },
    'quanta': {
        'breathing': 2.0 / 3600.0,   # quanta/s
        'talking': 10.0 / 3600.0,
        'singing': 100.0 / 3600.0,
        'coughing': 500.0 / 3600.0,
        'sneezing': 1000.0 / 3600.0,
        'exercising_light': 5.0 / 3600.0,
        'exercising_intense': 15.0 / 3600.0
    },
    'heat': {
        'sleeping': 70,      # W
        'seated_quiet': 100,
        'seated_typing': 115,
        'standing': 130,
        'walking': 200,
        'exercising_light': 350,
        'exercising_intense': 600
    }
}

# Limites de exposição
EXPOSURE_LIMITS = {
    'co2': {
        'excellent': 600,    # ppm
        'good': 800,
        'moderate': 1000,
        'poor': 1500,
        'unhealthy': 2000
    },
    'hcho': {
        'who_30min': 81.8,   # ppb
        'who_annual': 24.5,
        'epa_chronic': 13.1,
        'ashrae': 81.8,
        'action_level': 100.0
    },
    'pm25': {
        'who_24h': 15,       # µg/m³
        'who_annual': 5,
        'epa_24h': 35,
        'action_level': 50
    },
    'pm10': {
        'who_24h': 45,       # µg/m³
        'who_annual': 15,
        'epa_24h': 150
    },
    'voc': {
        'recommended': 500,  # ppb
        'action_level': 1000,
        'unhealthy': 2000
    },
    'temperature': {
        'comfort_min': 20.0,  # °C
        'comfort_max': 24.0,
        'acceptable_min': 18.0,
        'acceptable_max': 26.0
    },
    'humidity': {
        'comfort_min': 40.0,  # %
        'comfort_max': 60.0,
        'acceptable_min': 30.0,
        'acceptable_max': 70.0
    }
}

# Fatores de conversão
CONVERSION_FACTORS = {
    'co2_ppm_to_kgm3': 1.8e-6,     # 1 ppm = 1.8e-6 kg/m³ @ 25°C
    'co2_kgm3_to_ppm': 5.56e5,
    'hcho_ppb_to_kgm3': 1.23e-9,   # 1 ppb = 1.23e-9 kg/m³
    'hcho_kgm3_to_ppb': 8.13e8,
    'voc_ppb_to_kgm3': 1.0e-9,     # aproximado
    'voc_kgm3_to_ppb': 1.0e9,
    'pm25_ugm3_to_kgm3': 1.0e-9,
    'pm25_kgm3_to_ugm3': 1.0e9,
    'temperature_k_to_c': lambda x: x - 273.15,
    'temperature_c_to_k': lambda x: x + 273.15
}

# Constantes físicas
PHYSICAL_CONSTANTS = {
    'air_density': 1.204,           # kg/m³ @ 20°C
    'air_specific_heat': 1005,      # J/kg·K
    'air_viscosity': 1.81e-5,       # Pa·s @ 20°C
    'air_thermal_conductivity': 0.026,  # W/m·K
    'gravity': 9.81,                # m/s²
    'gas_constant': 287.05,         # J/kg·K para ar seco
    'molar_mass_air': 0.029,        # kg/mol
    'molar_mass_co2': 0.044,        # kg/mol
    'molar_mass_hcho': 0.030,       # kg/mol
    'molar_mass_water': 0.018       # kg/mol
}

# Eficiências de filtragem
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

# Fatores de redução por intervenções
INTERVENTION_EFFECTIVENESS = {
    'mask_surgical': {
        'virus_emission': 0.5,
        'virus_inhalation': 0.3
    },
    'mask_n95': {
        'virus_emission': 0.9,
        'virus_inhalation': 0.95
    },
    'air_cleaner_small': {
        'ach_equivalent': 2.0,  # ACH adicional
        'virus_removal': 0.7
    },
    'air_cleaner_large': {
        'ach_equivalent': 5.0,
        'virus_removal': 0.9
    },
    'uvc_upper_room': {
        'virus_inactivation': 0.8,  # por passagem
        'bacteria_inactivation': 0.95
    }
}

# ============================================================================
# 5. FUNÇÕES UTILITÁRIAS
# ============================================================================

def get_scenario_config(scenario_name: str) -> BuildingScenario:
    """Retorna configuração de cenário pelo nome."""
    scenarios = {
        "school": create_school_scenario,
        "office": create_office_scenario,
        "gym": create_gym_scenario,
        "hospital": create_hospital_scenario,
        "residential": create_residential_scenario
    }
    
    if scenario_name in scenarios:
        return scenarios[scenario_name]()
    else:
        return create_school_scenario()  # default

def calculate_zone_parameters(zone: ZoneConfig, total_width: float, total_height: float, floor_height: float = 2.7) -> Dict:
    """Calcula parâmetros físicos de uma zona."""
    width = zone.width_ratio * total_width
    height = zone.height_ratio * total_height
    area = width * height
    volume = area * floor_height
    
    max_occupants = int(area * zone.occupancy_density)
    
    # Ventilação requerida (ASHRAE 62.1: 2.5 L/s por pessoa + 0.3 L/s por m²)
    required_ventilation_lps = max_occupants * 2.5 + area * 0.3
    
    # Conversão para ACH
    required_ach = (required_ventilation_lps * 3.6) / volume if volume > 0 else 0
    
    return {
        'width': width,
        'height': height,
        'area': area,
        'volume': volume,
        'max_occupants': max_occupants,
        'required_ventilation_lps': required_ventilation_lps,
        'required_ach': required_ach,
        'window_area': area * zone.window_area_ratio if zone.has_windows else 0.0
    }

def validate_scenario(scenario: BuildingScenario) -> List[str]:
    """Valida um cenário e retorna lista de problemas."""
    issues = []
    
    # Verifica soma das proporções
    total_width_ratio = sum(z.width_ratio for z in scenario.zones)
    total_height_ratio = sum(z.height_ratio for z in scenario.zones)
    
    if abs(total_width_ratio - 1.0) > 0.01:
        issues.append(f"Total width ratio is {total_width_ratio:.2f}, should be 1.0 ±0.01")
    
    if abs(total_height_ratio - 1.0) > 0.01:
        issues.append(f"Total height ratio is {total_height_ratio:.2f}, should be 1.0 ±0.01")
    
    # Verifica ocupação
    total_max_occupants = sum(
        calculate_zone_parameters(z, scenario.total_width, scenario.total_height, scenario.floor_height)['max_occupants']
        for z in scenario.zones
    )
    
    if scenario.total_occupants > total_max_occupants * 1.1:  # 10% de tolerância
        issues.append(f"Total occupants ({scenario.total_occupants}) exceeds maximum capacity ({total_max_occupants})")
    
    # Verifica atividades dos agentes
    activity_sum = sum(scenario.agent_config.activity_distribution.values())
    if abs(activity_sum - 1.0) > 0.01:
        issues.append(f"Agent activity distribution sums to {activity_sum:.2f}, should be 1.0")
    
    # Verifica idade dos agentes
    age_sum = sum(scenario.agent_config.age_distribution.values())
    if abs(age_sum - 1.0) > 0.01:
        issues.append(f"Agent age distribution sums to {age_sum:.2f}, should be 1.0")
    
    # Verifica que todas as atividades têm taxa metabólica definida
    for activity in scenario.agent_config.activity_distribution.keys():
        if activity not in scenario.agent_config.metabolic_rates:
            issues.append(f"Activity {activity.value} has no metabolic rate defined")
    
    # Verifica materiais
    for zone in scenario.zones:
        for material_config in zone.materials:
            material_type = material_config.get('type')
            if material_type not in scenario.default_materials:
                issues.append(f"Zone '{zone.name}': Material {material_type} not defined in default materials")
    
    # Verifica conexões
    zone_names = [zone.name for zone in scenario.zones]
    for connection in scenario.connections:
        if connection['from_zone'] not in zone_names:
            issues.append(f"Connection references unknown zone: {connection['from_zone']}")
        if connection['to_zone'] not in zone_names:
            issues.append(f"Connection references unknown zone: {connection['to_zone']}")
    
    return issues

def create_custom_scenario(
    name: str,
    building_type: BuildingType,
    description: str,
    total_width: float,
    total_height: float,
    zones: List[ZoneConfig],
    connections: List[Dict[str, Any]],
    agent_config: AgentConfig,
    total_occupants: int,
    **kwargs
) -> BuildingScenario:
    """Cria um cenário personalizado com validação."""
    
    # Configuração padrão
    floor_height = kwargs.get('floor_height', 2.7)
    initial_infected_ratio = kwargs.get('initial_infected_ratio', 0.05)
    overall_ventilation_strategy = kwargs.get('overall_ventilation_strategy', 'demand_controlled')
    co2_setpoint = kwargs.get('co2_setpoint', 800)
    temperature_setpoint = kwargs.get('temperature_setpoint', 22.0)
    humidity_setpoint = kwargs.get('humidity_setpoint', 50.0)
    default_materials = kwargs.get('default_materials', {})
    operation_hours = kwargs.get('operation_hours', (8.0, 18.0))
    occupancy_schedule = kwargs.get('occupancy_schedule', {"default": 0.7})
    
    scenario = BuildingScenario(
        name=name,
        building_type=building_type,
        description=description,
        total_width=total_width,
        total_height=total_height,
        floor_height=floor_height,
        zones=zones,
        connections=connections,
        agent_config=agent_config,
        total_occupants=total_occupants,
        initial_infected_ratio=initial_infected_ratio,
        overall_ventilation_strategy=overall_ventilation_strategy,
        co2_setpoint=co2_setpoint,
        temperature_setpoint=temperature_setpoint,
        humidity_setpoint=humidity_setpoint,
        default_materials=default_materials,
        operation_hours=operation_hours,
        occupancy_schedule=occupancy_schedule
    )
    
    # Valida
    issues = validate_scenario(scenario)
    if issues:
        print(f"⚠️ Avisos para cenário '{name}':")
        for issue in issues:
            print(f"  - {issue}")
    
    return scenario

def calculate_scenario_statistics(scenario: BuildingScenario) -> Dict[str, Any]:
    """Calcula estatísticas detalhadas de um cenário."""
    total_area = 0.0
    total_volume = 0.0
    zone_stats = {}
    
    for zone in scenario.zones:
        params = calculate_zone_parameters(zone, scenario.total_width, 
                                          scenario.total_height, scenario.floor_height)
        total_area += params['area']
        total_volume += params['volume']
        
        zone_stats[zone.name] = {
            'area_m2': params['area'],
            'volume_m3': params['volume'],
            'max_occupants': params['max_occupants'],
            'occupancy_density': zone.occupancy_density,
            'target_ach': zone.target_ach,
            'required_ach': params['required_ach'],
            'ventilation_mode': zone.ventilation_mode.value,
            'window_area': params['window_area'],
            'materials_count': len(zone.materials)
        }
    
    # Percentual de ocupação
    occupancy_percentage = (scenario.total_occupants / 
                           sum(stats['max_occupants'] for stats in zone_stats.values())) * 100
    
    # Distribuição de tipos de ventilação
    ventilation_modes = {}
    for zone in scenario.zones:
        mode = zone.ventilation_mode.value
        ventilation_modes[mode] = ventilation_modes.get(mode, 0) + 1
    
    return {
        'total_area_m2': total_area,
        'total_volume_m3': total_volume,
        'occupancy_percentage': occupancy_percentage,
        'zone_count': len(scenario.zones),
        'connection_count': len(scenario.connections),
        'material_types_count': len(scenario.default_materials),
        'zone_statistics': zone_stats,
        'ventilation_distribution': ventilation_modes,
        'agent_activity_distribution': scenario.agent_config.activity_distribution,
        'agent_age_distribution': scenario.agent_config.age_distribution,
        'operation_hours': scenario.operation_hours
    }

# ============================================================================
# 6. DICIONÁRIO CONSOLIDADO
# ============================================================================

CONSTANTS = {
    # Cenários
    'SCENARIOS': {
        'school': create_school_scenario,
        'office': create_office_scenario,
        'gym': create_gym_scenario,
        'hospital': create_hospital_scenario,
        'residential': create_residential_scenario
    },
    
    # Emissões
    'HUMAN_EMISSION_RATES': HUMAN_EMISSION_RATES,
    
    # Limites
    'EXPOSURE_LIMITS': EXPOSURE_LIMITS,
    
    # Conversão
    'CONVERSION_FACTORS': CONVERSION_FACTORS,
    
    # Física
    'PHYSICAL_CONSTANTS': PHYSICAL_CONSTANTS,
    
    # Filtragem
    'FILTRATION_EFFICIENCIES': FILTRATION_EFFICIENCIES,
    
    # Intervenções
    'INTERVENTION_EFFECTIVENESS': INTERVENTION_EFFECTIVENESS,
    
    # Física padrão
    'PHYSICS_DEFAULTS': {
        'cell_size': 0.1,
        'dt_max': 0.1,
        'stability_factor': 0.8,
        'd_molecular_co2': 1.6e-5,
        'd_turbulent': 5.0e-3
    },
    
    # Constantes de materiais (valores típicos)
    'MATERIAL_EMISSION_RANGES': {
        'hcho': {
            'low': 1e-12,    # kg/m²/s
            'medium': 1e-11,
            'high': 1e-10,
            'very_high': 1e-9
        },
        'voc': {
            'low': 1e-12,
            'medium': 1e-11,
            'high': 1e-10,
            'very_high': 1e-9
        }
    },
    
    # Referências de ACH
    'ACH_REFERENCES': {
        'residential': 0.5,
        'office': 2.0,
        'classroom': 3.0,
        'hospital_room': 6.0,
        'operating_room': 20.0,
        'cleanroom': 100.0
    }
}

# ============================================================================
# 7. FUNÇÕES DE AJUSTE DINÂMICO
# ============================================================================

def adjust_scenario_for_season(scenario: BuildingScenario, season: str) -> BuildingScenario:
    """Ajusta um cenário para diferentes estações do ano."""
    seasonal_factors = {
        'winter': {
            'temperature_setpoint': 21.0,
            'humidity_setpoint': 40.0,
            'ach_multiplier': 0.8,  # menor ventilação para economizar energia
            'window_open_probability': 0.1
        },
        'spring': {
            'temperature_setpoint': 22.0,
            'humidity_setpoint': 50.0,
            'ach_multiplier': 1.0,
            'window_open_probability': 0.5
        },
        'summer': {
            'temperature_setpoint': 24.0,
            'humidity_setpoint': 55.0,
            'ach_multiplier': 1.2,  # maior ventilação
            'window_open_probability': 0.7
        },
        'fall': {
            'temperature_setpoint': 21.0,
            'humidity_setpoint': 45.0,
            'ach_multiplier': 0.9,
            'window_open_probability': 0.3
        }
    }
    
    if season not in seasonal_factors:
        return scenario
    
    factors = seasonal_factors[season]
    
    # Ajusta o cenário
    scenario.temperature_setpoint = factors['temperature_setpoint']
    scenario.humidity_setpoint = factors['humidity_setpoint']
    
    # Ajusta ACH das zonas
    for zone in scenario.zones:
        if zone.ventilation_mode in [VentilationMode.NATURAL, VentilationMode.MIXED]:
            zone.target_ach *= factors['ach_multiplier']
    
    return scenario

def adjust_scenario_for_pandemic(scenario: BuildingScenario, pandemic_level: str = 'medium') -> BuildingScenario:
    """Ajusta um cenário para diferentes níveis de pandemia."""
    pandemic_levels = {
        'low': {
            'mask_probability': 0.2,
            'ach_multiplier': 1.2,
            'occupancy_multiplier': 0.9
        },
        'medium': {
            'mask_probability': 0.5,
            'ach_multiplier': 1.5,
            'occupancy_multiplier': 0.7
        },
        'high': {
            'mask_probability': 0.8,
            'ach_multiplier': 2.0,
            'occupancy_multiplier': 0.5
        },
        'critical': {
            'mask_probability': 0.95,
            'ach_multiplier': 3.0,
            'occupancy_multiplier': 0.3
        }
    }
    
    if pandemic_level not in pandemic_levels:
        return scenario
    
    factors = pandemic_levels[pandemic_level]
    
    # Ajusta comportamento dos agentes
    scenario.agent_config.mask_wearing_prob = factors['mask_probability']
    
    # Ajusta ventilação
    for zone in scenario.zones:
        zone.target_ach *= factors['ach_multiplier']
    
    # Ajusta ocupação
    scenario.total_occupants = int(scenario.total_occupants * factors['occupancy_multiplier'])
    
    return scenario

def create_scenario_from_json(json_data: Dict) -> BuildingScenario:
    """Cria um cenário a partir de dados JSON."""
    # Implementação básica - em produção seria mais completa
    try:
        # Converte enums de strings
        building_type = BuildingType(json_data['building_type'])
        
        # Converte zonas
        zones = []
        for zone_data in json_data['zones']:
            zone_type = ZoneType(zone_data['zone_type'])
            ventilation_mode = VentilationMode(zone_data['ventilation_mode'])
            
            zone = ZoneConfig(
                zone_type=zone_type,
                name=zone_data['name'],
                width_ratio=zone_data['width_ratio'],
                height_ratio=zone_data['height_ratio'],
                materials=zone_data.get('materials', []),
                occupancy_density=zone_data['occupancy_density'],
                target_ach=zone_data['target_ach'],
                ventilation_mode=ventilation_mode,
                has_windows=zone_data.get('has_windows', False),
                window_area_ratio=zone_data.get('window_area_ratio', 0.0),
                furniture_density=zone_data.get('furniture_density', 0.3)
            )
            zones.append(zone)
        
        # Converte configuração dos agentes
        agent_data = json_data['agent_config']
        
        # Converte atividades
        activity_distribution = {}
        for activity_str, prob in agent_data['activity_distribution'].items():
            activity = AgentActivity(activity_str)
            activity_distribution[activity] = prob
        
        # Converte taxas metabólicas
        metabolic_rates = {}
        for activity_str, rate in agent_data['metabolic_rates'].items():
            activity = AgentActivity(activity_str)
            metabolic_rates[activity] = rate
        
        agent_config = AgentConfig(
            activity_distribution=activity_distribution,
            age_distribution=agent_data['age_distribution'],
            metabolic_rates=metabolic_rates,
            mask_wearing_prob=agent_data.get('mask_wearing_prob', 0.3),
            vaccination_rate=agent_data.get('vaccination_rate', 0.7),
            compliance_to_rules=agent_data.get('compliance_to_rules', 0.8),
            avg_stay_duration_hours=agent_data.get('avg_stay_duration_hours', 8.0)
        )
        
        # Converte materiais padrão
        default_materials = {}
        for material_name, material_data in json_data.get('default_materials', {}).items():
            material_type = MaterialType(material_data['material_type'])
            material_props = MaterialProperties(
                name=material_data['name'],
                material_type=material_type,
                hcho_emission_rate=material_data['hcho_emission_rate'],
                voc_emission_rate=material_data['voc_emission_rate'],
                decay_rate=material_data['decay_rate'],
                surface_factor=material_data['surface_factor'],
                age_days=material_data.get('age_days', 0.0),
                temperature_coefficient=material_data.get('temperature_coefficient', 0.02)
            )
            default_materials[material_type] = material_props
        
        # Cria cenário
        scenario = BuildingScenario(
            name=json_data['name'],
            building_type=building_type,
            description=json_data['description'],
            total_width=json_data['total_width'],
            total_height=json_data['total_height'],
            floor_height=json_data.get('floor_height', 2.7),
            zones=zones,
            connections=json_data.get('connections', []),
            agent_config=agent_config,
            total_occupants=json_data['total_occupants'],
            initial_infected_ratio=json_data.get('initial_infected_ratio', 0.05),
            overall_ventilation_strategy=json_data.get('overall_ventilation_strategy', 'demand_controlled'),
            co2_setpoint=json_data.get('co2_setpoint', 800),
            temperature_setpoint=json_data.get('temperature_setpoint', 22.0),
            humidity_setpoint=json_data.get('humidity_setpoint', 50.0),
            default_materials=default_materials,
            operation_hours=tuple(json_data.get('operation_hours', (8.0, 18.0))),
            occupancy_schedule=json_data.get('occupancy_schedule', {})
        )
        
        return scenario
        
    except Exception as e:
        print(f"Erro ao criar cenário a partir de JSON: {e}")
        # Retorna cenário padrão em caso de erro
        return create_school_scenario()

# ============================================================================
# 8. FUNÇÕES PARA VISUALIZAÇÃO
# ============================================================================

def scenario_to_dataframe(scenario: BuildingScenario) -> Dict[str, pd.DataFrame]:
    """Converte um cenário em DataFrames para visualização."""
    # DataFrame de zonas
    zone_data = []
    for zone in scenario.zones:
        params = calculate_zone_parameters(zone, scenario.total_width, 
                                          scenario.total_height, scenario.floor_height)
        
        zone_data.append({
            'Nome': zone.name,
            'Tipo': zone.zone_type.value,
            'Largura (m)': params['width'],
            'Altura (m)': params['height'],
            'Área (m²)': params['area'],
            'Volume (m³)': params['volume'],
            'Densidade Ocupação (pessoas/m²)': zone.occupancy_density,
            'Ocupantes Máx.': params['max_occupants'],
            'ACH Alvo': zone.target_ach,
            'ACH Requerido': params['required_ach'],
            'Modo Ventilação': zone.ventilation_mode.value,
            'Janelas': 'Sim' if zone.has_windows else 'Não',
            'Área Janela (%)': zone.window_area_ratio * 100,
            'Materiais': len(zone.materials)
        })
    
    zones_df = pd.DataFrame(zone_data)
    
    # DataFrame de agentes
    agent_data = {
        'Atividade': [a.value for a in scenario.agent_config.activity_distribution.keys()],
        'Probabilidade': list(scenario.agent_config.activity_distribution.values())
    }
    
    agents_df = pd.DataFrame(agent_data)
    
    # DataFrame de idades
    age_df = pd.DataFrame(list(scenario.agent_config.age_distribution.items()),
                         columns=['Faixa Etária', 'Probabilidade'])
    
    return {
        'zones': zones_df,
        'agent_activities': agents_df,
        'agent_ages': age_df
    }

# ============================================================================
# 9. EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CONFIGURAÇÃO FINAL - SIMULADOR IAQ AVANÇADO")
    print("=" * 80)
    
    # Testa todos os cenários
    scenarios_to_test = ['school', 'office', 'gym', 'hospital', 'residential']
    
    for scenario_name in scenarios_to_test:
        print(f"\n📊 Testando cenário: {scenario_name.upper()}")
        print("-" * 40)
        
        try:
            scenario = get_scenario_config(scenario_name)
            print(f"✅ Cenário criado: {scenario.name}")
            print(f"   Descrição: {scenario.description}")
            print(f"   Tipo: {scenario.building_type.value}")
            print(f"   Zonas: {len(scenario.zones)}")
            print(f"   Ocupantes: {scenario.total_occupants}")
            print(f"   Infectados iniciais: {scenario.initial_infected_ratio*100:.1f}%")
            
            # Valida cenário
            issues = validate_scenario(scenario)
            if issues:
                print(f"   ⚠️ Problemas encontrados: {len(issues)}")
                for issue in issues[:3]:  # Mostra apenas os 3 primeiros
                    print(f"     - {issue}")
                if len(issues) > 3:
                    print(f"     ... e mais {len(issues)-3} problemas")
            else:
                print(f"   ✅ Cenário válido!")
            
            # Estatísticas
            stats = calculate_scenario_statistics(scenario)
            print(f"   📐 Área total: {stats['total_area_m2']:.1f} m²")
            print(f"   📦 Volume total: {stats['total_volume_m3']:.1f} m³")
            print(f"   👥 Ocupação: {stats['occupancy_percentage']:.1f}% da capacidade")
            
            # Verifica ACH mínimo recomendado
            for zone_name, zone_stat in stats['zone_statistics'].items():
                if zone_stat['target_ach'] < zone_stat['required_ach'] * 0.8:
                    print(f"   ⚠️ Zona '{zone_name}': ACH alvo ({zone_stat['target_ach']:.1f}) " +
                          f"abaixo do recomendado ({zone_stat['required_ach']:.1f})")
            
        except Exception as e:
            print(f"❌ Erro ao criar cenário '{scenario_name}': {str(e)}")
    
    print("\n" + "=" * 80)
    print("TESTE DE FUNCIONALIDADES ADICIONAIS")
    print("=" * 80)
    
    # Testa ajuste sazonal
    school = create_school_scenario()
    winter_school = adjust_scenario_for_season(school, 'winter')
    print(f"\n❄️  Escola ajustada para inverno:")
    print(f"   Temperatura: {winter_school.temperature_setpoint:.1f}°C (original: {school.temperature_setpoint:.1f}°C)")
    
    # Testa ajuste pandêmico
    pandemic_school = adjust_scenario_for_pandemic(school, 'high')
    print(f"\n🦠 Escola ajustada para pandemia alta:")
    print(f"   Uso de máscaras: {pandemic_school.agent_config.mask_wearing_prob*100:.0f}%")
    print(f"   Ocupantes: {pandemic_school.total_occupants} (original: {school.total_occupants})")
    
    # Testa criação de DataFrame
    try:
        dfs = scenario_to_dataframe(school)
        print(f"\n📊 DataFrames criados: {', '.join(dfs.keys())}")
        print(f"   Zonas: {len(dfs['zones'])} linhas")
        print(f"   Atividades: {len(dfs['agent_activities'])} tipos")
    except ImportError:
        print("\n📊 Pandas não instalado, pulando criação de DataFrames")
    
    print("\n" + "=" * 80)
    print("CONSTANTES DISPONÍVEIS")
    print("=" * 80)
    
    print(f"\n📚 Cenários pré-definidos: {len(CONSTANTS['SCENARIOS'])}")
    print(f"🌡️  Limites de exposição: {len(CONSTANTS['EXPOSURE_LIMITS'])} categorias")
    print(f"🔄 Fatores de conversão: {len(CONSTANTS['CONVERSION_FACTORS'])}")
    print(f"⚛️  Constantes físicas: {len(CONSTANTS['PHYSICAL_CONSTANTS'])}")
    
    # Mostra alguns valores de exemplo
    print(f"\n📈 Exemplos de valores:")
    print(f"   CO₂ excelente: < {CONSTANTS['EXPOSURE_LIMITS']['co2']['excellent']} ppm")
    print(f"   HCHO limite WHO 30min: {CONSTANTS['EXPOSURE_LIMITS']['hcho']['who_30min']} ppb")
    print(f"   Conversão CO₂ ppm→kg/m³: {CONSTANTS['CONVERSION_FACTORS']['co2_ppm_to_kgm3']:.1e}")
    print(f"   Densidade do ar: {CONSTANTS['PHYSICAL_CONSTANTS']['air_density']} kg/m³")
    
    print("\n✅ Configuração final carregada com sucesso!")
    print("=" * 80)