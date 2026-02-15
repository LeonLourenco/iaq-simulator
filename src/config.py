"""
Módulo de Configuração e Definição de Tipos para Simulador IAQ.

ARQUITETURA:
Este módulo atua como o 'Schema Definition' do projeto.
Ele define as constantes científicas e as estruturas de dados (Dataclasses).
Implementa o padrão 'Data Transfer Object' (DTO) para converter JSONs brutos
em objetos Python tipados e validados cientificamente.

Responsabilidade:
- Definir Dataclasses para tipagem forte.
- Centralizar constantes científicas (Buonanno, ASHRAE).
- Serialização e Deserialização (JSON <-> Python Object).
- Factories para criação dinâmica de cenários.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

# ============================================================================
# 1. CONSTANTES FÍSICAS GERAIS
# ============================================================================

AIR_DENSITY = 1.225  # kg/m³ (ISA Standard)
DEFAULT_DIFFUSION_COEFF = 1e-6  # m²/s

# ============================================================================
# 2. ENUMS (Domínio Discreto)
# ============================================================================

class AgentState(str, Enum):
    """Estados epidemiológicos SIR."""
    SUSCEPTIBLE = "SUSCEPTIBLE"
    INFECTED = "INFECTED"
    RECOVERED = "RECOVERED"

class ActivityLevel(str, Enum):
    """Níveis de atividade metabólica e vocal."""
    SEDENTARY = "sedentary"      # Repouso/Escritório
    LIGHT = "light"              # Caminhada/Aula
    MODERATE = "moderate"        # Exercício leve
    HEAVY = "heavy"              # Exercício intenso

class VentilationType(str, Enum):
    """Tipos de ventilação."""
    NATURAL = "natural"
    MECHANICAL = "mechanical"
    HYBRID = "hybrid"

# ============================================================================
# 3. CONSTANTES CIENTÍFICAS (Literature-based)
# ============================================================================

@dataclass(frozen=True)
class DiseaseParams:
    """
    Parâmetros do SARS-CoV-2 (Buonanno et al. 2020, Keeling & Rohani).
    """
    ID50: float = 50.0            # Dose infectante média (quanta)
    INCUBATION_DAYS: float = 5.2  # Período incubação
    INFECTIOUS_DAYS: float = 12.0 # Período infeccioso
    R0_BASE: float = 2.5          # R0 ancestral

@dataclass(frozen=True)
class EmissionRates:
    """
    Emissão viral (quanta/h) por atividade (Buonanno et al. 2020).
    Considera carga viral alta (10^8 copies/mL).
    """
    SEATED_QUIET: float = 10.0
    TALKING: float = 50.0
    SINGING: float = 200.0
    EXERCISE_LIGHT: float = 30.0
    EXERCISE_HEAVY: float = 100.0
    COUGHING: float = 500.0

@dataclass(frozen=True)
class RespirationRates:
    """Taxas respiratórias (m³/h) (ASHRAE Fundamentals)."""
    SEDENTARY: float = 0.54
    LIGHT: float = 0.96
    MODERATE: float = 1.38
    HEAVY: float = 2.10

# ============================================================================
# 4. DATA STRUCTURES (O Schema do Cenário)
# ============================================================================

@dataclass
class PhysicsConfig:
    """Configuração do motor físico (CFD simplificado)."""
    room_width_m: float
    room_height_m: float  # Profundidade (Y)
    ceiling_height_m: float
    cell_size_m: float = 0.5
    total_volume_m3: float = field(init=False)

    def __post_init__(self):
        self.total_volume_m3 = self.room_width_m * self.room_height_m * self.ceiling_height_m

@dataclass
class VentilationConfig:
    """Configuração de HVAC."""
    ach: float  # Trocas por hora
    type: VentilationType = VentilationType.MECHANICAL
    efficiency: float = 1.0  # 1.0 = mistura perfeita
    filtration_merv: int = 8  # Eficiência filtro

@dataclass
class AgentsConfig:
    """Configuração da população."""
    total_occupants: int
    initial_infected: int
    activity_level: ActivityLevel
    mask_compliance: float  # 0.0 a 1.0
    mask_efficiency: float  # 0.0 a 1.0

@dataclass
class ObstacleConfig:
    """Definição de obstáculos físicos."""
    id: str
    x: float
    y: float
    width: float
    height: float
    type: str = "furniture" # wall, furniture

@dataclass
class ScenarioConfig:
    """
    Objeto Raiz de Configuração.
    Representa o conteúdo completo de um arquivo .json de cenário.
    """
    name: str
    description: str
    duration_hours: float
    physics: PhysicsConfig
    ventilation: VentilationConfig
    agents: AgentsConfig
    obstacles: List[ObstacleConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScenarioConfig':
        """
        Factory method que hidrata um dicionário (do JSON) em objetos tipados.
        Realiza conversão de Strings para Enums e Nested Objects.
        """
        try:
            # 1. Hidrata Physics
            phy_data = data['physics']
            physics = PhysicsConfig(
                room_width_m=phy_data['width'],
                room_height_m=phy_data['height'],
                ceiling_height_m=phy_data['ceiling'],
                cell_size_m=phy_data.get('cell_size', 0.5)
            )

            # 2. Hidrata Ventilation
            vent_data = data['ventilation']
            ventilation = VentilationConfig(
                ach=vent_data['ach'],
                type=VentilationType(vent_data.get('type', 'mechanical')),
                efficiency=vent_data.get('efficiency', 1.0),
                filtration_merv=vent_data.get('merv', 8)
            )

            # 3. Hidrata Agents
            ag_data = data['agents']
            agents = AgentsConfig(
                total_occupants=ag_data['total'],
                initial_infected=ag_data['infected'],
                activity_level=ActivityLevel(ag_data['activity']),
                mask_compliance=ag_data.get('mask_compliance', 0.0),
                mask_efficiency=ag_data.get('mask_efficiency', 0.0)
            )

            # 4. Hidrata Obstacles
            obstacles = []
            for obs in data.get('obstacles', []):
                obstacles.append(ObstacleConfig(**obs))

            # Retorna objeto completo
            return cls(
                name=data['name'],
                description=data.get('description', ''),
                duration_hours=data['duration_hours'],
                physics=physics,
                ventilation=ventilation,
                agents=agents,
                obstacles=obstacles
            )
        except KeyError as e:
            raise ValueError(f"JSON de cenário inválido. Campo faltando: {e}")
        except ValueError as e:
            raise ValueError(f"Valor inválido no JSON: {e}")

    @classmethod
    def load_from_json(cls, filepath: Union[str, Path]) -> 'ScenarioConfig':
        """Carrega e valida um arquivo JSON do disco."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo de cenário não encontrado: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls.from_dict(data)

    def save_to_json(self, filepath: Union[str, Path]):
        """Salva a configuração atual em JSON (útil para criar templates)."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=4, default=str)

# ============================================================================
# 5. PRESETS ESTÁTICOS (Geradores de Default)
# ============================================================================

def get_default_school_config() -> ScenarioConfig:
    """Gera a configuração padrão de escola em memória."""
    return ScenarioConfig(
        name="Escola Padrão UFRPE",
        description="Sala de aula típica com ventilação natural assistida",
        duration_hours=4.0,
        physics=PhysicsConfig(10.0, 8.0, 3.0),
        ventilation=VentilationConfig(ach=4.0, type=VentilationType.HYBRID),
        agents=AgentsConfig(
            total_occupants=30,
            initial_infected=1,
            activity_level=ActivityLevel.LIGHT,
            mask_compliance=0.5,
            mask_efficiency=0.4
        ),
        obstacles=[
            ObstacleConfig("teacher_desk", 4.0, 1.0, 1.6, 0.8, "furniture"),
            ObstacleConfig("row1_left", 1.0, 3.0, 3.0, 0.6, "furniture"),
            ObstacleConfig("row1_right", 6.0, 3.0, 3.0, 0.6, "furniture")
        ]
    )

def get_default_office_config() -> ScenarioConfig:
    return ScenarioConfig(
        name="Escritório Open Space",
        description="Ambiente corporativo climatizado",
        duration_hours=8.0,
        physics=PhysicsConfig(15.0, 10.0, 2.8),
        ventilation=VentilationConfig(ach=3.0, type=VentilationType.MECHANICAL),
        agents=AgentsConfig(
            total_occupants=20,
            initial_infected=1,
            activity_level=ActivityLevel.SEDENTARY,
            mask_compliance=0.2,
            mask_efficiency=0.5
        ),
        obstacles=[
             ObstacleConfig("island_1", 2.0, 2.0, 4.0, 3.0, "furniture")
        ]
    )

def get_default_gym_config() -> ScenarioConfig:
    return ScenarioConfig(
        name="Academia Crossfit",
        description="Alta emissão viral e respiração intensa",
        duration_hours=2.0,
        physics=PhysicsConfig(20.0, 12.0, 4.5),
        ventilation=VentilationConfig(ach=6.0, type=VentilationType.MECHANICAL),
        agents=AgentsConfig(
            total_occupants=15,
            initial_infected=1,
            activity_level=ActivityLevel.HEAVY,
            mask_compliance=0.0,
            mask_efficiency=0.0
        ),
        obstacles=[
            ObstacleConfig("rig", 8.0, 4.0, 4.0, 4.0, "equipment")
        ]
    )

# ============================================================================
# 6. DYNAMIC FACTORIES (Para compatibilidade com Testes e CLI)
# ============================================================================

def create_school_scenario(occupants: int = 30, infected: int = 1, ach: float = 4.0) -> ScenarioConfig:
    """Cria cenário de Escola com parâmetros customizáveis."""
    config = get_default_school_config()
    config.agents.total_occupants = occupants
    config.agents.initial_infected = infected
    config.ventilation.ach = ach
    return config

def create_office_scenario(occupants: int = 20, infected: int = 1, ach: float = 3.0) -> ScenarioConfig:
    """Cria cenário de Escritório com parâmetros customizáveis."""
    config = get_default_office_config()
    config.agents.total_occupants = occupants
    config.agents.initial_infected = infected
    config.ventilation.ach = ach
    return config

def create_gym_scenario(occupants: int = 15, infected: int = 1, ach: float = 6.0) -> ScenarioConfig:
    """Cria cenário de Academia com parâmetros customizáveis."""
    config = get_default_gym_config()
    config.agents.total_occupants = occupants
    config.agents.initial_infected = infected
    config.ventilation.ach = ach
    return config