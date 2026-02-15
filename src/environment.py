"""
Módulo de Gerenciamento do Ambiente (Environment Facade).

Responsabilidade:
- Abstrair a complexidade do Grid e da Física para os Agentes.
- Gerenciar Pontos de Interesse (POIs) como Mesas, Equipamentos, etc.
- Converter coordenadas (Metros <-> Células).
- Fornecer API de consulta espacial (ex: "estou numa parede?").

Padrão de Projeto: Facade / Service Layer.
"""

import random
import math
import logging
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass

from config import ScenarioConfig, ObstacleConfig, PhysicsConfig

logger = logging.getLogger(__name__)

@dataclass
class PointOfInterest:
    """Representa um local relevante para o agente (ex: uma mesa de trabalho)."""
    id: str
    type: str  # 'furniture', 'equipment', 'zone'
    cell_pos: Tuple[int, int]
    real_pos: Tuple[float, float]

class Environment:
    """
    Fachada que gerencia a interação espacial e semântica do ambiente.
    Os agentes devem consultar esta classe, não a física bruta.
    """

    def __init__(self, config: ScenarioConfig):
        """
        Inicializa o ambiente com base na configuração do cenário.
        
        Args:
            config: Objeto ScenarioConfig carregado do JSON.
        """
        self.physics_conf = config.physics
        self.cell_size = config.physics.cell_size_m
        self.width_cells = int(config.physics.room_width_m / self.cell_size)
        self.height_cells = int(config.physics.room_height_m / self.cell_size)
        
        # Mapa de obstáculos lógicos (para navegação rápida)
        # 0 = Livre, 1 = Bloqueado
        self._nav_grid = [[0 for _ in range(self.height_cells)] for _ in range(self.width_cells)]
        
        # Lista de Pontos de Interesse (POIs)
        self.pois: Dict[str, List[PointOfInterest]] = {
            "furniture": [],
            "equipment": [],
            "wall": []
        }
        
        self._build_environment(config.obstacles)
        logger.info(f"Ambiente inicializado: {self.width_cells}x{self.height_cells} células.")

    def _build_environment(self, obstacles: List[ObstacleConfig]):
        """Processa os obstáculos para criar o mapa de navegação e POIs."""
        for obs in obstacles:
            # Converter dimensões reais para grid (Bounding Box)
            start_x = int(obs.x / self.cell_size)
            start_y = int(obs.y / self.cell_size)
            end_x = int((obs.x + obs.width) / self.cell_size)
            end_y = int((obs.y + obs.height) / self.cell_size)

            # Clamp para garantir que está dentro do grid
            start_x = max(0, start_x)
            start_y = max(0, start_y)
            end_x = min(self.width_cells, end_x)
            end_y = min(self.height_cells, end_y)

            # Centro do obstáculo (para ser o ponto de destino)
            center_x = (start_x + end_x) // 2
            center_y = (start_y + end_y) // 2
            
            # Registrar POI
            poi = PointOfInterest(
                id=obs.id,
                type=obs.type,
                cell_pos=(center_x, center_y),
                real_pos=(obs.x + obs.width/2, obs.y + obs.height/2)
            )
            
            if obs.type not in self.pois:
                self.pois[obs.type] = []
            self.pois[obs.type].append(poi)

            # Marcar no grid de navegação (se for parede ou móvel intransponível)
            # Aqui assumimos que agentes podem "entrar" em móveis para usá-los, 
            # mas não em paredes.
            is_blocking = (obs.type == "wall") or (obs.type == "column")
            
            if is_blocking:
                for x in range(start_x, end_x):
                    for y in range(start_y, end_y):
                        self._nav_grid[x][y] = 1

    # ========================================================================
    # API PÚBLICA PARA AGENTES (Navigation Services)
    # ========================================================================

    def get_random_poi(self, poi_type: str = "furniture") -> Optional[Tuple[int, int]]:
        """
        Retorna as coordenadas de célula de um Ponto de Interesse aleatório.
        Útil para agentes escolherem uma mesa para trabalhar.
        """
        targets = self.pois.get(poi_type, [])
        if not targets:
            return None
        return random.choice(targets).cell_pos

    def get_nearest_poi(self, current_pos: Tuple[int, int], poi_type: str) -> Optional[Tuple[int, int]]:
        """Encontra o POI mais próximo (distância euclidiana simples)."""
        targets = self.pois.get(poi_type, [])
        if not targets:
            return None
            
        nearest = min(targets, key=lambda p: math.dist(current_pos, p.cell_pos))
        return nearest.cell_pos

    def is_valid_move(self, pos: Tuple[int, int]) -> bool:
        """
        Valida se uma posição é navegável (dentro dos limites e sem paredes).
        O agente NÃO precisa saber o tamanho do grid, só chama isso.
        """
        x, y = pos
        # Checa limites
        if not (0 <= x < self.width_cells and 0 <= y < self.height_cells):
            return False
        
        # Checa colisão estática (paredes)
        if self._nav_grid[x][y] == 1:
            return False
            
        return True

    def meters_to_grid(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """Converte coordenadas reais (m) para grid."""
        return (int(x_m / self.cell_size), int(y_m / self.cell_size))

    def grid_to_meters(self, x_cell: int, y_cell: int) -> Tuple[float, float]:
        """Retorna o centro da célula em metros."""
        return (
            (x_cell * self.cell_size) + (self.cell_size / 2),
            (y_cell * self.cell_size) + (self.cell_size / 2)
        )

    # ========================================================================
    # API PÚBLICA PARA MODELO/FÍSICA
    # ========================================================================
    
    def get_obstacle_mask(self) -> List[List[int]]:
        """Retorna a máscara binária de obstáculos para o PhysicsEngine."""
        # Retorna uma cópia para garantir encapsulamento
        return [row[:] for row in self._nav_grid]