"""
Motor de Física (CFD Simplificado) para Transporte de Aerossóis.

Responsabilidade:
- Resolver a equação de Advecção-Difusão-Reação para contaminantes.
- Gerenciar campos escalares (Vírus, CO2).
- Garantir conservação de massa e estabilidade numérica (CFL).
- Interagir com barreiras físicas (máscara de obstáculos).

Equação Governante:
    ∂C/∂t = D·∇²C - v·∇C - λ·C + S
    
    Onde:
    - ∇²C: Termo de Difusão (Espalhamento turbulento)
    - v·∇C: Termo de Advecção (Transporte pelo vento)
    - λ·C: Termo de Decaimento (Remoção por ventilação + biológico)
    - S: Termo Fonte (Emissão dos agentes)

Dependências:
- numpy: Operações matriciais vetorizadas.
- scipy.ndimage: Convolução para cálculo do Laplaciano.
"""

import logging
import numpy as np
from scipy.ndimage import convolve
from dataclasses import dataclass
from typing import Tuple, List, Optional

from config import PhysicsConfig, VentilationConfig

# Logger setup
logger = logging.getLogger(__name__)

class PhysicsEngine:
    """
    Solver numérico para transporte de massa em grid 2D.
    Gerencia múltiplos campos escalares (Vírus, CO2).
    """

    def __init__(
        self, 
        physics_config: PhysicsConfig, 
        ventilation_config: VentilationConfig,
        obstacle_mask: List[List[int]]
    ):
        """
        Inicializa o grid físico e parâmetros do solver.

        Args:
            physics_config: Configuração geométrica (dimensões, cell_size).
            ventilation_config: Configuração de fluxo (ACH, tipo).
            obstacle_mask: Matriz binária do Environment (1=Parede, 0=Ar).
        """
        self.cfg = physics_config
        self.vent_cfg = ventilation_config
        
        # Dimensões do Grid (Derivadas da máscara para consistência total)
        mask_arr = np.array(obstacle_mask)
        self.width_cells, self.height_cells = mask_arr.shape
        self.cell_size = physics_config.cell_size_m
        self.cell_area = self.cell_size ** 2
        self.cell_volume = self.cell_area * physics_config.ceiling_height_m

        # --- Máscaras ---
        # Environment usa 1=Parede. Física usa 1.0=Fluido, 0.0=Sólido.
        self.fluid_mask = (1.0 - mask_arr).astype(np.float64)

        # --- Campos Escalares (Estados) ---
        # 1. Concentração Viral (Quanta/m³)
        self.virus_grid = np.zeros((self.width_cells, self.height_cells), dtype=np.float64)
        
        # 2. CO2 (ppm) - Inicializa com valor base atmosférico
        self.co2_grid = np.full((self.width_cells, self.height_cells), 420.0, dtype=np.float64)

        # --- Parâmetros Físicos ---
        # Coeficiente de difusão turbulenta indoor (valor empírico típico)
        self.diffusion_coeff = 1e-4  # m²/s
        
        # Velocidade de Advecção (Campo de Vetores U, V)
        # Inicializa campo de fluxo base (Piston Flow: Esquerda -> Direita)
        self.u_vel, self.v_vel = self._initialize_velocity_field()
        
        # Taxa de remoção (λ) = Taxa de Ventilação (ACH -> 1/s)
        # Nota: Decaimento biológico do vírus pode ser somado aqui se necessário
        self.removal_rate = self.vent_cfg.ach / 3600.0

        logger.info(
            f"Physics Engine Iniciado: {self.width_cells}x{self.height_cells} células. "
            f"Volume Célula: {self.cell_volume:.2f}m³. "
            f"Vento Base: {np.mean(self.u_vel):.3f} m/s"
        )

    def step(self, dt: float):
        """
        Executa um passo de integração temporal (Operator Splitting).
        
        Ordem:
        1. Verificar estabilidade (CFL).
        2. Advecção (Transporte).
        3. Difusão (Espalhamento).
        4. Reação/Decaimento (Remoção).
        5. Condições de Contorno (Paredes).
        """
        self._check_cfl_condition(dt)

        # --- Resolver Campo Viral ---
        self.virus_grid = self._solve_transport_step(self.virus_grid, dt, decay=True)
        
        # --- Resolver Campo de CO2 ---
        # CO2 também difunde e advecta, mas o "background" é 420ppm.
        # Simplificação: Resolvemos transporte da diferença (delta) e somamos base depois,
        # ou aplicamos decay tendendo a 420. Aqui aplicamos transporte direto.
        self.co2_grid = self._solve_transport_step(self.co2_grid, dt, decay=True)
        
        # Garante que CO2 não caia abaixo do nível atmosférico (ar fresco entrando)
        self.co2_grid = np.maximum(self.co2_grid, 420.0)

    def add_source(self, cell_x: int, cell_y: int, emission_rate_value: float, dt: float, field_type: str = 'virus'):
        """
        Injeta massa no grid (Termo Fonte 'S').
        
        Args:
            emission_rate_value: 
                - Vírus: quanta/h
                - CO2: L/h (será convertido para massa/volume apropriado)
            field_type: 'virus' ou 'co2'
        """
        # Proteção de limites
        if not (0 <= cell_x < self.width_cells and 0 <= cell_y < self.height_cells):
            return

        # Converter taxa horária para taxa no timestep (valor total emitido no dt)
        amount_emitted = (emission_rate_value / 3600.0) * dt
        
        # Calcular aumento de concentração (Delta C = Massa / Volume)
        delta_concentration = amount_emitted / self.cell_volume

        if field_type == 'virus':
            self.virus_grid[cell_x, cell_y] += delta_concentration
        
        elif field_type == 'co2':
            # Para CO2, emission geralmente é L/h ou g/h. 
            # Se for volume (L), converter para m³ (1 L = 0.001 m³)
            # A concentração de CO2 é PPM (partes por milhão = cm³/m³)
            # 1 ppm = 1e-6 volume_frac. 
            # Simplificação: Tratamos grid como valor absoluto escalar, depois convertemos.
            # Assumindo emission_rate em m³/h puro (ex: respiração humana ~0.02 m³/h CO2 puro)
            
            # Se a entrada for L/h -> m³/h
            rate_m3_s = (emission_rate_value * 0.001) / 3600.0 * dt
            
            # Concentração adicionada (fração volumétrica)
            added_fraction = rate_m3_s / self.cell_volume
            
            # Converter fração para PPM (1e6)
            delta_ppm = added_fraction * 1_000_000
            
            self.co2_grid[cell_x, cell_y] += delta_ppm

    def get_concentration_at(self, x: int, y: int) -> float:
        """Retorna concentração viral na célula (x,y). Retorna 0.0 se for parede."""
        if 0 <= x < self.width_cells and 0 <= y < self.height_cells:
            return self.virus_grid[x, y]
        return 0.0

    # ========================================================================
    # SOLVERS NUMÉRICOS (PRIVADO)
    # ========================================================================

    def _solve_transport_step(self, grid: np.ndarray, dt: float, decay: bool) -> np.ndarray:
        """Aplica os operadores diferenciais sequencialmente."""
        new_grid = grid.copy()

        # 1. Advecção (Upwind)
        new_grid = self._advection_upwind(new_grid, dt)

        # 2. Difusão (Convolução Laplaciana)
        laplacian = self._laplacian(new_grid)
        diffusion_term = self.diffusion_coeff * laplacian * dt
        new_grid += diffusion_term

        # 3. Decaimento / Remoção (Ventilação)
        if decay:
            # C(t+1) = C(t) * (1 - lambda * dt) -> Aproximação Euler
            decay_factor = 1.0 - (self.removal_rate * dt)
            new_grid *= max(0.0, decay_factor) # Evita fator negativo se dt for enorme

        # 4. Condições de Contorno (Obstáculos)
        # Zera concentração onde é parede sólida (absorção total ou barreira impermeável)
        # Para barreira impermeável (Neumann), seria mais complexo. 
        # Aqui simplificamos: Paredes não acumulam vírus, são "sumidouros" ou nulos.
        # Multiplicar pela fluid_mask garante que não há vírus "dentro" da parede.
        new_grid *= self.fluid_mask

        # 5. Segurança Numérica (Não-negatividade)
        np.maximum(new_grid, 0.0, out=new_grid)
        
        return new_grid

    def _laplacian(self, grid: np.ndarray) -> np.ndarray:
        """
        Calcula o Laplaciano discreto (∇²C) usando convolução.
        Kernel de diferenças finitas de 5 pontos (2D).
        """
        # Kernel Laplaciano: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        
        # Convolução
        result = convolve(grid, kernel, mode='constant', cval=0.0)
        
        # Normalização espacial (d²C/dx²) -> Dividir por dx²
        return result / (self.cell_size ** 2)

    def _advection_upwind(self, grid: np.ndarray, dt: float) -> np.ndarray:
        """
        Esquema Upwind de 1ª ordem para Advecção.
        Lida com vento positivo (da esquerda para direita).
        
        Fórmula: C[i] = C[i] - (v * dt / dx) * (C[i] - C[i-1])
        """
        # Courant number local
        courant_x = (self.u_vel * dt) / self.cell_size
        
        # Cálculo vetorizado usando slicing do Numpy
        # Fluxo X: Assumindo u_vel > 0 (Vento para Direita)
        # dC/dx ~ (C[i,j] - C[i-1,j]) / dx
        
        # Criamos array de diferenças (Backward Difference)
        # diff_x[i] = grid[i] - grid[i-1]
        diff_x = np.zeros_like(grid)
        diff_x[1:, :] = grid[1:, :] - grid[:-1, :]
        
        # Aplica advecção
        # grid_new = grid_old - (u * dt / dx) * diff
        advection_term = (self.u_vel * dt / self.cell_size) * diff_x
        
        return grid - advection_term

    def _initialize_velocity_field(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Define o campo de vetores de vento inicial.
        Modelo Simplificado: Fluxo laminar da esquerda para a direita (Piston Flow).
        A velocidade é derivada do ACH (Trocas de Ar por Hora).
        """
        # v_eff (m/s) = (Volume * ACH / 3600) / Área_Transversal
        # Área Transversal = Largura(Y) * Altura(Z) se o fluxo for em X.
        # No nosso grid 2D (X, Y), assumimos fluxo em X.
        # Logo área transversal é room_height_m * room_height_m (Profundidade Y * Pé direito Z).
        
        flow_rate_m3_s = (self.cell_volume * self.width_cells * self.height_cells * self.vent_cfg.ach) / 3600.0
        
        # Área da seção transversal ao fluxo (plano Y-Z)
        cross_section_area = (self.height_cells * self.cell_size) * self.cfg.ceiling_height_m
        
        # Velocidade média
        if cross_section_area > 0:
            avg_velocity = flow_rate_m3_s / cross_section_area
        else:
            avg_velocity = 0.0

        # Cria grids (u=X, v=Y)
        u_grid = np.full((self.width_cells, self.height_cells), avg_velocity, dtype=np.float64)
        v_grid = np.zeros((self.width_cells, self.height_cells), dtype=np.float64)
        
        # Zera velocidade dentro de obstáculos
        u_grid *= self.fluid_mask
        v_grid *= self.fluid_mask
        
        return u_grid, v_grid

    def _check_cfl_condition(self, dt: float):
        """Monitora estabilidade numérica (Courant-Friedrichs-Lewy)."""
        max_u = np.max(self.u_vel)
        if max_u > 0:
            cfl = (max_u * dt) / self.cell_size
            if cfl > 1.0:
                logger.warning(
                    f"Instabilidade CFL detectada! Courant={cfl:.2f} (>1.0). "
                    f"Reduza o dt ou aumente o cell_size. Simulação pode divergir."
                )

    # ========================================================================
    # API DE DADOS (PARA DASHBOARD)
    # ========================================================================
    
    def get_virus_snapshot(self) -> np.ndarray:
        """Retorna cópia segura do grid de vírus (para visualização)."""
        return self.virus_grid.copy()
        
    def get_co2_snapshot(self) -> np.ndarray:
        """Retorna cópia segura do grid de CO2."""
        return self.co2_grid.copy()