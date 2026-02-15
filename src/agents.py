"""
Módulo de Agentes Epidemiológicos (HumanAgent).

Implementa a lógica comportamental e fisiológica dos ocupantes,
incluindo modelos de infecção (Wells-Riley), evolução de carga viral
e máquinas de estado para movimentação realista.

Dependências:
- mesa: Framework de ABM.
- src.config: Constantes científicas e definições de tipos.
- src.behaviors: Padrão Strategy para decisão de movimento.
"""

import logging
import math
import random
from typing import Tuple, Optional, Dict, Any

from mesa import Agent
import numpy as np

from behaviors import BehaviorFactory
from config import (
    AgentState,
    ActivityLevel,
    DiseaseParams,
    EmissionRates,
    RespirationRates,
    AgentsConfig
)

# Configuração de Logger
logger = logging.getLogger(__name__)

class HumanAgent(Agent):
    """
    Agente humano com fisiologia respiratória e comportamento social.

    Atributos:
        unique_id (int): Identificador único.
        pos (Tuple[int, int]): Posição (x, y) no grid.
        state (AgentState): Estado epidemiológico (S, I, R).
        viral_load (float): Carga viral normalizada (0.0 a 1.0).
        accumulated_dose (float): Dose viral acumulada em quanta.
    """

    def __init__(
        self,
        unique_id: int,
        model: Any,
        pos: Tuple[int, int],
        agent_config: AgentsConfig,
        initial_state: AgentState = AgentState.SUSCEPTIBLE,
        profile_type: str = "student_focused"  # Valor default para compatibilidade
    ):
        """
        Inicializa o agente humano.

        Args:
            unique_id: ID do agente.
            model: Referência ao modelo Mesa (IAQModel).
            pos: Posição inicial.
            agent_config: Configuração de agentes do cenário (AgentsConfig).
            initial_state: Estado inicial (padrão SUSCEPTIBLE).
            profile_type: Identificador do perfil comportamental (Strategy).
        """
        super().__init__(unique_id, model)
        self.pos = pos
        self.config = agent_config # Guarda config para acesso futuro das estratégias
        
        # --- Propriedades Epidemiológicas ---
        self.state = initial_state
        self.infection_time: Optional[float] = 0.0 if initial_state == AgentState.INFECTED else None
        self.recovery_time: Optional[float] = None
        self.accumulated_dose: float = 0.0
        self.viral_load: float = 0.0
        
        # Se começar infectado, inicializa carga viral no início da curva
        if self.state == AgentState.INFECTED:
            self.viral_load = 0.1  # Início da infecção
            
        # --- Propriedades Fisiológicas ---
        self.activity_level = agent_config.activity_level
        self.emission_rate_base = self._get_emission_rate_base(self.activity_level)
        self.respiration_rate = self._get_respiration_rate(self.activity_level)
        
        # --- Equipamento de Proteção (EPI) ---
        # Determina uso de máscara baseado no compliance do cenário (Bernoulli trial)
        self.wears_mask = random.random() < agent_config.mask_compliance
        self.mask_efficiency = agent_config.mask_efficiency if self.wears_mask else 0.0

        # --- Comportamento/Movimento (Strategy Pattern) ---
        # Injeta a lógica de decisão baseada no perfil (Arquétipo)
        # O agente delega a decisão de "onde ir" para esta classe
        self.behavior = BehaviorFactory.create(profile_type, self)

        # Variáveis de estado de movimento (usadas pelas estratégias)
        self.target_pos: Optional[Tuple[int, int]] = None
        self.movement_state = "WORKING" 
        self.ticks_in_state = 0

    def step(self):
        """
        Executa um passo de simulação do agente.
        Ordem: 
        1. Atualizar Fisiologia (Carga Viral/Recuperação).
        2. Movimentação (Delegada ao Comportamento).
        """
        # Atualiza dinâmica viral se infectado
        if self.state == AgentState.INFECTED:
            self._update_viral_dynamics()

        # Executa movimentação baseada na estratégia injetada
        # 1. Pede ao cérebro (Behavior) para onde ir
        new_pos = self.behavior.decide_movement()
        
        # 2. Se o cérebro decidiu mover, tenta executar
        if new_pos and self._is_cell_available(new_pos):
            self.model.grid.move_agent(self, new_pos)

    # ========================================================================
    # LÓGICA EPIDEMIOLÓGICA (PÚBLICA)
    # ========================================================================

    def calculate_emission_quanta_per_s(self) -> float:
        """
        Calcula a emissão instantânea de quanta viral por segundo.
        
        Fórmula: (EmissãoBase / 3600) * CargaViral(t) * (1 - EficiênciaMascara) * MultiplicadorComportamental
        
        Returns:
            float: Quanta emitidos por segundo neste passo.
        """
        if self.state != AgentState.INFECTED:
            return 0.0
        
        # Conversão hora -> segundo
        emission_per_second = self.emission_rate_base / 3600.0
        
        # Obtém multiplicador dinâmico do comportamento (ex: falando alto = 5x)
        behavior_multiplier = self.behavior.get_emission_multiplier()
        
        # Fator de redução da máscara (na exalação)
        mask_factor = 1.0 - self.mask_efficiency
        
        return emission_per_second * self.viral_load * mask_factor * behavior_multiplier

    def inhale(self, concentration_quanta_m3: float, dt_seconds: float):
        """
        Processa a inalação de ar contaminado e acumula dose viral.

        Args:
            concentration_quanta_m3: Concentração local de vírus (quanta/m³).
            dt_seconds: Passo de tempo da simulação física em segundos.
        """
        if self.state != AgentState.SUSCEPTIBLE:
            return

        # Conversão do tempo para horas (taxas respiratórias são m³/h)
        dt_hours = dt_seconds / 3600.0
        
        # Proteção da máscara na inalação (50% da eficiência nominal)
        protection_factor = 1.0 - (self.mask_efficiency / 2.0)
        
        # Dose = C * Q * t * Proteção
        dose_step = (
            concentration_quanta_m3 * self.respiration_rate * dt_hours * protection_factor
        )
        
        self.accumulated_dose += dose_step
        self._attempt_infection()

    # ========================================================================
    # SENSORES E ATUADORES (USADOS PELOS BEHAVIORS)
    # ========================================================================

    def _is_cell_available(self, pos: Tuple[int, int]) -> bool:
        """Valida se uma célula está livre (sem parede, sem gente)."""
        # 1. Validação Estática (Environment Facade)
        if not self.model.environment.is_valid_move(pos):
            return False

        # 2. Validação Dinâmica (Mesa Grid)
        if not self.model.grid.is_cell_empty(pos):
             return False
             
        return True

    def _random_move_in_radius(self, radius: int) -> Optional[Tuple[int, int]]:
        """Tenta encontrar um destino aleatório válido dentro de um raio R."""
        # Tenta 5 vezes encontrar um lugar válido
        for _ in range(5):
            dx = random.randint(-radius, radius)
            dy = random.randint(-radius, radius)
            target = (self.pos[0] + dx, self.pos[1] + dy)
            
            if self._is_cell_available(target):
                return target
        return None

    def _move_towards_density(self) -> Optional[Tuple[int, int]]:
        """Retorna a célula vizinha que aproxima o agente da maior aglomeração."""
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        best_pos = None
        max_density = -1
        
        for pos in neighbors:
            if not self._is_cell_available(pos):
                continue
                
            # Conta vizinhos dessa célula candidata (look-ahead)
            count = len(self.model.grid.get_neighbors(pos, moore=True, include_center=False, radius=1))
            if count > max_density:
                max_density = count
                best_pos = pos
                
        return best_pos

    def _move_away_from_density(self) -> Optional[Tuple[int, int]]:
        """Retorna a célula vizinha que afasta o agente de aglomerações."""
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        best_pos = None
        min_density = 999
        
        for pos in neighbors:
            if not self._is_cell_available(pos):
                continue
                
            count = len(self.model.grid.get_neighbors(pos, moore=True, include_center=False, radius=1))
            if count < min_density:
                min_density = count
                best_pos = pos
                
        return best_pos
        
    def _get_next_step_towards(self, target: Tuple[int, int]) -> Tuple[int, int]:
        """Calcula próximo passo (Heurística Chebyshev)."""
        x, y = self.pos
        tx, ty = target
        dx = np.sign(tx - x)
        dy = np.sign(ty - y)
        return (x + dx, y + dy)
    
    # ========================================================================
    # LÓGICA INTERNA (PRIVADA)
    # ========================================================================

    def _attempt_infection(self):
        """Avalia probabilidade de infecção (Wells-Riley)."""
        if self.accumulated_dose <= 0: return

        # Modelo Exponencial: P = 1 - exp(-Dose / ID50)
        infection_prob = 1.0 - math.exp(-self.accumulated_dose / DiseaseParams.ID50)
        
        if random.random() < infection_prob:
            self._become_infected()

    def _become_infected(self):
        """Transiciona para estado infectado e loga o evento."""
        self.state = AgentState.INFECTED
        self.infection_time = self.model.time
        self.viral_load = 0.1
        
        logger.info(f"Agente {self.unique_id} infectado na posição {self.pos}. Dose: {self.accumulated_dose:.4f}")
        
        # Log centralizado no modelo para Rastreamento de Contatos
        if hasattr(self.model, "log_infection"):
            self.model.log_infection(self)

    def _update_viral_dynamics(self):
        """
        Atualiza a curva de carga viral e verifica recuperação.
        Baseado em dias desde a infecção.
        """
        if self.infection_time is None:
            return

        # Tempo decorrido em dias
        seconds_since_infection = self.model.time - self.infection_time
        days_since_infection = seconds_since_infection / (24 * 3600.0)
        
        # Parâmetros da curva
        peak_day = 4.0
        end_day = DiseaseParams.INFECTIOUS_DAYS # 12.0
        
        if days_since_infection <= peak_day:
            # Fase Ascendente (0 a 1.0)
            self.viral_load = days_since_infection / peak_day
        elif days_since_infection < end_day:
            # Fase Descendente (1.0 a 0)
            # Normaliza o tempo restante entre pico e fim
            remaining_duration = end_day - peak_day
            elapsed_since_peak = days_since_infection - peak_day
            self.viral_load = 1.0 - (elapsed_since_peak / remaining_duration)
        else:
            # Recuperação
            self.viral_load = 0.0
            self.state = AgentState.RECOVERED
            self.recovery_time = self.model.time
            logger.info(f"Agente {self.unique_id} recuperado.")

    def _get_emission_rate_base(self, activity: ActivityLevel) -> float:
        """Mapeia ActivityLevel para constantes."""
        mapping = {
            ActivityLevel.SEDENTARY: EmissionRates.SEATED_QUIET,
            ActivityLevel.LIGHT: EmissionRates.TALKING,
            ActivityLevel.MODERATE: EmissionRates.EXERCISE_LIGHT,
            ActivityLevel.HEAVY: EmissionRates.EXERCISE_HEAVY
        }
        return mapping.get(activity, EmissionRates.SEATED_QUIET)

    def _get_respiration_rate(self, activity: ActivityLevel) -> float:
        """Mapeia ActivityLevel para constantes."""
        mapping = {
            ActivityLevel.SEDENTARY: RespirationRates.SEDENTARY,
            ActivityLevel.LIGHT: RespirationRates.LIGHT,
            ActivityLevel.MODERATE: RespirationRates.MODERATE,
            ActivityLevel.HEAVY: RespirationRates.HEAVY
        }
        return mapping.get(activity, RespirationRates.SEDENTARY)
    