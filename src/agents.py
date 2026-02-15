"""
Módulo de Agentes Epidemiológicos (HumanAgent).

Implementa a lógica comportamental e fisiológica dos ocupantes,
incluindo modelos de infecção (Wells-Riley), evolução de carga viral
e máquinas de estado para movimentação realista.

Dependências:
- mesa: Framework de ABM.
- src.config: Constantes científicas e definições de tipos.
"""

import logging
import math
import random
from typing import Tuple, Optional, Dict, Any

from mesa import Agent
import numpy as np

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
        initial_state: AgentState = AgentState.SUSCEPTIBLE
    ):
        """
        Inicializa o agente humano.

        Args:
            unique_id: ID do agente.
            model: Referência ao modelo Mesa (IAQModel).
            pos: Posição inicial.
            agent_config: Configuração de agentes do cenário (AgentsConfig).
            initial_state: Estado inicial (padrão SUSCEPTIBLE).
        """
        super().__init__(unique_id, model)
        self.pos = pos
        
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

        # --- Comportamento/Movimento ---
        # Define um alvo de movimento (ex: mesa de trabalho, equipamento)
        self.target_pos: Optional[Tuple[int, int]] = None
        self.movement_state = "WANDERING" # WANDERING, WORKING, RESTING
        self.ticks_in_state = 0

    def step(self):
        """
        Executa um passo de simulação do agente.
        Ordem: 
        1. Atualizar Fisiologia (Carga Viral/Recuperação).
        2. Movimentação (Comportamento).
        """
        # Atualiza dinâmica viral se infectado
        if self.state == AgentState.INFECTED:
            self._update_viral_dynamics()

        # Executa movimentação complexa
        self._execute_movement_logic()

    # ========================================================================
    # LÓGICA EPIDEMIOLÓGICA (PÚBLICA)
    # ========================================================================

    def calculate_emission_quanta_per_s(self) -> float:
        """
        Calcula a emissão instantânea de quanta viral por segundo.
        
        Fórmula: (EmissãoBase / 3600) * CargaViral(t) * (1 - EficiênciaMascara)
        
        Returns:
            float: Quanta emitidos por segundo neste passo.
        """
        if self.state != AgentState.INFECTED:
            return 0.0
        
        # Conversão hora -> segundo
        emission_per_second = self.emission_rate_base / 3600.0
        
        # Fator de redução da máscara (na exalação)
        # Assumindo eficiência simétrica para simplificação, ou conforme literatura
        mask_factor = 1.0 - self.mask_efficiency
        
        return emission_per_second * self.viral_load * mask_factor

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
        
        # Proteção da máscara na inalação
        # Nota: O prompt pediu (1 - mask_efficiency/2), mas para ser rigoroso
        # com a literatura, máscaras N95 filtram 95% na entrada. 
        # Seguirei a instrução do prompt para compliance com o requisito.
        protection_factor = 1.0 - (self.mask_efficiency / 2.0)
        
        # Dose = C * Q * t * Proteção
        dose_step = (
            concentration_quanta_m3 * self.respiration_rate * dt_hours * protection_factor
        )
        
        self.accumulated_dose += dose_step
        self._attempt_infection()

    # ========================================================================
    # LÓGICA INTERNA (PRIVADA)
    # ========================================================================

    def _attempt_infection(self):
        """
        Avalia a probabilidade de infecção baseada na dose acumulada (Wells-Riley).
        """
        if self.accumulated_dose <= 0:
            return

        # Modelo Exponencial de Dose-Resposta
        # P = 1 - exp(-Dose / ID50)
        infection_prob = 1.0 - math.exp(-self.accumulated_dose / DiseaseParams.ID50)
        
        # Sorteio de Bernoulli
        if random.random() < infection_prob:
            self._become_infected()

    def _become_infected(self):
        """Transiciona o agente para o estado INFECTED."""
        self.state = AgentState.INFECTED
        self.infection_time = self.model.time  # Tempo atual da simulação em segundos
        self.viral_load = 0.1 # Inicia baixo e sobe
        
        logger.info(f"Agente {self.unique_id} infectado na posição {self.pos}. Dose: {self.accumulated_dose:.4f}")

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
        """Mapeia ActivityLevel para constantes de EmissionRates (quanta/h)."""
        mapping = {
            ActivityLevel.SEDENTARY: EmissionRates.SEATED_QUIET,
            ActivityLevel.LIGHT: EmissionRates.TALKING, # Assumindo leve conversação/movimento
            ActivityLevel.MODERATE: EmissionRates.EXERCISE_LIGHT,
            ActivityLevel.HEAVY: EmissionRates.EXERCISE_HEAVY
        }
        return mapping.get(activity, EmissionRates.SEATED_QUIET)

    def _get_respiration_rate(self, activity: ActivityLevel) -> float:
        """Mapeia ActivityLevel para RespirationRates (m³/h)."""
        mapping = {
            ActivityLevel.SEDENTARY: RespirationRates.SEDENTARY,
            ActivityLevel.LIGHT: RespirationRates.LIGHT,
            ActivityLevel.MODERATE: RespirationRates.MODERATE,
            ActivityLevel.HEAVY: RespirationRates.HEAVY
        }
        return mapping.get(activity, RespirationRates.SEDENTARY)

    # ========================================================================
    # LÓGICA DE MOVIMENTAÇÃO (Complexa/Human-like)
    # ========================================================================

    def _execute_movement_logic(self):
        """
        Implementa uma máquina de estados finitos para movimentação.
        Simula: Trabalho -> Pausa -> Socialização -> Trabalho.
        """
        # Se o modelo não tiver grid definido ainda (teste unitário), pula
        if not hasattr(self.model, "grid") or not self.model.grid:
            return

        self.ticks_in_state += 1
        
        # 1. Definição de Destinos (Se não tiver um)
        if self.target_pos is None:
            self._pick_new_target()

        # 2. Movimento passo a passo (Pathfinding simples)
        if self.target_pos:
            next_step = self._get_next_step_towards(self.target_pos)
            
            # Verifica se a célula está livre (sem outros agentes)
            # O Grid do Mesa lida com isso se for 'SingleGrid', mas assumimos 'MultiGrid'
            # Para realismo, evitamos sobreposição
            if self._is_cell_available(next_step):
                self.model.grid.move_agent(self, next_step)
                
            # Chegou no destino?
            if self.pos == self.target_pos:
                self.target_pos = None # Aguarda no local (Working/Exercising)
                
    def _pick_new_target(self):
        """O agente pede ao ambiente um lugar para ir."""
        # Ex: "Quero uma mesa"
        target = self.model.environment.get_random_poi("furniture")
        
        if target:
            self.target_pos = target
        else:
            self._random_move()

    def _get_next_step_towards(self, target: Tuple[int, int]) -> Tuple[int, int]:
        """Calcula a próxima célula na direção do alvo (Heurística Chebyshev)."""
        x, y = self.pos
        tx, ty = target
        
        dx = np.sign(tx - x)
        dy = np.sign(ty - y)
        
        return (x + dx, y + dy)

    def _is_cell_available(self, pos: Tuple[int, int]) -> bool:
        """
        Valida se uma célula está livre para ocupação.
        """
        # 1. Validação Estática (Environment Facade)
        # Verifica limites do mapa e obstáculos fixos (paredes/colunas)
        if not self.model.environment.is_valid_move(pos):
            return False

        # 2. Validação Dinâmica (Mesa Grid)
        # Verifica se já existe outro agente ocupando a célula
        if not self.model.grid.is_cell_empty(pos):
             return False
             
        return True

    def _random_move(self):
        """
        Realiza um movimento aleatório local (fallback).
        
        Lógica:
        1. Obtém vizinhos imediatos (Moore neighborhood).
        2. Filtra posições usando a fachada de ambiente (Environment) e o Grid.
        3. Escolhe aleatoriamente e move.
        """
        # Obtém coordenadas vizinhas (8 direções)
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, 
            moore=True, 
            include_center=False
        )

        # Filtra passos válidos usando o método auxiliar que consulta o Environment
        valid_steps = [
            pos for pos in possible_steps 
            if self._is_cell_available(pos)
        ]

        # Se houver pelo menos um passo válido, move
        if valid_steps:
            new_pos = random.choice(valid_steps)
            self.model.grid.move_agent(self, new_pos)