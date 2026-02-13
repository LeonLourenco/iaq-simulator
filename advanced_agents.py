"""
SISTEMA DE AGENTES
=================================================
Agentes humanos com fisiologia realista, comportamento adaptativo,
capacidade de aprendizado, física de colisão com obstáculos e
rastreabilidade completa de exposição.

MELHORIAS DESTA VERSÃO:
- Prontuário médico detalhado (exposure_history com snapshots completos)
- Física de colisão AABB (Axis-Aligned Bounding Box) com obstáculos
- Máquina de estados inteligente (procura mesas para sentar, etc.)
- Compatibilidade total com config_final.py, main_model.py e unified_physics.py
"""

import numpy as np
from mesa import Agent
from mesa.time import RandomActivation
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

# Import do arquivo de configuração
import config_final as cfg


# ============================================================================
# ENUMERAÇÕES AUXILIARES
# ============================================================================

class AgentActivity(str, Enum):
    """
    Atividades possíveis dos agentes humanos.
    Baseado em ASHRAE Fundamentals e Buonanno et al. (2020).
    """
    # Atividades sedentárias
    SLEEPING = "sleeping"
    SEATED_QUIET = "seated_quiet"
    SEATED_TYPING = "seated_typing"
    SEATED_WORK = "seated_work"  # Nova: para trabalho em mesa
    READING = "reading"
    
    # Atividades em pé
    STANDING = "standing"
    WALKING = "walking"
    
    # Atividades vocais
    TALKING = "talking"
    SINGING = "singing"
    PRESENTING = "presenting"
    
    # Atividades físicas
    EXERCISING_LIGHT = "exercising_light"
    EXERCISING_INTENSE = "exercising_intense"
    
    # Atividades de alto risco
    COUGHING = "coughing"
    SNEEZING = "sneezing"
    
    # Outras atividades
    EATING = "eating"
    DRINKING = "drinking"


class ZoneType(str, Enum):
    """Tipos de zonas no ambiente."""
    CLASSROOM = "classroom"
    OFFICE_SPACE = "office_space"
    GYM_AREA = "gym_area"
    CAFETERIA = "cafeteria"
    LIBRARY = "library"
    MEETING_ROOM = "meeting_room"
    PATIENT_ROOM = "patient_room"
    RESTROOM = "restroom"
    CORRIDOR = "corridor"
    LIVING_ROOM = "living_room"
    BEDROOM = "bedroom"
    GENERAL = "general"


# ============================================================================
# CLASSE PRINCIPAL: AGENTE HUMANO
# ============================================================================

class HumanAgent(Agent):
    """
    Agente humano com fisiologia realista, comportamento adaptativo,
    capacidade de aprendizado e física de colisão.
    
    NOVIDADES DESTA VERSÃO:
    - exposure_history com snapshots completos (rastreabilidade total)
    - Detecção de colisão AABB com obstáculos
    - Busca inteligente de mobília para atividades específicas
    """
    
    def __init__(self, unique_id, model, 
                 initial_pos: Tuple[float, float],
                 agent_config: cfg.AgentConfig,
                 zone_config: Any,
                 mask_efficiency: float = 0.0,
                 initial_infected: bool = False):
        """
        Inicializa o agente humano.
        
        Args:
            unique_id: ID único do agente
            model: Referência ao modelo
            initial_pos: Posição inicial (x, y)
            agent_config: Configuração do agente (perfil)
            zone_config: Objeto de configuração da zona (contém type, name, etc)
            mask_efficiency: Eficiência da máscara (0.0 a 1.0)
            initial_infected: Se o agente começa infectado (padrão False)
        """
        super().__init__(unique_id, model)
        
        # ============================================================
        # 1. INTEGRAÇÃO COM ZONE CONFIG (Correção do Erro)
        # ============================================================
        self.zone_config = zone_config
        # Extrai propriedades de forma segura usando getattr
        self.zone = getattr(zone_config, 'name', 'Unknown Zone')
        self.zone_type = getattr(zone_config, 'zone_type', 'unknown')
        self.zone_id = getattr(zone_config, 'id', 'unknown')
        
        # ============================================================
        # 2. CONFIGURAÇÃO DO AGENTE
        # ============================================================
        # Configuração padrão se não fornecida
        if agent_config is None:
            agent_config = cfg.AgentConfig(
                activity_level=cfg.ActivityLevel.LIGHT,
                base_quanta_emission=2.0,
                activity_multiplier=1.0,
                respiration_rate=0.54,
                mask_efficiency=0.0,
                vaccination_factor=0.0
            )
        self.config = agent_config # Salva referência da config
        
        # Características físicas
        self.age_group = self._assign_age_group()
        self.weight = self._assign_weight()  # kg
        self.height = self._assign_height()  # m
        self.bmr = self._calculate_bmr()  # basal metabolic rate (W)
        
        # ============================================================
        # 3. ESTADO DE SAÚDE E INFECÇÃO
        # ============================================================
        self.infected = initial_infected
        
        if self.infected:
            # Assume que a infecção começou entre 0 e 5 dias atrás
            days_infected = np.random.uniform(0, 5)
            self.infection_start_time = -days_infected * 86400
            
            # Define carga viral inicial baseada no tempo
            peak_time = 4 * 86400  # pico médio
            duration = 12 * 86400  # duração média
            infection_duration = -self.infection_start_time
            
            if infection_duration < peak_time:
                # Fase de crescimento
                self.viral_load = infection_duration / peak_time
            elif infection_duration < duration:
                # Fase de declínio
                decline_phase = (infection_duration - peak_time) / (duration - peak_time)
                self.viral_load = max(0.0, 1.0 - decline_phase * 0.8)
            else:
                self.viral_load = 0.1  # Finalzinho da infecção
        else:
            self.infection_start_time = None
            self.viral_load = 0.0

        self.symptoms = False
        if self.infected:
            self.symptoms = np.random.random() < 0.7  # 70% sintomáticos
        
        # ============================================================
        # 4. PROTEÇÕES E COMPORTAMENTO
        # ============================================================
        self.mask_wearing = np.random.random() < mask_efficiency # Usa o argumento passado
        self.vaccinated = np.random.random() < agent_config.vaccination_factor
        self.mask_type = self._assign_mask_type() if self.mask_wearing else None
        
        # Estado comportamental
        self.current_activity = self._assign_initial_activity()
        self.activity_start_time = 0
        self.activity_duration = self._generate_activity_duration()
        self.talking = False
        self.moving = False
        self.eating = False
        self.drinking = False
        
        # Estado emocional/adaptativo
        self.comfort_level = 0.8  # 0-1
        self.risk_perception = 0.3
        self.rule_compliance = np.random.uniform(0.7, 1.0)
        self.social_distance_preference = np.random.uniform(0.5, 1.0)
        
        # ============================================================
        # 5. PRONTUÁRIO MÉDICO DETALHADO E VARIÁVEIS DE CONTROLE
        # ============================================================
        self.exposure_history = []  # Lista de dicionários detalhados
        self.movement_history = []
        self.infection_risk_history = []
        self.comfort_history = []
        
        # Parâmetros fisiológicos calculados
        self.metabolic_rate = agent_config.metabolic_rate
        self.respiration_rate = self._calculate_respiration_rate()  # respirações/min
        self.tidal_volume = self._calculate_tidal_volume()  # L
        self.inhalation_rate = self.respiration_rate * self.tidal_volume / 60.0  # L/s
        self.metabolic_heat = self._calculate_metabolic_heat()  # W
        self.moisture_production = self._calculate_moisture_production()  # kg/s
        
        # Emissões calculadas
        self.emission_rates = self._calculate_emission_rates()
        
        # Variáveis de movimento e Posição
        self.pos = initial_pos # Define a posição inicial corretamente
        self.target_pos = None
        self.movement_speed = 0.0  # m/s
        self.preferred_social_distance = np.random.uniform(1.0, 2.0)  # metros
        
        # Aprendizado (Q-Learning)
        self.q_table = {} 
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.7
        self.last_action = None
        self.last_state = None
        
        # Dose acumulada de vírus (para cálculo de infecção)
        self.accumulated_dose = 0.0

    # ========================================================================
    # MÉTODOS AUXILIARES DE INICIALIZAÇÃO
    # ========================================================================
    
    def _safe_choice(self, options: List[Any], p: Optional[List[float]] = None) -> Any:
        """
        Helper para escolher elemento de uma lista preservando seu tipo original.
        Evita que o NumPy converta Enums para strings.
        """
        opts = list(options)
        idx = np.random.choice(len(opts), p=p)
        return opts[idx]
    
    def _assign_age_group(self) -> str:
        """Atribui grupo etário."""
        groups = ["child", "teen", "young_adult", "adult", "senior"]
        probabilities = [0.1, 0.15, 0.35, 0.3, 0.1]
        return self._safe_choice(groups, p=probabilities)
    
    def _assign_weight(self) -> float:
        """Atribui peso baseado no grupo etário."""
        if self.age_group == "child":
            return np.random.normal(30, 5)
        elif self.age_group == "teen":
            return np.random.normal(60, 10)
        elif self.age_group == "young_adult":
            return np.random.normal(70, 8)
        elif self.age_group == "adult":
            return np.random.normal(75, 12)
        elif self.age_group == "senior":
            return np.random.normal(70, 10)
        else:
            return np.random.normal(75, 12)
    
    def _assign_height(self) -> float:
        """Atribui altura baseada no grupo etário."""
        if self.age_group == "child":
            return np.random.normal(1.2, 0.1)
        elif self.age_group == "teen":
            return np.random.normal(1.6, 0.15)
        elif self.age_group == "young_adult":
            return np.random.normal(1.7, 0.1)
        elif self.age_group == "adult":
            return np.random.normal(1.7, 0.1)
        elif self.age_group == "senior":
            return np.random.normal(1.65, 0.1)
        else:
            return np.random.normal(1.7, 0.1)
    
    def _assign_mask_type(self) -> str:
        """Atribui tipo de máscara."""
        mask_types = ['surgical', 'n95', 'cloth']
        probabilities = [0.5, 0.3, 0.2]
        return self._safe_choice(mask_types, p=probabilities)
    
    def _calculate_bmr(self) -> float:
        """Calcula taxa metabólica basal (Harris-Benedict simplificado)."""
        return self.weight * 1.2 * 24 / 86400 * 4184
    
    def _calculate_respiration_rate(self) -> float:
        """
        Calcula taxa de respiração baseada na atividade.
        Retorna respirações/minuto.
        """
        base_rate = 12  # respirações/min em repouso
        
        activity_multipliers = {
            AgentActivity.SLEEPING: 0.8,
            AgentActivity.SEATED_QUIET: 1.0,
            AgentActivity.SEATED_TYPING: 1.1,
            AgentActivity.SEATED_WORK: 1.1,
            AgentActivity.READING: 1.0,
            AgentActivity.STANDING: 1.2,
            AgentActivity.WALKING: 1.5,
            AgentActivity.TALKING: 1.8,
            AgentActivity.SINGING: 2.5,
            AgentActivity.PRESENTING: 2.0,
            AgentActivity.EXERCISING_LIGHT: 2.0,
            AgentActivity.EXERCISING_INTENSE: 3.0,
            AgentActivity.COUGHING: 3.0,
            AgentActivity.SNEEZING: 3.5,
            AgentActivity.EATING: 1.2,
            AgentActivity.DRINKING: 1.1,
        }
        
        multiplier = activity_multipliers.get(self.current_activity, 1.0)
        return base_rate * multiplier
    
    def _calculate_tidal_volume(self) -> float:
        """
        Calcula volume corrente (tidal volume) baseado na atividade.
        Retorna em litros.
        """
        base_tv = 0.5  # L (500 ml)
        
        activity_multipliers = {
            AgentActivity.SLEEPING: 0.9,
            AgentActivity.SEATED_QUIET: 1.0,
            AgentActivity.SEATED_TYPING: 1.0,
            AgentActivity.SEATED_WORK: 1.0,
            AgentActivity.READING: 1.0,
            AgentActivity.STANDING: 1.1,
            AgentActivity.WALKING: 1.3,
            AgentActivity.TALKING: 1.2,
            AgentActivity.SINGING: 1.5,
            AgentActivity.PRESENTING: 1.4,
            AgentActivity.EXERCISING_LIGHT: 2.0,
            AgentActivity.EXERCISING_INTENSE: 3.0,
            AgentActivity.COUGHING: 2.0,
            AgentActivity.SNEEZING: 2.5,
            AgentActivity.EATING: 1.0,
            AgentActivity.DRINKING: 1.0,
        }
        
        multiplier = activity_multipliers.get(self.current_activity, 1.0)
        return base_tv * multiplier
    
    def _calculate_metabolic_heat(self) -> float:
        """
        Calcula calor metabólico em Watts.
        Baseado em ASHRAE Fundamentals.
        """
        met_heat_map = {
            AgentActivity.SLEEPING: 70,
            AgentActivity.SEATED_QUIET: 100,
            AgentActivity.SEATED_TYPING: 105,
            AgentActivity.SEATED_WORK: 105,
            AgentActivity.READING: 100,
            AgentActivity.STANDING: 130,
            AgentActivity.WALKING: 200,
            AgentActivity.EXERCISING_LIGHT: 350,
            AgentActivity.EXERCISING_INTENSE: 600,
            AgentActivity.TALKING: 110,
            AgentActivity.SINGING: 150,
            AgentActivity.EATING: 105,
            AgentActivity.DRINKING: 100,
            AgentActivity.PRESENTING: 120
        }
        
        return met_heat_map.get(self.current_activity, 100)
    
    def _calculate_moisture_production(self) -> float:
        """Calcula produção de umidade (suor + respiração) em kg/s."""
        base_moisture = 5e-6  # ~0.3 g/min em repouso
        
        activity_multipliers = {
            AgentActivity.SLEEPING: 0.8,
            AgentActivity.SEATED_QUIET: 1.0,
            AgentActivity.SEATED_TYPING: 1.0,
            AgentActivity.SEATED_WORK: 1.0,
            AgentActivity.READING: 1.0,
            AgentActivity.STANDING: 1.2,
            AgentActivity.WALKING: 2.0,
            AgentActivity.EXERCISING_LIGHT: 5.0,
            AgentActivity.EXERCISING_INTENSE: 15.0,
            AgentActivity.TALKING: 1.5,
            AgentActivity.SINGING: 1.5,
            AgentActivity.EATING: 1.2,
            AgentActivity.DRINKING: 1.2,
            AgentActivity.PRESENTING: 1.8
        }
        
        multiplier = activity_multipliers.get(self.current_activity, 1.0)
        return base_moisture * multiplier
    
    def _assign_initial_activity(self) -> AgentActivity:
        """Atribui atividade inicial baseada no tipo de zona."""
        if self.zone_type == ZoneType.CLASSROOM:
            activities = [AgentActivity.SEATED_QUIET, AgentActivity.SEATED_TYPING, 
                         AgentActivity.READING, AgentActivity.TALKING]
            return self._safe_choice(activities)
        elif self.zone_type == ZoneType.OFFICE_SPACE:
            activities = [AgentActivity.SEATED_TYPING, AgentActivity.SEATED_WORK,
                         AgentActivity.STANDING, AgentActivity.WALKING]
            return self._safe_choice(activities)
        elif self.zone_type == ZoneType.GYM_AREA:
            activities = [AgentActivity.EXERCISING_LIGHT, AgentActivity.EXERCISING_INTENSE,
                         AgentActivity.WALKING]
            return self._safe_choice(activities)
        else:
            return AgentActivity.SEATED_QUIET
    
    def _generate_activity_duration(self) -> float:
        """Gera duração para a atividade atual (em segundos)."""
        durations = {
            AgentActivity.SLEEPING: np.random.uniform(6, 9) * 3600,
            AgentActivity.SEATED_QUIET: np.random.uniform(0.5, 2) * 3600,
            AgentActivity.SEATED_TYPING: np.random.uniform(0.5, 2) * 3600,
            AgentActivity.SEATED_WORK: np.random.uniform(0.5, 2) * 3600,
            AgentActivity.READING: np.random.uniform(0.5, 2) * 3600,
            AgentActivity.WALKING: np.random.uniform(1, 5) * 60,
            AgentActivity.EXERCISING_LIGHT: np.random.uniform(0.5, 1.5) * 3600,
            AgentActivity.EXERCISING_INTENSE: np.random.uniform(0.5, 1.5) * 3600,
            AgentActivity.EATING: np.random.uniform(10, 30) * 60,
            AgentActivity.DRINKING: np.random.uniform(10, 30) * 60,
            AgentActivity.TALKING: np.random.uniform(5, 20) * 60,
            AgentActivity.SINGING: np.random.uniform(5, 20) * 60,
            AgentActivity.PRESENTING: np.random.uniform(15, 60) * 60,
            AgentActivity.COUGHING: np.random.uniform(0.1, 0.5) * 60,
            AgentActivity.SNEEZING: np.random.uniform(0.1, 0.5) * 60,
        }
        
        return durations.get(self.current_activity, np.random.uniform(5, 30) * 60)
    
    def _calculate_emission_rates(self) -> Dict[str, float]:
        """
        Calcula taxas de emissão baseadas no estado atual.
        Retorna dicionário com taxas em kg/s ou quanta/s.
        """
        rates = {}
        
        # CO2 baseado na atividade (usa constantes do config_final)
        activity_value = self.current_activity.value
        if activity_value in cfg.HUMAN_EMISSION_RATES['co2']:
            rates['co2'] = cfg.HUMAN_EMISSION_RATES['co2'][activity_value]
        else:
            # Fallback: cálculo baseado na taxa metabólica
            met = self.metabolic_rate
            co2_production_lps = 0.00276 * met  # L/s
            rates['co2'] = co2_production_lps * 1.8e-3  # kg/s
        
        # VOCs humanos
        if self.current_activity in [AgentActivity.EXERCISING_LIGHT, AgentActivity.EXERCISING_INTENSE]:
            rates['voc'] = cfg.HUMAN_EMISSION_RATES.get('vocs', {}).get('exercising', 1e-9)
        elif self.current_activity in [AgentActivity.TALKING, AgentActivity.SINGING, AgentActivity.PRESENTING]:
            rates['voc'] = cfg.HUMAN_EMISSION_RATES.get('vocs', {}).get('active', 5e-10)
        else:
            rates['voc'] = cfg.HUMAN_EMISSION_RATES.get('vocs', {}).get('baseline', 2e-10)
        
        # ================================================================
        # EMISSÃO VIRAL
        # ================================================================
        rates['virus'] = 0.0  # Inicializa como zero
        
        if self.infected and self.viral_load > 0:
            # Taxas base de quanta por segundo (Buonanno et al. 2020)
            base_quanta_per_second = {
                # Atividades sedentárias (aumentado de 2 para 10-20 quanta/h)
                AgentActivity.SEATED_QUIET: 10.0 / 3600.0,      # 10 quanta/h (era 2)
                AgentActivity.SEATED_TYPING: 12.0 / 3600.0,     # 12 quanta/h
                AgentActivity.SEATED_WORK: 15.0 / 3600.0,       # 15 quanta/h (mais interação)
                AgentActivity.READING: 8.0 / 3600.0,           # 8 quanta/h
                AgentActivity.SLEEPING: 5.0 / 3600.0,          # 5 quanta/h
                
                # Atividades em pé
                AgentActivity.STANDING: 15.0 / 3600.0,         # 15 quanta/h
                AgentActivity.WALKING: 20.0 / 3600.0,          # 20 quanta/h
                
                # Atividades vocais (AUMENTADO SIGNIFICATIVAMENTE)
                AgentActivity.TALKING: 50.0 / 3600.0,          # 50 quanta/h (era 10)
                AgentActivity.SINGING: 200.0 / 3600.0,         # 200 quanta/h (era 30)
                AgentActivity.PRESENTING: 80.0 / 3600.0,       # 80 quanta/h
                
                # Exercício (AUMENTADO)
                AgentActivity.EXERCISING_LIGHT: 30.0 / 3600.0,  # 30 quanta/h (era 5)
                AgentActivity.EXERCISING_INTENSE: 100.0 / 3600.0, # 100 quanta/h (era 15)
                
                # Eventos de alta emissão
                AgentActivity.COUGHING: 500.0 / 3600.0,        # 500 quanta/h (evento pontual)
                AgentActivity.SNEEZING: 2000.0 / 3600.0,       # 2000 quanta/h (evento raro)
                
                # Outras
                AgentActivity.EATING: 20.0 / 3600.0,
                AgentActivity.DRINKING: 15.0 / 3600.0,
            }
            
            # Obtém taxa base para atividade atual (default para SEATED_QUIET)
            base_rate = base_quanta_per_second.get(
                self.current_activity, 
                2.0 / 3600.0  # Default: 2 quanta/h
            )
            
            # Aplica multiplicador de atividade da configuração
            activity_mult = getattr(self.config, 'activity_multiplier', 1.0)
            
            # Escala pela carga viral atual (0.0 a 1.0)
            viral_scale = max(0.0, min(1.0, self.viral_load))
            
            # Calcula emissão bruta
            raw_emission = base_rate * activity_mult * viral_scale
            
            # Aplica redução por máscara (se houver)
            if self.mask_wearing and self.mask_type:
                mask_efficiencies = {
                    'surgical': 0.5,
                    'n95': 0.95,
                    'cloth': 0.3
                }
                mask_eff = mask_efficiencies.get(self.mask_type, 0.5)
                raw_emission *= (1.0 - mask_eff)
            
            rates['virus'] = raw_emission
            
            # DEBUG: Log quando há emissão significativa
            if rates['virus'] > 1e-6 and np.random.random() < 0.01:  # 1% dos casos para não poluir
                print(f"  [AGENTE {self.unique_id}] Emitindo {rates['virus']:.6f} quanta/s "
                      f"(atividade: {self.current_activity.value}, viral_load: {self.viral_load:.2f})")
        
        return rates

    # ========================================================================
    # ATUALIZAÇÃO DE ATIVIDADES E ESTADOS
    # ========================================================================
    
    def update_activity(self):
        """Atualiza atividade baseada no tempo e comportamento."""
        current_time = self.model.schedule.time
        
        # Verifica se deve mudar de atividade
        if current_time - self.activity_start_time >= self.activity_duration:
            self._select_new_activity()
    
    def _select_new_activity(self):
        """
        Seleciona nova atividade baseada no contexto.
        
        Máquina de estados inteligente que considera:
        - Tipo de zona
        - Hora do dia
        - Dia da semana
        - Proximidade de mobília (para atividades sentadas)
        """
        current_time = self.model.schedule.time
        
        # Horários do dia
        hour_of_day = (current_time % 86400) / 3600
        is_morning = 6 <= hour_of_day < 12
        is_afternoon = 12 <= hour_of_day < 18
        is_evening = 18 <= hour_of_day < 22
        is_night = hour_of_day >= 22 or hour_of_day < 6
        
        # Dia da semana
        day_of_week = int(current_time // 86400) % 7
        is_weekday = day_of_week < 5
        
        # Lógica de atividade por tipo de zona
        if self.zone_type == ZoneType.CLASSROOM:
            if is_weekday and is_morning:
                new_activity = self._safe_choice([
                    AgentActivity.SEATED_QUIET,
                    AgentActivity.SEATED_TYPING,
                    AgentActivity.TALKING,
                    AgentActivity.STANDING,
                    AgentActivity.READING
                ], p=[0.4, 0.2, 0.2, 0.1, 0.1])
            elif is_weekday and is_afternoon:
                new_activity = self._safe_choice([
                    AgentActivity.SEATED_QUIET,
                    AgentActivity.SEATED_TYPING,
                    AgentActivity.TALKING,
                    AgentActivity.WALKING
                ], p=[0.5, 0.2, 0.2, 0.1])
            else:
                new_activity = self._safe_choice([
                    AgentActivity.WALKING,
                    AgentActivity.STANDING,
                    AgentActivity.SEATED_QUIET,
                    AgentActivity.READING
                ])
        
        elif self.zone_type == ZoneType.OFFICE_SPACE:
            if is_weekday and (is_morning or is_afternoon):
                new_activity = self._safe_choice([
                    AgentActivity.SEATED_TYPING,
                    AgentActivity.SEATED_WORK,
                    AgentActivity.SEATED_QUIET,
                    AgentActivity.TALKING,
                    AgentActivity.STANDING,
                    AgentActivity.WALKING,
                    AgentActivity.READING
                ], p=[0.3, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05])
            else:
                new_activity = self._safe_choice([
                    AgentActivity.READING,
                    AgentActivity.SEATED_QUIET,
                    AgentActivity.WALKING
                ])
        
        elif self.zone_type == ZoneType.GYM_AREA:
            if is_evening or is_afternoon:
                new_activity = self._safe_choice([
                    AgentActivity.EXERCISING_LIGHT,
                    AgentActivity.EXERCISING_INTENSE,
                    AgentActivity.WALKING,
                    AgentActivity.STANDING
                ], p=[0.5, 0.3, 0.1, 0.1])
            else:
                new_activity = self._safe_choice([
                    AgentActivity.EXERCISING_LIGHT,
                    AgentActivity.WALKING,
                    AgentActivity.STANDING
                ])
        
        elif self.zone_type == ZoneType.CAFETERIA:
            if (11 <= hour_of_day < 13) or (17 <= hour_of_day < 19):
                new_activity = self._safe_choice([
                    AgentActivity.EATING,
                    AgentActivity.DRINKING,
                    AgentActivity.TALKING
                ], p=[0.6, 0.2, 0.2])
            else:
                new_activity = self._safe_choice([
                    AgentActivity.SEATED_QUIET,
                    AgentActivity.DRINKING,
                    AgentActivity.READING,
                    AgentActivity.TALKING
                ])
        
        elif self.zone_type == ZoneType.LIBRARY:
            new_activity = self._safe_choice([
                AgentActivity.READING,
                AgentActivity.SEATED_QUIET,
                AgentActivity.SEATED_TYPING,
                AgentActivity.WALKING
            ], p=[0.5, 0.3, 0.1, 0.1])
        
        elif self.zone_type == ZoneType.MEETING_ROOM:
            if is_weekday and (is_morning or is_afternoon):
                new_activity = self._safe_choice([
                    AgentActivity.TALKING,
                    AgentActivity.PRESENTING,
                    AgentActivity.SEATED_QUIET,
                    AgentActivity.SEATED_TYPING
                ], p=[0.5, 0.2, 0.2, 0.1])
            else:
                new_activity = AgentActivity.SEATED_QUIET
        
        elif self.zone_type == ZoneType.BEDROOM:
            if is_night:
                new_activity = AgentActivity.SLEEPING
            else:
                new_activity = self._safe_choice([
                    AgentActivity.SEATED_QUIET,
                    AgentActivity.READING,
                    AgentActivity.STANDING
                ])
        
        else:
            # Para outras zonas, usa distribuição padrão
            new_activity = self._safe_choice([
                AgentActivity.SEATED_QUIET,
                AgentActivity.STANDING,
                AgentActivity.WALKING,
                AgentActivity.READING
            ])
        
        # ===================================================================
        # BUSCA INTELIGENTE DE MOBÍLIA
        # ===================================================================
        # Se a nova atividade requer sentar, procura uma mesa próxima
        if new_activity in [AgentActivity.SEATED_QUIET, AgentActivity.SEATED_TYPING, 
                           AgentActivity.SEATED_WORK, AgentActivity.READING, 
                           AgentActivity.EATING]:
            furniture_pos = self._find_nearest_furniture()
            if furniture_pos:
                self.target_pos = furniture_pos
        
        # Atualiza atividade
        self.current_activity = new_activity
        self.activity_start_time = current_time
        self.activity_duration = self._generate_activity_duration()
        
        # Recalcula parâmetros que dependem da atividade
        self.respiration_rate = self._calculate_respiration_rate()
        self.tidal_volume = self._calculate_tidal_volume()
        self.inhalation_rate = self.respiration_rate * self.tidal_volume / 60.0
        self.metabolic_heat = self._calculate_metabolic_heat()
        self.moisture_production = self._calculate_moisture_production()
        self.emission_rates = self._calculate_emission_rates()

    # ========================================================================
    # FÍSICA DE COLISÃO E MOVIMENTO
    # ========================================================================
    
    def _find_nearest_furniture(self) -> Optional[Tuple[int, int]]:
        """
        Encontra o móvel tipo FURNITURE mais próximo para sentar.
        
        Heurística simples para comportamento realista.
        - Procura obstáculos do tipo FURNITURE no cenário
        - Retorna a posição em células próxima ao móvel
        """
        if self.pos is None:
            return None
        
        # Obtém lista de obstáculos do cenário
        if not hasattr(self.model, 'scenario') or not hasattr(self.model.scenario, 'obstacles'):
            return None
        
        obstacles = self.model.scenario.obstacles
        if not obstacles:
            return None
        
        furniture_obstacles = []
        for obs in obstacles:
            # Verifica se é dicionário ou objeto Obstacle
            if isinstance(obs, dict):
                # Formato legado (dict)
                obs_type = obs.get('type', '')
                is_furniture = (obs_type == cfg.ObstacleType.FURNITURE.value)
            else:
                # É objeto Obstacle - acessa atributo diretamente
                obs_type = getattr(obs, 'obstacle_type', None)
                if obs_type is None:
                    obs_type = getattr(obs, 'type', None)
                
                # Comparação segura com Enum ou string
                is_furniture = (
                    obs_type == cfg.ObstacleType.FURNITURE or 
                    (hasattr(obs_type, 'value') and obs_type.value == cfg.ObstacleType.FURNITURE.value) or
                    str(obs_type).lower() == 'furniture'
                )
            
            if is_furniture:
                furniture_obstacles.append(obs)
        
        if not furniture_obstacles:
            return None
        
        # Encontra o mais próximo
        min_distance = float('inf')
        best_pos = None
        
        my_x_m = self.pos[0] * self.model.physics.config.cell_size
        my_y_m = self.pos[1] * self.model.physics.config.cell_size
        
        for obs in furniture_obstacles:
            # Extrai coordenadas de dict ou objeto
            if isinstance(obs, dict):
                obs_x = obs.get('x', 0.0)
                obs_y = obs.get('y', 0.0)
                obs_width = obs.get('width', 1.0)
                obs_height = obs.get('height', 1.0)
            else:
                # É objeto Obstacle - acessa atributos diretamente
                obs_x = getattr(obs, 'x', 0.0)
                obs_y = getattr(obs, 'y', 0.0)
                obs_width = getattr(obs, 'width', 1.0)
                obs_height = getattr(obs, 'height', 1.0)
            
            # Centro do obstáculo em metros
            obs_center_x = obs_x + obs_width / 2
            obs_center_y = obs_y + obs_height / 2
            
            # Distância euclidiana
            dx = obs_center_x - my_x_m
            dy = obs_center_y - my_y_m
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance < min_distance:
                min_distance = distance
                # Converte de volta para células
                target_x_cell = int(obs_center_x / self.model.physics.config.cell_size)
                target_y_cell = int(obs_center_y / self.model.physics.config.cell_size)
                best_pos = (target_x_cell, target_y_cell)
        
        return best_pos
    
    def _check_collision_with_obstacles(self, new_x, new_y):
        """
        Verifica colisão usando AABB (Axis-Aligned Bounding Box).
        Suporta objetos Obstacle e dicts.
        """
        # Converter para metros
        new_x_m = new_x * self.model.physics_config.cell_size
        new_y_m = new_y * self.model.physics_config.cell_size
        
        for obs in self.model.scenario.obstacles:
            # 1. Normalização de atributos (Híbrido Dict/Objeto)
            if isinstance(obs, dict):
                ox, oy = obs['x'], obs['y']
                ow, oh = obs['width'], obs['height']
                otype = obs['type']
            else:
                # Acesso via objeto (Novo Padrão)
                ox, oy = obs.x, obs.y
                ow, oh = obs.width, obs.height
                # Tenta acessar obstacle_type, se não tiver usa type
                if hasattr(obs, 'obstacle_type'):
                    otype = obs.obstacle_type
                else:
                    otype = getattr(obs, 'type', 'wall')

            # 2. Lógica de Colisão AABB
            if (new_x_m >= ox and new_x_m <= ox + ow and
                new_y_m >= oy and new_y_m <= oy + oh):
                
                # Se for PAREDE, bloqueia
                # Verifica se é Enum ou String para robustez
                is_wall = False
                if hasattr(cfg, 'ObstacleType'):
                    if otype == cfg.ObstacleType.WALL or otype == cfg.ObstacleType.PARTITION:
                        is_wall = True
                elif str(otype).lower() in ['wall', 'partition']:
                     is_wall = True
                
                if is_wall:
                    return True # Colidiu
                
        return False # Caminho livre
    
    def decide_movement(self) -> Optional[Tuple[int, int]]:
        """
        Decide o próximo movimento do agente.
        
        Verifica colisão antes de retornar posição.
        
        Returns:
            Nova posição (x, y) em células, ou None se não houver movimento
        """
        if self.pos is None:
            return None
        
        # Se não está em movimento e não tem destino, talvez não se mova
        if not self.moving and self.target_pos is None:
            if np.random.random() > 0.3:  # 70% de chance de ficar parado
                return None
        
        # Velocidade baseada na atividade
        if self.current_activity == AgentActivity.WALKING:
            speed = 1.2  # m/s
            self.moving = True
        elif self.current_activity in [AgentActivity.EXERCISING_LIGHT, 
                                       AgentActivity.EXERCISING_INTENSE]:
            speed = 0.8  # m/s (movendo entre equipamentos)
            self.moving = True
        elif self.current_activity in [AgentActivity.STANDING]:
            speed = 0.3  # m/s (pequenos ajustes)
            self.moving = False
        else:
            # Atividades sedentárias: sem movimento
            self.moving = False
            return None
        
        self.movement_speed = speed
        
        # Direção do movimento
        dx, dy = 0.0, 0.0
        
        # Considera distanciamento social
        nearby_agents = self.model.grid.get_neighbors(self.pos, moore=True, radius=3)
        if nearby_agents and self.social_distance_preference > 0.5:
            # Calcula vetor de repulsão
            repulsion_vector = np.zeros(2)
            
            for agent in nearby_agents:
                if isinstance(agent, HumanAgent):
                    agent_dx = self.pos[0] - agent.pos[0]
                    agent_dy = self.pos[1] - agent.pos[1]
                    distance = max(0.1, np.sqrt(agent_dx**2 + agent_dy**2))
                    
                    # Distância desejada
                    desired_distance = self.preferred_social_distance / self.model.physics.config.cell_size
                    
                    if distance < desired_distance:
                        strength = (desired_distance - distance) / desired_distance
                        repulsion_vector[0] += agent_dx / distance * strength
                        repulsion_vector[1] += agent_dy / distance * strength
            
            # Normaliza vetor de repulsão
            repulsion_norm = np.linalg.norm(repulsion_vector)
            if repulsion_norm > 0:
                repulsion_vector /= repulsion_norm
            
            # Combina com direção do destino
            if self.target_pos:
                target_dx = self.target_pos[0] - self.pos[0]
                target_dy = self.target_pos[1] - self.pos[1]
                dist = np.sqrt(target_dx**2 + target_dy**2)
                
                if dist > 0:
                    target_vector = np.array([target_dx/dist, target_dy/dist])
                    # Ponderação: 70% destino, 30% distanciamento
                    direction = 0.7 * target_vector + 0.3 * repulsion_vector
                else:
                    # Chegou ao destino
                    self.target_pos = None
                    return None
            else:
                # Movimento aleatório com distanciamento
                if np.random.random() < 0.3 or repulsion_norm > 0:
                    direction = repulsion_vector if repulsion_norm > 0 else np.random.uniform(-1, 1, 2)
                else:
                    return None
            
            # Normaliza direção final
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0:
                direction /= direction_norm
                dx, dy = direction[0], direction[1]
            else:
                return None
        else:
            # Comportamento sem distanciamento social forte
            if self.target_pos:
                target_dx = self.target_pos[0] - self.pos[0]
                target_dy = self.target_pos[1] - self.pos[1]
                dist = np.sqrt(target_dx**2 + target_dy**2)
                
                if dist < 1.0:  # Chegou ao destino
                    self.target_pos = None
                    return None
                
                if dist > 0:
                    dx = target_dx / dist
                    dy = target_dy / dist
            else:
                # Movimento aleatório
                if np.random.random() < 0.3:
                    dx = np.random.uniform(-1, 1)
                    dy = np.random.uniform(-1, 1)
                    
                    norm = np.sqrt(dx**2 + dy**2)
                    if norm > 0:
                        dx /= norm
                        dy /= norm
                else:
                    return None
        
        # Calcula nova posição em células
        dt = self.model.dt
        cells_dx = dx * speed * dt / self.model.physics.config.cell_size
        cells_dy = dy * speed * dt / self.model.physics.config.cell_size
        
        new_x = int(self.pos[0] + cells_dx)
        new_y = int(self.pos[1] + cells_dy)
        
        # ===================================================================
        # VERIFICA COLISÃO COM OBSTÁCULOS
        # ===================================================================
        # Se a nova posição colide com parede/partição, cancela o movimento
        if self._check_collision_with_obstacles(new_x, new_y):
            # Tenta uma direção alternativa simples (direita ou esquerda)
            alternative_positions = [
                (int(self.pos[0] + cells_dy), int(self.pos[1] + cells_dx)),  # 90° rotação
                (int(self.pos[0] - cells_dy), int(self.pos[1] - cells_dx)),  # -90° rotação
            ]
            
            for alt_pos in alternative_positions:
                if not self._check_collision_with_obstacles(alt_pos[0], alt_pos[1]):
                    return alt_pos
            
            # Nenhuma alternativa funciona: fica parado
            return None
        
        return (new_x, new_y)

    # ========================================================================
    # VERIFICAÇÃO E RASTREAMENTO DE INFECÇÃO
    # ========================================================================
    
    def check_infection(self, virus_concentration: float, puf_factor: float = 1.0):
        """
        Verifica e atualiza risco de infecção com cálculo correto de dose.
        """
        # Se já infectado ou vacinado, ainda registra exposição para análise
        is_susceptible = not self.infected and not self.vaccinated
        
        # ================================================================
        # CÁLCULO DE DOSE INALADA (Wells-Riley)
        # ================================================================
        
        # Concentração efetiva (quanta/m³)
        effective_concentration = virus_concentration * puf_factor
        
        # Taxa de inalação em m³/s (converte L/s para m³/s)
        inhalation_rate_m3_s = self.inhalation_rate / 1000.0  # L/s → m³/s
        
        # Tempo de exposição = dt da simulação
        dt = self.model.dt if hasattr(self.model, 'dt') else 1.0
        
        # Fator de proteção da máscara (inalação)
        mask_protection = 1.0
        if self.mask_wearing and self.mask_type:
            mask_efficiencies = {'surgical': 0.3, 'n95': 0.95, 'cloth': 0.2}
            mask_eff = mask_efficiencies.get(self.mask_type, 0.3)
            mask_protection = 1.0 - mask_eff
        
        # Dose inalada = concentração × taxa inalação × tempo × proteção
        inhaled_quanta = (effective_concentration * 
                         inhalation_rate_m3_s * 
                         dt * 
                         mask_protection)
        
        # Atualiza dose acumulada
        self.accumulated_dose += inhaled_quanta
        
        # ================================================================
        # REGISTRO NO PRONTUÁRIO MÉDICO
        # ================================================================
        if effective_concentration > 0.001 or inhaled_quanta > 0:
            exposure_record = {
                'time': self.model.schedule.time,
                'x': self.pos[0] if self.pos else None,
                'y': self.pos[1] if self.pos else None,
                'viral_load_cell': float(virus_concentration),
                'inhaled_dose': float(inhaled_quanta),
                'accumulated_dose': float(self.accumulated_dose),
                'activity': self.current_activity.value,
                'respiratory_rate': float(self.respiration_rate),
                'mask_protection': float(mask_protection),
                'is_infected': self.infected,
                'puf_factor': float(puf_factor),
                'inhalation_rate_m3_s': float(inhalation_rate_m3_s)
            }
            self.exposure_history.append(exposure_record)
        
        # ================================================================
        # TENTATIVA DE INFECÇÃO (apenas para suscetíveis)
        # ================================================================
        if is_susceptible and inhaled_quanta > 0:
            # Modelo Wells-Riley: probabilidade = 1 - exp(-dose)
            # Ajuste: dose infectante típica = 300 quanta (ID50)
            ID50 = 300.0  # quanta para 50% de chance
            infection_probability = 1.0 - np.exp(-inhaled_quanta / ID50)
            
            # Ajuste por vacinação (se aplicável no futuro)
            if self.vaccinated:
                infection_probability *= 0.1  # 90% de proteção
            
            # Atualiza percepção de risco (aprendizado)
            self.risk_perception = min(1.0, self.risk_perception + infection_probability * 0.1)
            
            # Tenta infectar
            if np.random.random() < infection_probability:
                self.infected = True
                self.infection_start_time = self.model.schedule.time
                self.viral_load = 1.0  # Começa com carga viral máxima
                self.symptoms = np.random.random() < 0.7  # 70% sintomáticos
                
                # Recalcula emissões (agora é infectante)
                self.emission_rates = self._calculate_emission_rates()
                
                print(f"  [INFECÇÃO] Agente {self.unique_id} infectado! "
                      f"(dose acumulada: {self.accumulated_dose:.4f}, "
                      f"prob: {infection_probability:.4f})")
    
    def update_infection(self):
        """Atualiza estado de infecção ao longo do tempo."""
        if not self.infected:
            return
        
        current_time = self.model.schedule.time
        
        # Segurança
        if self.infection_start_time is None:
            self.infection_start_time = current_time
        
        infection_duration = current_time - self.infection_start_time
        
        # Evolução da carga viral (curva gaussiana)
        peak_time = np.random.uniform(3, 5) * 86400  # 3-5 dias
        duration = np.random.uniform(10, 14) * 86400  # 10-14 dias
        
        if infection_duration < peak_time:
            # Fase de crescimento
            self.viral_load = infection_duration / peak_time
        elif infection_duration < duration:
            # Fase de declínio
            decline_phase = (infection_duration - peak_time) / (duration - peak_time)
            self.viral_load = 1.0 - decline_phase * 0.8
        else:
            # Recuperação
            self.infected = False
            self.viral_load = 0.0
            if 'virus' in self.emission_rates:
                self.emission_rates['virus'] = 0.0
        
        # Escala emissão viral pela carga viral
        if 'virus' in self.emission_rates:
            self.emission_rates = self._calculate_emission_rates()

    # ========================================================================
    # ADAPTAÇÃO DE COMPORTAMENTO
    # ========================================================================
    
    def adapt_behavior(self):
        """Adapta comportamento baseado no conforto e percepção de risco."""
        if self.pos is None:
            return
        
        # Obtém condições locais
        local_data = self.model.physics.get_concentrations_at(self.pos[0], self.pos[1])
        
        if not local_data:
            return
        
        # Conforto térmico
        temp_setpoint = getattr(self.model.scenario, 'temperature', 22.0)
        temp_diff = abs(local_data.get('temperature_c', temp_setpoint) - temp_setpoint)
        temp_discomfort = min(1.0, temp_diff / 5.0)
        
        # Conforto de umidade
        hum_setpoint = getattr(self.model.scenario, 'relative_humidity', 50.0)
        hum_diff = abs(local_data.get('humidity_percent', hum_setpoint) - hum_setpoint)
        hum_discomfort = min(1.0, hum_diff / 30.0)
        
        # Conforto de qualidade do ar
        co2_level = local_data.get('co2_ppm', 400)
        co2_discomfort = min(1.0, max(0, co2_level - 600) / 1000)
        
        # Conforto de velocidade do ar
        air_velocity = local_data.get('air_velocity_ms', 0.2)
        if air_velocity < 0.05:
            velocity_discomfort = 0.3  # ar estagnado
        elif air_velocity > 0.8:
            velocity_discomfort = min(1.0, (air_velocity - 0.8) / 0.5)
        else:
            velocity_discomfort = 0.0
        
        # Conforto de idade do ar
        air_age = local_data.get('air_age_minutes', 0)
        age_discomfort = min(1.0, air_age / 60.0)
        
        # Conforto combinado
        discomfort = (0.25 * temp_discomfort + 
                     0.15 * hum_discomfort + 
                     0.25 * co2_discomfort + 
                     0.10 * velocity_discomfort + 
                     0.25 * age_discomfort)
        
        self.comfort_level = max(0.0, 1.0 - discomfort)
        self.comfort_history.append({
            'time': self.model.schedule.time,
            'comfort': self.comfort_level
        })
        
        # Aprendizado: ajusta preferências
        if len(self.comfort_history) > 10:
            recent_comfort = np.mean([ch['comfort'] for ch in self.comfort_history[-10:]])
            if recent_comfort < 0.6:
                self.social_distance_preference = min(1.0, self.social_distance_preference + 0.1)
                self.preferred_social_distance = min(3.0, self.preferred_social_distance + 0.2)
        
        # Adaptação baseada no desconforto
        if discomfort > 0.6 and self.rule_compliance > 0.5:
            if np.random.random() < 0.4:
                self.target_pos = self._find_better_location()
        
        # Adaptação baseada no risco percebido
        if self.risk_perception > self.adaptation_threshold:
            # Alto risco percebido
            if not self.mask_wearing and np.random.random() < 0.6:
                self.mask_wearing = True
                self.mask_type = self._assign_mask_type()
                self.emission_rates = self._calculate_emission_rates()
            
            # Tenta se afastar de áreas lotadas
            nearby_agents = self.model.grid.get_neighbors(self.pos, moore=True, radius=3)
            if len(nearby_agents) > 4:
                if np.random.random() < 0.5:
                    self.target_pos = self._find_less_crowded_location()
            
            # Aumenta preferência por distanciamento
            self.social_distance_preference = min(1.0, self.social_distance_preference + 0.1)
    
    def _find_better_location(self) -> Optional[Tuple[int, int]]:
        """Encontra local com melhor qualidade do ar."""
        if self.pos is None:
            return None
        
        x, y = self.pos
        
        # Procura em raio crescente
        for search_radius in [5, 10, 15]:
            best_score = -float('inf')
            best_pos = None
            
            for dx in range(-search_radius, search_radius + 1):
                for dy in range(-search_radius, search_radius + 1):
                    nx, ny = x + dx, y + dy
                    
                    if 0 <= nx < self.model.physics.cells_x and 0 <= ny < self.model.physics.cells_y:
                        # Verifica colisão
                        if self._check_collision_with_obstacles(nx, ny):
                            continue
                        
                        data = self.model.physics.get_concentrations_at(nx, ny)
                        if data:
                            # Score baseado em múltiplos fatores
                            temp_setpoint = getattr(self.model.scenario, 'temperature', 22.0)
                            temp_score = 100 - abs(data.get('temperature_c', temp_setpoint) - temp_setpoint) * 10
                            co2_score = 100 - max(0, data.get('co2_ppm', 400) - 400) / 10
                            hum_setpoint = getattr(self.model.scenario, 'relative_humidity', 50.0)
                            hum_score = 100 - abs(data.get('humidity_percent', hum_setpoint) - hum_setpoint) * 2
                            air_age_score = 100 - min(100, data.get('air_age_minutes', 0))
                            velocity = data.get('air_velocity_ms', 0.2)
                            velocity_score = 100 if 0.1 <= velocity <= 0.5 else 50
                            
                            # Evita áreas com alta concentração viral
                            virus_score = 100 - min(100, data.get('virus_quanta_m3', 0) * 1e3)
                            
                            score = (temp_score * 0.25 + co2_score * 0.25 + 
                                    hum_score * 0.15 + air_age_score * 0.15 + 
                                    velocity_score * 0.10 + virus_score * 0.10)
                            
                            if score > best_score and self.model.grid.is_cell_empty((nx, ny)):
                                best_score = score
                                best_pos = (nx, ny)
            
            if best_pos:
                return best_pos
        
        return None
    
    def _find_less_crowded_location(self) -> Optional[Tuple[int, int]]:
        """Encontra local com menos agentes próximos."""
        if self.pos is None:
            return None
        
        x, y = self.pos
        
        # Procura células vazias em raio crescente
        for radius in range(1, 8):
            candidate_positions = []
            
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = x + dx, y + dy
                    
                    if 0 <= nx < self.model.physics.cells_x and 0 <= ny < self.model.physics.cells_y:
                        # Verifica colisão
                        if self._check_collision_with_obstacles(nx, ny):
                            continue
                        
                        if self.model.grid.is_cell_empty((nx, ny)):
                            # Conta agentes próximos
                            nearby = self.model.grid.get_neighbors((nx, ny), moore=True, radius=2)
                            agent_count = len([a for a in nearby if isinstance(a, HumanAgent)])
                            distance = np.sqrt(dx**2 + dy**2)
                            
                            candidate_positions.append({
                                'pos': (nx, ny),
                                'agent_count': agent_count,
                                'distance': distance
                            })
            
            if candidate_positions:
                # Ordena por contagem de agentes, depois por distância
                candidate_positions.sort(key=lambda cp: (cp['agent_count'], cp['distance']))
                return candidate_positions[0]['pos']
        
        return None

    # ========================================================================
    # MÉTODO PRINCIPAL: STEP
    # ========================================================================
    
    def step(self):
        """Executa um passo do agente."""
        # 1. Atualiza atividade
        self.update_activity()
        
        # 2. Atualiza infecção
        self.update_infection()
        
        # 3. Adapta comportamento
        self.adapt_behavior()
        
        # 4. Decide movimento (com detecção de colisão)
        new_pos = self.decide_movement()
        if new_pos and self.model.grid.is_cell_empty(new_pos):
            self.model.grid.move_agent(self, new_pos)
            self.pos = new_pos
        
        # 5. Emite poluentes, calor e umidade
        self.emit_pollutants()
        
        # 6. Verifica infecção (com registro detalhado)
        if not self.infected and not self.vaccinated:
            if self.pos:
                local_data = self.model.physics.get_concentrations_at(self.pos[0], self.pos[1])
                if local_data:
                    virus_concentration = local_data.get('virus_exposure_quanta_m3', 
                                                         local_data.get('virus_quanta_m3', 0))
                    puf_factor = local_data.get('puf_factor', 1.0)
                    self.check_infection(virus_concentration, puf_factor)
    
    def emit_pollutants(self):
        """Emite poluentes, calor e umidade na posição atual."""
        if self.pos is None:
            return
        
        x, y = self.pos
        
        # ================================================================
        # PREPARA EMISSÕES
        # ================================================================
        emissions_list = []
        
        # Emissões de poluentes (CO2, VOCs, Vírus)
        for species, rate in self.emission_rates.items():
            if rate > 0 and species in ['co2', 'voc', 'virus']:
                amount = rate * self.model.dt
                emissions_list.append({
                    'x': x,
                    'y': y,
                    'species': species,
                    'amount': amount
                })
                
                # DEBUG para vírus
                # if species == 'virus' and amount > 1e-6:
                #     print(f"  [EMISSÃO] Agente {self.unique_id} emite {amount:.6f} quanta "
                #           f"na célula ({x}, {y})")
        
        # Calor metabólico
        if self.metabolic_heat > 0:
            emissions_list.append({
                'x': x,
                'y': y,
                'species': 'heat',
                'amount': self.metabolic_heat * self.model.dt
            })
        
        # Umidade
        if self.moisture_production > 0:
            emissions_list.append({
                'x': x,
                'y': y,
                'species': 'moisture',
                'amount': self.moisture_production * self.model.dt
            })
        
        # ================================================================
        # ENVIA PARA O MODELO
        # ================================================================
        if hasattr(self.model, 'add_agent_emissions'):
            self.model.add_agent_emissions(emissions_list)
        elif hasattr(self.model, 'current_agent_emissions'):
            # Acumula para processamento em lote
            self.model.current_agent_emissions.extend(emissions_list)
        elif hasattr(self.model, 'physics'):
            # Envia diretamente para motor físico
            for emission in emissions_list:
                species_amount = {emission['species']: emission['amount']}
                self.model.physics.add_agent_emission(
                    emission['x'], 
                    emission['y'],
                    species_amount,
                    metabolic_heat=0.0,
                    moisture_production=0.0
                )
    
    def get_agent_data(self) -> Dict[str, Any]:
        """Retorna dados do agente para análise."""
        local_data = self.model.physics.get_concentrations_at(self.pos[0], self.pos[1]) if self.pos else {}
        
        data = {
            'id': self.unique_id,
            'zone': self.zone,
            'zone_type': self.zone_type.value,
            'age_group': self.age_group,
            'weight': self.weight,
            'height': self.height,
            'infected': self.infected,
            'viral_load': self.viral_load,
            'symptoms': self.symptoms,
            'mask_wearing': self.mask_wearing,
            'mask_type': self.mask_type,
            'vaccinated': self.vaccinated,
            'current_activity': self.current_activity.value,
            'metabolic_rate': self.metabolic_rate,
            'metabolic_heat_w': self.metabolic_heat,
            'moisture_production_kg_s': self.moisture_production,
            'respiration_rate_breaths_min': self.respiration_rate,
            'tidal_volume_l': self.tidal_volume,
            'inhalation_rate_m3_s': self.inhalation_rate / 1000.0,
            'comfort_level': self.comfort_level,
            'risk_perception': self.risk_perception,
            'rule_compliance': self.rule_compliance,
            'social_distance_preference': self.social_distance_preference,
            'preferred_social_distance_m': self.preferred_social_distance,
            'position': self.pos,
            'local_concentrations': local_data,
            'emission_rates': self.emission_rates,
            'total_exposure': sum(exp['inhaled_dose'] for exp in self.exposure_history) if self.exposure_history else 0,
            'accumulated_dose': self.accumulated_dose,
            'exposure_count': len(self.exposure_history),
            'average_comfort': np.mean([ch['comfort'] for ch in self.comfort_history]) if self.comfort_history else 0.0,
            'movement_speed_ms': self.movement_speed
        }
        
        return data


# ============================================================================
# AGENDADOR ADAPTATIVO
# ============================================================================

class AdaptiveScheduler(RandomActivation):
    """
    Agendador adaptativo que prioriza agentes com maior impacto.
    """
    
    def __init__(self, model):
        super().__init__(model)
        self.prioritization_enabled = True
    
    def step(self) -> None:
        """Executa um passo para todos os agentes, com priorização."""
        agents = list(self.agents)
        
        if self.prioritization_enabled:
            # Ordena agentes por prioridade
            agents.sort(key=lambda x: self._calculate_priority(x))
        
        for agent in agents:
            agent.step()
        
        self.steps += 1
        self.time += 1
    
    def _calculate_priority(self, agent: HumanAgent) -> float:
        """Calcula prioridade do agente para agendamento."""
        priority = 0.0
        
        # Infectados têm alta prioridade
        if agent.infected:
            priority += 1000
            priority += agent.viral_load * 500
        
        # Agentes com alta emissão viral
        if hasattr(agent, 'emission_rates') and 'virus' in agent.emission_rates:
            priority += agent.emission_rates['virus'] * 1000
        
        # Agentes em atividades de alto risco
        high_risk_activities = [
            AgentActivity.COUGHING,
            AgentActivity.SNEEZING,
            AgentActivity.SINGING,
            AgentActivity.TALKING,
            AgentActivity.EXERCISING_INTENSE
        ]
        
        if agent.current_activity in high_risk_activities:
            priority += 100
        
        # Agentes com baixo conforto
        if agent.comfort_level < 0.6:
            priority += 50
        
        # Agentes com alto risco percebido
        if agent.risk_perception > 0.7:
            priority += 30
        
        # Agentes em movimento
        if agent.moving:
            priority += 20
        
        return -priority  # Ordenação decrescente


# ============================================================================
# AGENTE COM APRENDIZADO POR REFORÇO
# ============================================================================

class LearningAgent(HumanAgent):
    """
    Agente com capacidade de aprendizado por reforço.
    Extensão do HumanAgent com Q-learning.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Parâmetros de aprendizado
        self.q_table = {}  # Tabela Q
        self.epsilon = 0.1  # Taxa de exploração
        self.alpha = 0.1  # Taxa de aprendizado
        self.gamma = 0.9  # Fator de desconto
        
        # Histórico
        self.reward_history = []
        self.action_history = []
    
    def adapt_behavior(self):
        """Adapta comportamento usando aprendizado por reforço."""
        super().adapt_behavior()
        
        # Estado atual
        state = self._get_state()
        
        # Escolhe ação usando política ε-greedy
        if np.random.random() < self.epsilon:
            action = self._choose_random_action()
        else:
            action = self._choose_best_action(state)
        
        # Executa ação
        reward = self._execute_action(action)
        
        # Atualiza Q-table
        self._update_q_table(state, action, reward)
        
        # Registra histórico
        self.reward_history.append(reward)
        self.action_history.append(action)
    
    def _get_state(self) -> str:
        """Codifica o estado atual."""
        state_parts = []
        
        # Informações de saúde
        state_parts.append('I' if self.infected else 'H')
        state_parts.append('M' if self.mask_wearing else 'N')
        
        # Nível de risco
        if self.risk_perception < 0.3:
            state_parts.append('LR')
        elif self.risk_perception < 0.7:
            state_parts.append('MR')
        else:
            state_parts.append('HR')
        
        # Nível de conforto
        if self.comfort_level > 0.7:
            state_parts.append('HC')
        elif self.comfort_level > 0.4:
            state_parts.append('MC')
        else:
            state_parts.append('LC')
        
        # Densidade local
        if self.pos:
            nearby = self.model.grid.get_neighbors(self.pos, moore=True, radius=2)
            density = len(nearby)
            if density < 2:
                state_parts.append('LD')
            elif density < 5:
                state_parts.append('MD')
            else:
                state_parts.append('HD')
        
        return '_'.join(state_parts)
    
    def _choose_random_action(self) -> str:
        """Escolhe ação aleatória."""
        actions = [
            'stay', 'move_random', 'move_to_better', 
            'wear_mask', 'remove_mask', 'increase_distance'
        ]
        return self._safe_choice(actions)
    
    def _choose_best_action(self, state: str) -> str:
        """Escolhe melhor ação baseada na Q-table."""
        if state not in self.q_table:
            return self._choose_random_action()
        
        q_values = self.q_table[state]
        return max(q_values.items(), key=lambda x: x[1])[0]
    
    def _execute_action(self, action: str) -> float:
        """Executa ação e retorna recompensa."""
        reward = 0.0
        
        if action == 'stay':
            reward += 0.1
        
        elif action == 'move_random':
            reward += 0.05
        
        elif action == 'move_to_better':
            new_pos = self._find_better_location()
            if new_pos:
                self.target_pos = new_pos
                reward += 0.2
        
        elif action == 'wear_mask' and not self.mask_wearing:
            self.mask_wearing = True
            self.mask_type = self._assign_mask_type()
            self.emission_rates = self._calculate_emission_rates()
            
            if self.risk_perception > 0.5:
                reward += 0.3
            else:
                reward += 0.1
        
        elif action == 'remove_mask' and self.mask_wearing:
            self.mask_wearing = False
            self.mask_type = None
            self.emission_rates = self._calculate_emission_rates()
            
            if self.risk_perception < 0.3 and self.comfort_level > 0.7:
                reward += 0.2
        
        elif action == 'increase_distance':
            self.social_distance_preference = min(1.0, self.social_distance_preference + 0.1)
            self.preferred_social_distance = min(3.0, self.preferred_social_distance + 0.2)
            
            if self.risk_perception > 0.6:
                reward += 0.15
        
        # Penalidade por desconforto
        reward -= (1 - self.comfort_level) * 0.1
        
        # Penalidade por exposição
        if self.exposure_history:
            recent_exposure = self.exposure_history[-1]['inhaled_dose'] if self.exposure_history else 0
            reward -= recent_exposure * 100
        
        return reward
    
    def _update_q_table(self, state: str, action: str, reward: float):
        """Atualiza Q-table usando Q-learning."""
        if state not in self.q_table:
            self.q_table[state] = {}
        
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        
        # Q-learning update
        next_state = self._get_state()
        next_max = max(self.q_table.get(next_state, {}).values(), default=0.0)
        
        self.q_table[state][action] = (
            (1 - self.alpha) * self.q_table[state][action] +
            self.alpha * (reward + self.gamma * next_max)
        )
