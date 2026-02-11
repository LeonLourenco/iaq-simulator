"""
SISTEMA DE AGENTES 
Agentes humanos com fisiologia realista, comportamento adaptativo
e capacidade de aprendizado
"""

import numpy as np
from mesa import Agent
from mesa.time import RandomActivation
from typing import Dict, List, Tuple, Optional, Any
import config_final as cfg

class HumanAgent(Agent):
    """
    Agente humano com fisiologia realista, comportamento adaptativo
    e capacidade de aprendizado.
    """
    
    def __init__(self, unique_id: int, model, zone_config: cfg.ZoneConfig, 
                 agent_config: cfg.AgentConfig, initial_infected: bool = False):
        super().__init__(unique_id, model)
        
        # Informações básicas
        self.zone = zone_config.name
        self.zone_type = zone_config.zone_type
        
        # Características físicas
        self.age_group = self._assign_age_group(agent_config.age_distribution)
        self.weight = self._assign_weight()  # kg
        self.height = self._assign_height()  # m
        self.bmr = self._calculate_bmr()  # basal metabolic rate (W)
        
        # Estado de saúde
        self.infected = initial_infected
        
        # Inicialização correta de infectados (evita erro de NoneType)
        if self.infected:
            # Assume que a infecção começou entre 0 e 5 dias atrás
            days_infected = np.random.uniform(0, 5)
            self.infection_start_time = -days_infected * 86400
            
            # Define carga viral inicial baseada no tempo
            peak_time = 4 * 86400 # pico médio
            duration = 12 * 86400 # duração média
            infection_duration = -self.infection_start_time
            
            if infection_duration < peak_time:
                self.viral_load = infection_duration / peak_time
            elif infection_duration < duration:
                decline_phase = (infection_duration - peak_time) / (duration - peak_time)
                self.viral_load = max(0.0, 1.0 - decline_phase * 0.8)
            else:
                self.viral_load = 0.1 # Finalzinho da infecção
        else:
            self.infection_start_time = None
            self.viral_load = 0.0

        self.symptoms = False
        if self.infected:
            self.symptoms = np.random.random() < 0.7 # 70% sintomáticos
            
        self.mask_wearing = np.random.random() < agent_config.mask_wearing_prob
        self.vaccinated = np.random.random() < agent_config.vaccination_rate
        self.mask_type = self._assign_mask_type() if self.mask_wearing else None
        
        # Estado comportamental
        self.current_activity = self._assign_initial_activity(agent_config.activity_distribution)
        self.activity_start_time = 0
        self.activity_duration = self._generate_activity_duration()
        self.talking = False
        self.moving = False
        self.eating = False
        self.drinking = False
        
        # Estado emocional/adaptativo
        self.comfort_level = 0.8  # 0-1
        self.risk_perception = 0.3
        self.rule_compliance = agent_config.compliance_to_rules * np.random.uniform(0.8, 1.2)
        self.social_distance_preference = np.random.uniform(0.5, 1.0)
        
        # Histórico
        self.exposure_history = []
        self.movement_history = []
        self.infection_risk_history = []
        self.comfort_history = []
        
        # Parâmetros fisiológicos
        self.metabolic_rate = agent_config.metabolic_rates.get(self.current_activity, 1.0)
        self.respiration_rate = self._calculate_respiration_rate()  # respirações/min
        self.tidal_volume = self._calculate_tidal_volume()  # L
        self.inhalation_rate = self.respiration_rate * self.tidal_volume / 60.0  # L/s
        self.metabolic_heat = self._calculate_metabolic_heat()  # W
        self.moisture_production = self._calculate_moisture_production()  # kg/s
        
        # Emissões calculadas
        self.emission_rates = self._calculate_emission_rates()
        
        # Posição inicial
        self.pos = None
        self.target_pos = None
        self.movement_speed = 0.0  # m/s
        self.preferred_social_distance = np.random.uniform(1.0, 2.0)  # metros
        
        # Aprendizado
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.7

    def _safe_choice(self, options: List[Any], p: Optional[List[float]] = None) -> Any:
        """
        Helper para escolher elemento de uma lista preservando seu tipo original.
        Evita que o NumPy converta Enums para strings (numpy.str_).
        """
        opts = list(options)
        idx = np.random.choice(len(opts), p=p)
        return opts[idx]
        
    def _assign_age_group(self, age_distribution: Dict[str, float]) -> str:
        """Atribui grupo etário baseado na distribuição."""
        groups = list(age_distribution.keys())
        probabilities = list(age_distribution.values())
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
        """Calcula taxa metabólica basal (Harris-Benedict)."""
        return self.weight * 1.2 * 24 / 86400 * 4184
    
    def _calculate_respiration_rate(self) -> float:
        """Calcula taxa de respiração baseada na atividade."""
        base_rate = 12
        if self.current_activity == cfg.AgentActivity.SLEEPING:
            return base_rate * 0.8
        elif self.current_activity in [cfg.AgentActivity.SEATED_QUIET, cfg.AgentActivity.READING]:
            return base_rate
        elif self.current_activity == cfg.AgentActivity.SEATED_TYPING:
            return base_rate * 1.1
        elif self.current_activity == cfg.AgentActivity.STANDING:
            return base_rate * 1.2
        elif self.current_activity == cfg.AgentActivity.WALKING:
            return base_rate * 1.5
        elif self.current_activity == cfg.AgentActivity.TALKING:
            return base_rate * 1.8
        elif self.current_activity == cfg.AgentActivity.SINGING:
            return base_rate * 2.5
        elif self.current_activity == cfg.AgentActivity.EXERCISING_LIGHT:
            return base_rate * 2.0
        elif self.current_activity == cfg.AgentActivity.EXERCISING_INTENSE:
            return base_rate * 3.0
        elif self.current_activity == cfg.AgentActivity.COUGHING:
            return base_rate * 3.0
        elif self.current_activity == cfg.AgentActivity.SNEEZING:
            return base_rate * 3.5
        elif self.current_activity == cfg.AgentActivity.EATING:
            return base_rate * 1.2
        elif self.current_activity == cfg.AgentActivity.DRINKING:
            return base_rate * 1.1
        elif self.current_activity == cfg.AgentActivity.PRESENTING:
            return base_rate * 2.0
        else:
            return base_rate
    
    def _calculate_tidal_volume(self) -> float:
        """Calcula volume corrente baseado na atividade e características."""
        base_tv = 500
        if self.current_activity == cfg.AgentActivity.SLEEPING:
            factor = 0.8
        elif self.current_activity in [cfg.AgentActivity.SEATED_QUIET, cfg.AgentActivity.SEATED_TYPING]:
            factor = 1.0
        elif self.current_activity in [cfg.AgentActivity.STANDING, cfg.AgentActivity.WALKING]:
            factor = 1.2
        elif self.current_activity in [cfg.AgentActivity.EXERCISING_LIGHT, cfg.AgentActivity.TALKING]:
            factor = 1.5
        elif self.current_activity == cfg.AgentActivity.EXERCISING_INTENSE:
            factor = 2.0
        elif self.current_activity in [cfg.AgentActivity.SINGING, cfg.AgentActivity.COUGHING, cfg.AgentActivity.SNEEZING]:
            factor = 1.8
        elif self.current_activity == cfg.AgentActivity.EATING:
            factor = 1.1
        elif self.current_activity == cfg.AgentActivity.DRINKING:
            factor = 1.1
        elif self.current_activity == cfg.AgentActivity.PRESENTING:
            factor = 1.5
        else:
            factor = 1.0
        return (base_tv * factor) / 1000.0
    
    def _calculate_metabolic_heat(self) -> float:
        """Calcula calor metabólico baseado na atividade."""
        if 'heat' in cfg.HUMAN_EMISSION_RATES:
            heat_rates = cfg.HUMAN_EMISSION_RATES['heat']
            activity_key = self.current_activity.value
            for key in heat_rates:
                if key in activity_key:
                    return heat_rates[key]
        
        met_heat_map = {
            cfg.AgentActivity.SLEEPING: 70,
            cfg.AgentActivity.SEATED_QUIET: 100,
            cfg.AgentActivity.SEATED_TYPING: 115,
            cfg.AgentActivity.STANDING: 130,
            cfg.AgentActivity.WALKING: 200,
            cfg.AgentActivity.EXERCISING_LIGHT: 350,
            cfg.AgentActivity.EXERCISING_INTENSE: 600,
            cfg.AgentActivity.TALKING: 110,
            cfg.AgentActivity.SINGING: 150,
            cfg.AgentActivity.EATING: 105,
            cfg.AgentActivity.DRINKING: 100,
            cfg.AgentActivity.READING: 100,
            cfg.AgentActivity.PRESENTING: 120
        }
        return met_heat_map.get(self.current_activity, 100)
    
    def _calculate_moisture_production(self) -> float:
        """Calcula produção de umidade (suor + respiração) em kg/s."""
        base_moisture = 5e-6
        if self.current_activity == cfg.AgentActivity.SLEEPING:
            return base_moisture * 0.8
        elif self.current_activity in [cfg.AgentActivity.SEATED_QUIET, cfg.AgentActivity.SEATED_TYPING]:
            return base_moisture
        elif self.current_activity == cfg.AgentActivity.STANDING:
            return base_moisture * 1.2
        elif self.current_activity == cfg.AgentActivity.WALKING:
            return base_moisture * 2.0
        elif self.current_activity == cfg.AgentActivity.EXERCISING_LIGHT:
            return base_moisture * 5.0
        elif self.current_activity == cfg.AgentActivity.EXERCISING_INTENSE:
            return base_moisture * 15.0
        elif self.current_activity in [cfg.AgentActivity.TALKING, cfg.AgentActivity.SINGING]:
            return base_moisture * 1.5
        elif self.current_activity in [cfg.AgentActivity.EATING, cfg.AgentActivity.DRINKING]:
            return base_moisture * 1.2
        elif self.current_activity == cfg.AgentActivity.PRESENTING:
            return base_moisture * 1.8
        else:
            return base_moisture
    
    def _assign_initial_activity(self, activity_distribution: Dict[cfg.AgentActivity, float]) -> cfg.AgentActivity:
        """Atribui atividade inicial baseada na distribuição."""
        activities = list(activity_distribution.keys())
        probabilities = list(activity_distribution.values())
        return self._safe_choice(activities, p=probabilities)
    
    def _generate_activity_duration(self) -> float:
        """Gera duração para a atividade atual."""
        if self.current_activity == cfg.AgentActivity.SLEEPING:
            return np.random.uniform(6, 9) * 3600
        elif self.current_activity in [cfg.AgentActivity.SEATED_QUIET, cfg.AgentActivity.SEATED_TYPING, cfg.AgentActivity.READING]:
            return np.random.uniform(0.5, 2) * 3600
        elif self.current_activity == cfg.AgentActivity.WALKING:
            return np.random.uniform(1, 5) * 60
        elif self.current_activity in [cfg.AgentActivity.EXERCISING_LIGHT, cfg.AgentActivity.EXERCISING_INTENSE]:
            return np.random.uniform(0.5, 1.5) * 3600
        elif self.current_activity in [cfg.AgentActivity.EATING, cfg.AgentActivity.DRINKING]:
            return np.random.uniform(10, 30) * 60
        elif self.current_activity in [cfg.AgentActivity.TALKING, cfg.AgentActivity.SINGING]:
            return np.random.uniform(5, 20) * 60
        elif self.current_activity == cfg.AgentActivity.PRESENTING:
            return np.random.uniform(15, 60) * 60
        elif self.current_activity in [cfg.AgentActivity.COUGHING, cfg.AgentActivity.SNEEZING]:
            return np.random.uniform(0.1, 0.5) * 60
        else:
            return np.random.uniform(5, 30) * 60
    
    def _calculate_emission_rates(self) -> Dict[str, float]:
        """Calcula taxas de emissão baseadas no estado atual."""
        rates = {}
        
        if self.current_activity.value in cfg.HUMAN_EMISSION_RATES['co2']:
            rates['co2'] = cfg.HUMAN_EMISSION_RATES['co2'][self.current_activity.value]
        else:
            met = self.metabolic_rate
            co2_production_lps = 0.00276 * met
            rates['co2'] = co2_production_lps * 1.8e-3
        
        if self.current_activity in [cfg.AgentActivity.EXERCISING_LIGHT, cfg.AgentActivity.EXERCISING_INTENSE]:
            rates['voc'] = cfg.HUMAN_EMISSION_RATES['vocs']['exercising']
        elif self.current_activity in [cfg.AgentActivity.TALKING, cfg.AgentActivity.SINGING, cfg.AgentActivity.PRESENTING]:
            rates['voc'] = cfg.HUMAN_EMISSION_RATES['vocs']['active']
        else:
            rates['voc'] = cfg.HUMAN_EMISSION_RATES['vocs']['baseline']
        
        rates['virus'] = 0.0
        if self.infected:
            activity_key = self.current_activity.value
            if activity_key in cfg.HUMAN_EMISSION_RATES['quanta']:
                rates['virus'] = cfg.HUMAN_EMISSION_RATES['quanta'][activity_key]
            elif self.current_activity == cfg.AgentActivity.COUGHING:
                rates['virus'] = cfg.HUMAN_EMISSION_RATES['quanta']['coughing']
            elif self.current_activity == cfg.AgentActivity.SNEEZING:
                rates['virus'] = cfg.HUMAN_EMISSION_RATES['quanta']['sneezing']
            elif self.current_activity in [cfg.AgentActivity.SINGING, cfg.AgentActivity.PRESENTING]:
                rates['virus'] = cfg.HUMAN_EMISSION_RATES['quanta']['singing']
            elif self.current_activity == cfg.AgentActivity.TALKING:
                rates['virus'] = cfg.HUMAN_EMISSION_RATES['quanta']['talking']
            elif self.current_activity == cfg.AgentActivity.EXERCISING_INTENSE:
                rates['virus'] = cfg.HUMAN_EMISSION_RATES['quanta']['exercising_intense']
            elif self.current_activity == cfg.AgentActivity.EXERCISING_LIGHT:
                rates['virus'] = cfg.HUMAN_EMISSION_RATES['quanta']['exercising_light']
            else:
                rates['virus'] = cfg.HUMAN_EMISSION_RATES['quanta']['breathing']
            
            if self.mask_wearing and self.mask_type:
                mask_efficiency = cfg.INTERVENTION_EFFECTIVENESS.get(
                    f'mask_{self.mask_type}', 
                    {'virus_emission': 0.5}
                )['virus_emission']
                rates['virus'] *= (1 - mask_efficiency)
            
            rates['virus'] *= self.viral_load
        
        return rates
    
    def update_activity(self):
        """Atualiza atividade baseada no tempo e comportamento."""
        current_time = self.model.schedule.time
        if current_time - self.activity_start_time >= self.activity_duration:
            self._select_new_activity()
    
    def _select_new_activity(self):
        """Seleciona nova atividade baseada no contexto."""
        current_time = self.model.schedule.time
        
        activities = list(self.model.agent_config.activity_distribution.keys())
        probabilities = list(self.model.agent_config.activity_distribution.values())
        new_activity = self._safe_choice(activities, p=probabilities)
        
        self.current_activity = new_activity
        self.activity_start_time = current_time
        self.activity_duration = self._generate_activity_duration()
        
        self.talking = new_activity in [cfg.AgentActivity.TALKING, cfg.AgentActivity.SINGING, cfg.AgentActivity.PRESENTING]
        self.eating = new_activity == cfg.AgentActivity.EATING
        self.drinking = new_activity == cfg.AgentActivity.DRINKING
        self.moving = new_activity in [cfg.AgentActivity.WALKING, cfg.AgentActivity.EXERCISING_LIGHT, cfg.AgentActivity.EXERCISING_INTENSE]
        
        self.metabolic_rate = self.model.agent_config.metabolic_rates.get(new_activity, 1.0)
        self.respiration_rate = self._calculate_respiration_rate()
        self.tidal_volume = self._calculate_tidal_volume()
        self.inhalation_rate = self.respiration_rate * self.tidal_volume / 60.0
        self.metabolic_heat = self._calculate_metabolic_heat()
        self.moisture_production = self._calculate_moisture_production()
        self.emission_rates = self._calculate_emission_rates()
    
    def decide_movement(self):
        """Decide movimento baseado no comportamento atual."""
        if not self.moving:
            return None
        
        if self.current_activity == cfg.AgentActivity.WALKING:
            speed = 1.4
        elif self.current_activity in [cfg.AgentActivity.EXERCISING_LIGHT, cfg.AgentActivity.EXERCISING_INTENSE]:
            speed = 2.0
        else:
            speed = 0.0
        
        self.movement_speed = speed
        direction = np.zeros(2)
        has_force = False
        nearby_agents = self.model.grid.get_neighbors(self.pos, moore=True, radius=3)
        
        # 1. Força de Distanciamento Social
        if nearby_agents and self.social_distance_preference > 0.5:
            repulsion_vector = np.zeros(2)
            for agent in nearby_agents:
                if isinstance(agent, HumanAgent):
                    dx = self.pos[0] - agent.pos[0]
                    dy = self.pos[1] - agent.pos[1]
                    distance = max(0.1, np.sqrt(dx**2 + dy**2))
                    desired_distance = self.preferred_social_distance / self.model.physics.config.cell_size
                    
                    if distance < desired_distance:
                        strength = (desired_distance - distance) / desired_distance
                        repulsion_vector[0] += dx / distance * strength
                        repulsion_vector[1] += dy / distance * strength
            
            repulsion_norm = np.linalg.norm(repulsion_vector)
            if repulsion_norm > 0:
                direction += repulsion_vector / repulsion_norm * 0.6  # Peso da repulsão
                has_force = True

        # 2. Força de Destino (Target)
        if self.target_pos:
            dx = self.target_pos[0] - self.pos[0]
            dy = self.target_pos[1] - self.pos[1]
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < 1.0:
                self.target_pos = None
                return None # Chegou
            
            if dist > 0:
                target_vector = np.array([dx/dist, dy/dist])
                direction += target_vector * 0.4  # Peso do destino
                has_force = True
        
        # 3. Componente Aleatória (se não houver forças fortes ou para variar)
        if not has_force or np.random.random() < 0.2:
            random_vec = np.random.uniform(-1, 1, 2)
            direction += random_vec * 0.5
        
        # Normalização Final
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0:
            direction /= direction_norm
            dx, dy = direction[0], direction[1]
        else:
            # Fallback seguro: movimento aleatório
            dx = np.random.uniform(-1, 1)
            dy = np.random.uniform(-1, 1)
            norm = np.sqrt(dx**2 + dy**2)
            if norm > 0:
                dx /= norm
                dy /= norm
        
        # Calcula nova posição em células
        dt = self.model.dt
        cells_dx = dx * speed * dt / self.model.physics.config.cell_size
        cells_dy = dy * speed * dt / self.model.physics.config.cell_size
        
        new_x = int(self.pos[0] + cells_dx)
        new_y = int(self.pos[1] + cells_dy)
        
        # --- VERIFICAÇÃO DE OBSTÁCULOS ---
        if self.model.physics.is_walkable(new_x, new_y):
            return (new_x, new_y)
        
        return None
    
    def check_infection(self, virus_concentration: float, puf_factor: float = 1.0):
        """Verifica e atualiza risco de infecção."""
        if self.infected or self.vaccinated:
            return
        
        effective_concentration = virus_concentration * puf_factor
        mask_protection = 1.0
        if self.mask_wearing and self.mask_type:
            mask_efficiency = cfg.INTERVENTION_EFFECTIVENESS.get(
                f'mask_{self.mask_type}', 
                {'virus_inhalation': 0.3}
            )['virus_inhalation']
            mask_protection = 1 - mask_efficiency
        
        inhaled_dose = self.inhalation_rate * effective_concentration * mask_protection * self.model.dt
        infection_probability = 1 - np.exp(-inhaled_dose)
        if self.vaccinated:
            vaccine_efficacy = 0.9
            infection_probability *= (1 - vaccine_efficacy)
        
        self.exposure_history.append({
            'time': self.model.schedule.time,
            'concentration': virus_concentration,
            'dose': inhaled_dose,
            'probability': infection_probability,
            'location': self.pos,
            'puf_factor': puf_factor,
            'mask_protection': mask_protection
        })
        
        self.risk_perception = min(1.0, self.risk_perception + infection_probability * self.learning_rate)
        
        if np.random.random() < infection_probability:
            self.infected = True
            self.infection_start_time = self.model.schedule.time
            self.viral_load = 1.0
            self.symptoms = np.random.random() < 0.7
            self.emission_rates = self._calculate_emission_rates()
    
    def update_infection(self):
        """Atualiza estado de infecção."""
        if not self.infected:
            return
        
        current_time = self.model.schedule.time
        if self.infection_start_time is None:
             self.infection_start_time = current_time
        
        infection_duration = current_time - self.infection_start_time
        peak_time = np.random.uniform(3, 5) * 86400
        duration = np.random.uniform(10, 14) * 86400
        
        if infection_duration < peak_time:
            self.viral_load = infection_duration / peak_time
        elif infection_duration < duration:
            decline_phase = (infection_duration - peak_time) / (duration - peak_time)
            self.viral_load = 1.0 - decline_phase * 0.8
        else:
            self.infected = False
            self.viral_load = 0.0
            if 'virus' in self.emission_rates:
                self.emission_rates['virus'] = 0.0
        
        if 'virus' in self.emission_rates:
            self.emission_rates['virus'] *= self.viral_load
    
    def adapt_behavior(self):
        """Adapta comportamento baseado no conforto e percepção de risco."""
        if self.pos is None:
            return
        
        local_data = self.model.physics.get_concentrations_at(self.pos[0], self.pos[1])
        if not local_data:
            return
        
        temp_diff = abs(local_data['temperature_c'] - self.model.scenario.temperature_setpoint)
        temp_discomfort = min(1.0, temp_diff / 5.0)
        
        hum_diff = abs(local_data['humidity_percent'] - self.model.scenario.humidity_setpoint)
        hum_discomfort = min(1.0, hum_diff / 30.0)
        
        co2_level = local_data['co2_ppm']
        co2_discomfort = min(1.0, max(0, co2_level - 600) / 1000)
        
        air_velocity = local_data['air_velocity_ms']
        if air_velocity < 0.05:
            velocity_discomfort = 0.3
        elif air_velocity > 0.8:
            velocity_discomfort = min(1.0, (air_velocity - 0.8) / 0.5)
        else:
            velocity_discomfort = 0.0
        
        air_age = local_data.get('air_age_minutes', 0)
        age_discomfort = min(1.0, air_age / 60.0)
        
        discomfort = (0.25 * temp_discomfort + 
                     0.15 * hum_discomfort + 
                     0.25 * co2_discomfort + 
                     0.10 * velocity_discomfort + 
                     0.25 * age_discomfort)
        
        self.comfort_level = max(0.0, 1.0 - discomfort)
        self.comfort_history.append({'time': self.model.schedule.time, 'comfort': self.comfort_level})
        
        if len(self.comfort_history) > 10:
            recent_comfort = np.mean([ch['comfort'] for ch in self.comfort_history[-10:]])
            if recent_comfort < 0.6:
                self.social_distance_preference = min(1.0, self.social_distance_preference + 0.1)
                self.preferred_social_distance = min(3.0, self.preferred_social_distance + 0.2)
        
        if discomfort > 0.6 and self.rule_compliance > 0.5:
            if np.random.random() < 0.4:
                self.target_pos = self._find_better_location()
        
        if self.risk_perception > self.adaptation_threshold:
            if not self.mask_wearing and np.random.random() < 0.6:
                self.mask_wearing = True
                self.mask_type = self._assign_mask_type()
                self.emission_rates = self._calculate_emission_rates()
            
            nearby_agents = self.model.grid.get_neighbors(self.pos, moore=True, radius=3)
            if len(nearby_agents) > 4:
                if np.random.random() < 0.5:
                    self.target_pos = self._find_less_crowded_location()
            
            self.social_distance_preference = min(1.0, self.social_distance_preference + 0.1)
    
    def _find_better_location(self) -> Optional[Tuple[int, int]]:
        """Encontra local com melhor qualidade do ar, evitando obstáculos."""
        if self.pos is None:
            return None
        
        x, y = self.pos
        
        for search_radius in [5, 10, 15]:
            best_score = -float('inf')
            best_pos = None
            
            for dx in range(-search_radius, search_radius + 1):
                for dy in range(-search_radius, search_radius + 1):
                    nx, ny = x + dx, y + dy
                    
                    if 0 <= nx < self.model.physics.cells_x and 0 <= ny < self.model.physics.cells_y:
                        data = self.model.physics.get_concentrations_at(nx, ny)
                        if data:
                            temp_score = 100 - abs(data['temperature_c'] - self.model.scenario.temperature_setpoint) * 10
                            co2_score = 100 - max(0, data['co2_ppm'] - 400) / 10
                            hum_score = 100 - abs(data['humidity_percent'] - self.model.scenario.humidity_setpoint) * 2
                            air_age_score = 100 - min(100, data.get('air_age_minutes', 0))
                            velocity_score = 100 if 0.1 <= data['air_velocity_ms'] <= 0.5 else 50
                            virus_score = 100 - min(100, data.get('virus_quanta_m3', 0) * 1e3)
                            
                            score = (temp_score * 0.25 + co2_score * 0.25 + 
                                    hum_score * 0.15 + air_age_score * 0.15 + 
                                    velocity_score * 0.10 + virus_score * 0.10)
                            if (score > best_score and 
                                self.model.grid.is_cell_empty((nx, ny)) and 
                                self.model.physics.is_walkable(nx, ny)):
                                best_score = score
                                best_pos = (nx, ny)
            
            if best_pos:
                return best_pos
        
        return None
    
    def _find_less_crowded_location(self) -> Optional[Tuple[int, int]]:
        """Encontra local com menos agentes, evitando obstáculos."""
        if self.pos is None:
            return None
        
        x, y = self.pos
        
        for radius in range(1, 8):
            candidate_positions = []
            
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < self.model.physics.cells_x and 
                        0 <= ny < self.model.physics.cells_y and
                        self.model.grid.is_cell_empty((nx, ny)) and
                        self.model.physics.is_walkable(nx, ny)):
                        
                        nearby = self.model.grid.get_neighbors((nx, ny), moore=True, radius=2)
                        agent_count = sum(1 for a in nearby if isinstance(a, HumanAgent))
                        
                        candidate_positions.append({
                            'pos': (nx, ny),
                            'agent_count': agent_count,
                            'distance': abs(dx) + abs(dy)
                        })
            
            if candidate_positions:
                candidate_positions.sort(key=lambda cp: (cp['agent_count'], cp['distance']))
                return candidate_positions[0]['pos']
        
        return None
    
    def step(self):
        """Executa um passo do agente."""
        self.update_activity()
        self.update_infection()
        self.adapt_behavior()
        
        new_pos = self.decide_movement()
        if new_pos and self.model.grid.is_cell_empty(new_pos):
            self.model.grid.move_agent(self, new_pos)
            self.pos = new_pos
        
        self.emit_pollutants()
        
        if not self.infected and not self.vaccinated and self.pos:
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
        
        emissions = []
        for species, rate in self.emission_rates.items():
            if rate > 0:
                emissions.append({
                    'x': self.pos[0],
                    'y': self.pos[1],
                    'species': species,
                    'amount': rate * self.model.dt
                })
        
        if self.metabolic_heat > 0:
            emissions.append({
                'x': self.pos[0],
                'y': self.pos[1],
                'species': 'heat',
                'amount': self.metabolic_heat * self.model.dt
            })
        
        if self.moisture_production > 0:
            emissions.append({
                'x': self.pos[0],
                'y': self.pos[1],
                'species': 'moisture',
                'amount': self.moisture_production * self.model.dt
            })
        
        if hasattr(self.model, 'add_agent_emissions'):
            self.model.add_agent_emissions(emissions)
    
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
            'inhalation_rate_m3_s': self.inhalation_rate,
            'comfort_level': self.comfort_level,
            'risk_perception': self.risk_perception,
            'rule_compliance': self.rule_compliance,
            'social_distance_preference': self.social_distance_preference,
            'preferred_social_distance_m': self.preferred_social_distance,
            'position': self.pos,
            'local_concentrations': local_data,
            'emission_rates': self.emission_rates,
            'total_exposure': sum(exp['dose'] for exp in self.exposure_history) if self.exposure_history else 0,
            'exposure_count': len(self.exposure_history),
            'average_comfort': np.mean([ch['comfort'] for ch in self.comfort_history]) if self.comfort_history else 0.0,
            'movement_speed_ms': self.movement_speed
        }
        
        return data


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
        
        if agent.infected:
            priority += 1000
            priority += agent.viral_load * 500
        
        if hasattr(agent, 'emission_rates') and 'virus' in agent.emission_rates:
            priority += agent.emission_rates['virus'] * 1000
        
        high_risk_activities = [
            cfg.AgentActivity.COUGHING,
            cfg.AgentActivity.SNEEZING,
            cfg.AgentActivity.SINGING,
            cfg.AgentActivity.TALKING,
            cfg.AgentActivity.EXERCISING_INTENSE
        ]
        
        if agent.current_activity in high_risk_activities:
            priority += 100
        
        if agent.comfort_level < 0.6:
            priority += 50
        
        if agent.risk_perception > 0.7:
            priority += 30
        
        if agent.moving:
            priority += 20
        
        return -priority


class LearningAgent(HumanAgent):
    """
    Agente com capacidade de aprendizado por reforço.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Parâmetros de aprendizado
        self.q_table = {}
        self.epsilon = 0.1
        self.alpha = 0.1
        self.gamma = 0.9
        
        # Histórico de recompensas
        self.reward_history = []
        self.action_history = []
    
    def adapt_behavior(self):
        """Adapta comportamento usando aprendizado por reforço."""
        super().adapt_behavior()
        
        state = self._get_state()
        
        if np.random.random() < self.epsilon:
            action = self._choose_random_action()
        else:
            action = self._choose_best_action(state)
        
        reward = self._execute_action(action)
        self._update_q_table(state, action, reward)
        self.reward_history.append(reward)
        self.action_history.append(action)
    
    def _get_state(self) -> str:
        """Codifica o estado atual."""
        state_parts = []
        state_parts.append('I' if self.infected else 'H')
        state_parts.append('M' if self.mask_wearing else 'N')
        
        if self.risk_perception < 0.3:
            state_parts.append('LR')
        elif self.risk_perception < 0.7:
            state_parts.append('MR')
        else:
            state_parts.append('HR')
        
        if self.comfort_level > 0.7:
            state_parts.append('HC')
        elif self.comfort_level > 0.4:
            state_parts.append('MC')
        else:
            state_parts.append('LC')
        
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
        
        reward -= (1 - self.comfort_level) * 0.1
        
        if self.exposure_history:
            recent_exposure = self.exposure_history[-1]['dose'] if self.exposure_history else 0
            reward -= recent_exposure * 100
        
        return reward
    
    def _update_q_table(self, state: str, action: str, reward: float):
        """Atualiza Q-table usando Q-learning."""
        if state not in self.q_table:
            self.q_table[state] = {}
        
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        
        next_state = self._get_state()
        next_max = max(self.q_table.get(next_state, {}).values(), default=0.0)
        
        self.q_table[state][action] = (
            (1 - self.alpha) * self.q_table[state][action] +
            self.alpha * (reward + self.gamma * next_max)
        )
