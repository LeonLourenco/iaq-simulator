"""
Orquestrador da Simulação (IAQModel).

Responsabilidade:
- Integrar os módulos de Configuração, Ambiente, Física e Agentes.
- Gerenciar o loop principal de tempo (Time Stepping).
- Sincronizar o ciclo epidemiológico: Emissão -> Transporte -> Inalação -> Infecção.
- Coletar métricas globais para análise.
- Gerenciar a distribuição de perfis comportamentais.

Arquitetura:
- Herda de mesa.Model.
- Possui instâncias de Environment, PhysicsEngine e Scheduler.
"""

import logging
import random
import pandas as pd
from typing import Dict, List, Any, Tuple

from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

from config import ScenarioConfig, AgentState
from environment import Environment
from physics import PhysicsEngine
from agents import HumanAgent

# Logger setup
logger = logging.getLogger(__name__)

class IAQModel(Model):
    """
    Modelo Integrado de Qualidade do Ar Interno e Epidemiologia.
    Simula a propagação de patógenos em ambientes fechados com ocupantes móveis
    e perfis comportamentais heterogêneos.
    """

    def __init__(self, scenario_config: ScenarioConfig):
        """
        Inicializa o modelo com base no cenário fornecido.

        Args:
            scenario_config: Objeto de configuração carregado (JSON validado).
        """
        super().__init__()
        self.config = scenario_config
        self.time = 0.0  # Tempo simulado em segundos
        self.step_count = 0
        
        # 0. Registro de Rastreamento de Contatos (Novo)
        # Lista de dicts: {time, victim_id, location, dose}
        self.infection_log: List[Dict[str, Any]] = []
        
        # 1. Inicializa Ambiente (Fachada de Geometria/Obstáculos)
        self.environment = Environment(scenario_config)
        
        # 2. Inicializa Grid do Mesa (Espaço Discreto para Agentes)
        # O Environment calcula as dimensões em células
        width = self.environment.width_cells
        height = self.environment.height_cells
        self.grid = MultiGrid(width, height, torus=False)
        
        # 3. Inicializa Motor de Física (CFD Simplificado)
        # Passa a máscara de obstáculos gerada pelo Environment
        self.physics = PhysicsEngine(
            physics_config=scenario_config.physics,
            ventilation_config=scenario_config.ventilation,
            obstacle_mask=self.environment.get_obstacle_mask()
        )
        
        # 4. Inicializa Agentes e Scheduler
        self.schedule = RandomActivation(self)
        self._initialize_agents()
        
        # 5. Coletores de Dados
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Datacollector do Mesa (opcional, mas útil para compatibilidade)
        self.datacollector = DataCollector(
            model_reporters={
                "S": lambda m: m.get_state_counts()["SUSCEPTIBLE"],
                "I": lambda m: m.get_state_counts()["INFECTED"],
                "R": lambda m: m.get_state_counts()["RECOVERED"],
                "Max_Virus_Conc": lambda m: m.physics.virus_grid.max()
            }
        )
        
        logger.info(f"Modelo inicializado: {self.config.name}. Agentes: {self.schedule.get_agent_count()}")

    def _initialize_agents(self):
        """
        Cria e posiciona a população inicial.
        Realiza o sorteio dos perfis comportamentais (Strategy) baseado na distribuição do config.
        """
        total_agents = self.config.agents.total_occupants
        initial_infected = self.config.agents.initial_infected
        
        # --- Lógica de Distribuição de Perfis (NOVO) ---
        dist = self.config.agents.profile_distribution
        profiles = []
        
        # Gera lista ponderada de perfis
        for profile_name, percentage in dist.items():
            count = int(total_agents * percentage)
            profiles.extend([profile_name] * count)
            
        # Preenche arredondamentos com o perfil majoritário ou padrão
        default_profile = list(dist.keys())[0] if dist else "student_focused"
        while len(profiles) < total_agents:
            profiles.append(default_profile)
            
        # Embaralha para que os perfis não fiquem agrupados por ID
        random.shuffle(profiles)
        
        # --- Criação dos Agentes ---
        infected_ids = set(range(initial_infected))
        
        for i in range(total_agents):
            state = AgentState.INFECTED if i in infected_ids else AgentState.SUSCEPTIBLE
            
            # Tenta encontrar uma posição válida (sem parede, sem gente)
            pos = self._find_valid_spawn_position()
            if pos is None:
                logger.warning(f"Não foi possível posicionar agente {i}. Grid cheio?")
                continue
            
            # Instancia o agente injetando o perfil sorteado
            agent = HumanAgent(
                unique_id=i,
                model=self,
                pos=pos,
                agent_config=self.config.agents,
                initial_state=state,
                profile_type=profiles[i] # Injeção do Behavior
            )
            
            self.schedule.add(agent)
            self.grid.place_agent(agent, pos)

    def _find_valid_spawn_position(self) -> Tuple[int, int]:
        """Encontra uma célula vazia e válida (não parede) aleatória."""
        for _ in range(100):
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            pos = (x, y)
            
            if self.environment.is_valid_move(pos) and self.grid.is_cell_empty(pos):
                return pos
        return None

    def step(self):
        """
        Executa um passo completo da simulação.
        Dt reduzido para 10s para garantir estabilidade mesmo em ACH alto (Hospitalar).
        """
        dt_seconds = 10.0
        
        # 1. Coletar Emissões (Agentes -> Física)
        for agent in self.schedule.agents:
            if isinstance(agent, HumanAgent):
                # Emissão Viral
                viral_emission = agent.calculate_emission_quanta_per_s()
                if viral_emission > 0:
                    self.physics.add_source(
                        agent.pos[0], agent.pos[1], 
                        viral_emission * 3600.0, 
                        dt_seconds, 
                        field_type='virus'
                    )
                
                # Emissão de CO2 (Respiração)
                co2_emission_m3_h = agent.respiration_rate * 0.04 
                self.physics.add_source(
                    agent.pos[0], agent.pos[1],
                    co2_emission_m3_h,
                    dt_seconds,
                    field_type='co2'
                )

        # 2. Avançar Física (Transporte e Difusão)
        self.physics.step(dt_seconds)

        # 3. Exposição e Inalação (Física -> Agentes)
        for agent in self.schedule.agents:
            if isinstance(agent, HumanAgent):
                x, y = agent.pos
                concentration = self.physics.get_concentration_at(x, y)
                agent.inhale(concentration, dt_seconds)

        # 4. Atualizar Comportamento dos Agentes (Movimento, Estado)
        # O agente agora delega a decisão para seu BehaviorStrategy
        self.schedule.step()
        
        # 5. Coleta de Dados e Avanço de Tempo
        self.time += dt_seconds
        self.step_count += 1
        self._update_metrics()
        self.datacollector.collect(self)
        
        # Critério de Parada
        if self.time >= self.config.duration_hours * 3600:
            self.running = False

    # ========================================================================
    # NOVOS MÉTODOS DE SUPORTE (INTERAÇÃO E LOGS)
    # ========================================================================

    def is_break_time(self) -> bool:
        """
        Define se é hora do intervalo/socialização.
        Usado pelos Behaviors (ex: StudentSocialBehavior) para decidir se levantam.
        
        Regra: Últimos 10 minutos de cada hora simulada.
        """
        minutes_in_hour = (self.time / 60.0) % 60
        return minutes_in_hour > 50

    def log_infection(self, agent: HumanAgent):
        """
        Registra detalhadamente um evento de infecção para o Contact Tracing.
        Chamado pelo agente quando o estado muda para INFECTED.
        """
        log_entry = {
            "time_h": self.time / 3600.0,
            "victim_id": agent.unique_id,
            "location": agent.pos,
            "dose_at_infection": agent.accumulated_dose,
            "state": "INFECTED"
        }
        self.infection_log.append(log_entry)
        logger.info(f"☣️ INFECÇÃO REGISTRADA: Agente {agent.unique_id} em {agent.pos} (T={log_entry['time_h']:.2f}h)")

    # ========================================================================
    # MÉTODOS DE MÉTRICAS (MANTIDOS)
    # ========================================================================

    def _update_metrics(self):
        """Calcula estatísticas do passo atual e salva no histórico."""
        counts = self.get_state_counts()
        
        total_dose = sum(a.accumulated_dose for a in self.schedule.agents if isinstance(a, HumanAgent))
        max_virus = self.physics.virus_grid.max()
        avg_co2 = self.physics.co2_grid.mean()
        
        metric = {
            "step": self.step_count,
            "time_hours": self.time / 3600.0,
            "S": counts["SUSCEPTIBLE"],
            "I": counts["INFECTED"],
            "R": counts["RECOVERED"],
            "total_dose_accumulated": total_dose,
            "max_virus_quanta_m3": max_virus,
            "avg_co2_ppm": avg_co2
        }
        self.metrics_history.append(metric)

    def get_state_counts(self) -> Dict[str, int]:
        """Retorna contagem de agentes por estado epidemiológico."""
        s = 0
        i = 0
        r = 0
        for a in self.schedule.agents:
            if a.state == AgentState.SUSCEPTIBLE: s += 1
            elif a.state == AgentState.INFECTED: i += 1
            elif a.state == AgentState.RECOVERED: r += 1
            
        return {"SUSCEPTIBLE": s, "INFECTED": i, "RECOVERED": r}

    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Exporta o histórico de métricas como DataFrame do Pandas."""
        return pd.DataFrame(self.metrics_history)
    
    def get_final_metrics(self) -> Dict[str, Any]:
        """Retorna um resumo consolidado do final da simulação."""
        if not self.metrics_history:
            return {}
            
        last = self.metrics_history[-1]
        initial_pop = self.config.agents.total_occupants
        initial_infected = self.config.agents.initial_infected
        
        # R Efetivo (Proxy: Novos Casos / Casos Iniciais)
        total_cases = (last['I'] + last['R'])
        new_cases = total_cases - initial_infected
        r_eff = new_cases / initial_infected if initial_infected > 0 else 0
        
        return {
            "attack_rate": new_cases / (initial_pop - initial_infected) if (initial_pop - initial_infected) > 0 else 0,
            "r_effective": r_eff,
            "peak_infected": max(m['I'] for m in self.metrics_history),
            "final_recovered": last['R'],
            "final_susceptible": last['S']
        }