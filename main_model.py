"""
MODELO DE SIMULAÇÃO PRINCIPAL
Integra física, agentes e cenários em uma arquitetura unificada.
"""

import numpy as np
import json
from mesa import Model
from mesa.space import MultiGrid
from typing import Dict, List, Optional, Any, Tuple
import config_final as cfg
from unified_physics import UnifiedPhysicsEngine
from advanced_agents import HumanAgent, AdaptiveScheduler, LearningAgent

class IAQSimulationModel(Model):
    """
    Modelo de simulação principal que integra física (CFD simplificado), 
    agentes comportamentais e lógica de controle predial.
    """
    
    def __init__(self, 
                 scenario: cfg.BuildingScenario,
                 physics_config: cfg.PhysicsConfig,
                 simulation_duration_hours: float = 8.0,
                 real_time_factor: float = 1.0,
                 use_learning_agents: bool = False):
        super().__init__()
        
        # Configurações Gerais
        self.scenario = scenario
        self.physics_config = physics_config
        self.simulation_duration = simulation_duration_hours * 3600  # segundos
        self.real_time_factor = real_time_factor
        self.use_learning_agents = use_learning_agents
        
        # Controle de Tempo
        self.schedule = AdaptiveScheduler(self)
        self.time = 0.0
        self.dt = self._calculate_optimal_dt()
        
        # Motor Físico
        self.physics = UnifiedPhysicsEngine(scenario, physics_config)
        
        # Espaço (Grid)
        self.grid = MultiGrid(
            width=self.physics.cells_x,
            height=self.physics.cells_y,
            torus=False
        )
        
        # Agentes e Listas
        self.simulation_agents: List[Any] = []
        
        self.agent_config = scenario.agent_config
        self.current_agent_emissions = []
        self._initialize_agents()
        
        # Armazenamento de Dados
        self.simulation_data = {
            'time': [],
            'zone_stats': [],
            'agent_stats': [],
            'risk_metrics': [],
            'energy_consumption': [],
            'intervention_effects': []
        }
        
        # Estado do Modelo
        self.running = True
        self.paused = False
        self.interventions_active = {}
        self.optimization_enabled = False
        self.optimization_targets = {}
        
        # Métricas Iniciais
        self.current_metrics = self._calculate_initial_metrics()

    def _calculate_optimal_dt(self) -> float:
        """
        Calcula passo de tempo (dt) para garantir estabilidade numérica (CFL).
        """
        # Condição CFL (Courant-Friedrichs-Lewy) para advecção
        dt_cfl = 0.5 * self.physics_config.cell_size / 1.0  # Assumindo vel. máx ~1 m/s
        
        # Condição de estabilidade para difusão
        # Soma das difusividades moleculares e turbulentas (pior caso)
        max_diffusion = (self.physics_config.molecular_diffusion_co2 + 
                         self.physics_config.turbulent_diffusion_high_vent)
        
        dt_diffusion = 0.25 * (self.physics_config.cell_size ** 2) / max_diffusion
        
        # Escolhe o menor passo respeitando o máximo configurado
        dt = min(dt_cfl, dt_diffusion, self.physics_config.dt_max)
        
        # Aplica fator de segurança
        return dt * self.physics_config.stability_safety_factor

    def _initialize_agents(self):
        """
        Inicializa e distribui agentes nas zonas baseando-se na densidade de ocupação.
        Garante que agentes não nasçam em obstáculos.
        """
        total_agents_created = 0
        total_occupancy_weight = 0.0
        
        # Pré-cálculo para distribuição proporcional
        zone_params_list = []
        for zone in self.scenario.zones:
            params = cfg.calculate_zone_parameters(
                zone, 
                self.scenario.total_width, 
                self.scenario.total_height,
                self.scenario.floor_height
            )
            weight = zone.occupancy_density * params['area']
            total_occupancy_weight += weight
            zone_params_list.append((params, weight))

        # Criação dos agentes por zona
        for zone_idx, zone in enumerate(self.scenario.zones):
            params, weight = zone_params_list[zone_idx]
            
            # Cálculo proporcional de agentes para esta zona
            if total_occupancy_weight > 0:
                target_count = int(self.scenario.total_occupants * (weight / total_occupancy_weight))
            else:
                target_count = 0
                
            # Limita pela capacidade máxima física da zona
            zone_agent_count = min(target_count, params['max_occupants'])
            
            # Encontra células válidas nesta zona
            zone_cells = np.where(self.physics.zone_map == zone_idx + 1)
            valid_cells_count = len(zone_cells[0])
            
            if valid_cells_count == 0:
                continue

            for i in range(zone_agent_count):
                agent_id = total_agents_created + i
                
                # Definição de infecção inicial
                is_infected = (i < zone_agent_count * self.scenario.initial_infected_ratio)
                
                # Seleção de classe de agente
                AgentClass = LearningAgent if self.use_learning_agents else HumanAgent
                
                agent = AgentClass(
                    unique_id=agent_id,
                    model=self,
                    zone_config=zone,
                    agent_config=self.agent_config,
                    initial_infected=is_infected
                )
                
                # Posicionamento aleatório seguro
                placed = False
                attempts = 0
                
                # Tenta 100 vezes encontrar um lugar que não seja parede/móvel
                while not placed and attempts < 100:
                    rnd_idx = np.random.randint(valid_cells_count)
                    x = zone_cells[1][rnd_idx]
                    y = zone_cells[0][rnd_idx]
                    
                    # Verifica se é andável (não parede)
                    if self.grid.is_cell_empty((x, y)) and self.physics.is_walkable(x, y):
                        self.grid.place_agent(agent, (x, y))
                        agent.pos = (x, y)
                        placed = True
                    attempts += 1
                
                # Se não achou lugar ideal, tenta forçar em qualquer lugar válido da zona (último recurso)
                if not placed:
                    for _ in range(20): # Tenta mais algumas vezes sem exigir célula vazia (sobreposição permitida em emergência)
                        rnd_idx = np.random.randint(valid_cells_count)
                        x = zone_cells[1][rnd_idx]
                        y = zone_cells[0][rnd_idx]
                        if self.physics.is_walkable(x, y):
                            self.grid.place_agent(agent, (x, y))
                            agent.pos = (x, y)
                            placed = True
                            break

                if placed:
                    self.schedule.add(agent)
                    self.simulation_agents.append(agent)
            
            total_agents_created += zone_agent_count
        
        print(f"Inicialização concluída: {total_agents_created} agentes criados.")

    def _calculate_initial_metrics(self) -> Dict[str, Any]:
        """Retorna estrutura base de métricas zerada."""
        return {
            'total_agents': len(self.simulation_agents),
            'infected_agents': sum(1 for a in self.simulation_agents if a.infected),
            'average_co2': 400.0,
            'average_hcho': 10.0,
            'average_virus': 0.0,
            'average_temperature': self.scenario.temperature_setpoint,
            'average_humidity': self.scenario.humidity_setpoint,
            'ventilation_efficiency': 1.0,
            'infection_risk': 0.0,
            'energy_consumption': 0.0,
            'comfort_index': 0.8
        }

    def add_agent_emissions(self, emissions: List[Dict]):
        """Callback usado pelos agentes para registrar emissões no ambiente."""
        self.current_agent_emissions.extend(emissions)

    def apply_interventions(self, intervention_type: str, parameters: Dict):
        """
        Aplica intervenções dinâmicas na simulação.
        Apenas intervenções funcionais são implementadas.
        """
        self.interventions_active[intervention_type] = {
            'type': intervention_type,
            'parameters': parameters,
            'start_time': self.time
        }
        
        if intervention_type == 'increase_ventilation':
            # Aumenta ACH em todas as zonas
            factor = parameters.get('factor', 1.5)
            for zone in self.scenario.zones:
                zone.target_ach *= factor
        
        elif intervention_type == 'mask_mandate':
            # Força uso de máscara baseado na conformidade
            compliance = parameters.get('compliance', 0.8)
            for agent in self.simulation_agents:
                if np.random.random() < compliance:
                    agent.mask_wearing = True
                    agent.mask_type = agent._assign_mask_type() if agent.mask_wearing else None
                    agent.emission_rates = agent._calculate_emission_rates()
        
        elif intervention_type == 'reduce_occupancy':
            # Remove agentes aleatoriamente
            if not self.simulation_agents:
                return
                
            reduction_ratio = parameters.get('reduction', 0.3)
            remove_count = int(len(self.simulation_agents) * reduction_ratio)
            
            # Escolhe aleatoriamente para remover
            agents_to_remove = np.random.choice(self.simulation_agents, remove_count, replace=False)
            
            for agent in agents_to_remove:
                if agent.pos:
                    self.grid.remove_agent(agent)
                self.schedule.remove(agent)
            
            # Atualiza lista principal
            ids_to_remove = set(a.unique_id for a in agents_to_remove)
            self.simulation_agents = [a for a in self.simulation_agents if a.unique_id not in ids_to_remove]
        
        print(f"Intervenção aplicada: {intervention_type}")

    def calculate_risk_metrics(self) -> Dict[str, float]:
        """Calcula métricas de risco epidemiológico (Wells-Riley)."""
        total_agents = len(self.simulation_agents)
        if total_agents == 0:
            return {'infection_risk': 0.0, 'r_effective': 0.0, 'infected_count': 0, 
                    'infection_rate': 0.0, 'total_viral_load': 0.0, 'high_risk_zones': 0}

        infected_count = sum(1 for a in self.simulation_agents if a.infected)
        
        # Carga viral total no ambiente (Quanta)
        zone_stats = self.physics.get_zone_statistics()
        total_virus = sum(stats['concentrations']['virus_mean'] * stats['volume_m3'] 
                          for stats in zone_stats.values())
        
        # Probabilidade média de infecção (Simplificação Wells-Riley)
        avg_inhalation = 0.5e-3  # m³/s (respiração média)
        exposure_time = max(self.time, 1.0) # evita zero
        
        # P = 1 - exp(-I * q * t / Q) -> aqui aproximado pela concentração média
        # Risco acumulado aproximado
        infection_risk = 1.0 - np.exp(-total_virus * avg_inhalation * exposure_time / max(1.0, len(zone_stats)))
        
        # R0 efetivo (Estimativa instantânea)
        prev_infected = self.current_metrics.get('infected_agents', 0)
        new_infections = max(0, infected_count - prev_infected)
        
        if prev_infected > 0 and self.time > 3600:
            r_effective = new_infections / prev_infected
        else:
            r_effective = 0.0
            
        high_risk_zones = sum(1 for s in zone_stats.values() if s['concentrations']['virus_mean'] > 1e-3)

        return {
            'infection_risk': float(min(infection_risk, 1.0)),
            'r_effective': float(r_effective),
            'infected_count': int(infected_count),
            'infection_rate': float(infected_count / total_agents),
            'total_viral_load': float(total_virus),
            'high_risk_zones': int(high_risk_zones)
        }

    def calculate_energy_metrics(self) -> Dict[str, float]:
        """Wrapper para cálculo de energia do motor físico."""
        return self.physics._calculate_energy_consumption()

    def optimize_parameters(self):
        """
        Lógica simples de controle preditivo/otimização.
        Ajusta ventilação baseada em setpoints de CO2.
        """
        if not self.optimization_enabled:
            return
        
        current_co2 = self.current_metrics['average_co2']
        targets = self.optimization_targets
        
        # Controle de Ventilação por Demanda (DCV Simples)
        if 'co2_target' in targets:
            target_co2 = targets['co2_target']
            
            # Histerese simples
            if current_co2 > target_co2 * 1.1:
                # Aumenta ventilação
                for zone in self.scenario.zones:
                    zone.target_ach = min(zone.target_ach * 1.05, 20.0) # Aumento gradual
            elif current_co2 < target_co2 * 0.8:
                # Economia de energia
                for zone in self.scenario.zones:
                    zone.target_ach = max(zone.target_ach * 0.95, 0.5) # Redução gradual

        # Controle de Ocupação por Risco
        if 'infection_risk_target' in targets:
            current_risk = self.current_metrics['infection_risk']
            target_risk = targets['infection_risk_target']
            
            if current_risk > target_risk:
                # Aciona redução de ocupação se risco superar limite
                self.apply_interventions('reduce_occupancy', {'reduction': 0.1})

    def step(self):
        """Avança um passo discreto na simulação."""
        if self.paused:
            return
        
        # Limpeza de buffers
        self.current_agent_emissions = []
        
        # 1. Comportamento dos Agentes
        self.schedule.step()
        
        # 2. Integração Física
        # Prepara dados agregados para o solver físico (otimização)
        # Filtra apenas agentes ativos
        active_agents = [a for a in self.simulation_agents if a.pos is not None]
        
        agent_data = {
            'emissions': self.current_agent_emissions,
            'positions': [a.pos for a in active_agents],
            'activities': [a.current_activity.value for a in active_agents]
        }
        
        # Passo do Solver Físico
        self.physics.step(self.dt, self.time, agent_data)
        
        # 3. Controle e Métricas
        self.optimize_parameters()
        self._update_metrics()
        
        # 4. Avanço do Tempo
        self.time += self.dt
        if self.time >= self.simulation_duration:
            self.running = False
        
        # 5. Coleta de Dados (Amostragem a cada 5 min simulados)
        self._collect_simulation_data()

    def _update_metrics(self):
        """Atualiza o dicionário de métricas em tempo real."""
        zone_stats = self.physics.get_zone_statistics()
        risk_metrics = self.calculate_risk_metrics()
        energy_metrics = self.calculate_energy_metrics()
        
        # Médias ponderadas ou simples das zonas
        if zone_stats:
            avg_co2 = np.mean([s['concentrations']['co2_ppm_mean'] for s in zone_stats.values()])
            avg_hcho = np.mean([s['concentrations']['hcho_ppb_mean'] for s in zone_stats.values()])
            avg_temp = np.mean([s['concentrations']['temperature_c_mean'] for s in zone_stats.values()])
            avg_hum = np.mean([s['concentrations']['humidity_percent_mean'] for s in zone_stats.values()])
        else:
            avg_co2, avg_hcho = 400.0, 10.0
            avg_temp, avg_hum = self.scenario.temperature_setpoint, self.scenario.humidity_setpoint
            
        # Conforto
        if self.simulation_agents:
            avg_comfort = np.mean([a.comfort_level for a in self.simulation_agents if hasattr(a, 'comfort_level')])
        else:
            avg_comfort = 1.0

        # Atualização do estado
        self.current_metrics.update({
            'time': self.time,
            'infected_agents': risk_metrics['infected_count'],
            'infection_rate': risk_metrics['infection_rate'],
            'r_effective': risk_metrics['r_effective'],
            'average_co2': float(avg_co2),
            'average_hcho': float(avg_hcho),
            'average_temperature': float(avg_temp),
            'average_humidity': float(avg_hum),
            'infection_risk': risk_metrics['infection_risk'],
            'energy_consumption': energy_metrics['total_energy_kwh'],
            'comfort_index': float(avg_comfort),
            'ventilation_efficiency': float(avg_co2 / max(1, self.scenario.co2_setpoint))
        })

    def _collect_simulation_data(self):
        """Salva snapshot dos dados para gráficos históricos."""
        # Salva a cada 300 segundos (5 min) ou se for o último passo
        if self.time % 300 < self.dt or not self.running:
            self.simulation_data['time'].append(self.time)
            self.simulation_data['zone_stats'].append(self.physics.get_zone_statistics())
            
            # Estatísticas agregadas dos agentes
            if self.simulation_agents:
                stats = {
                    'total': len(self.simulation_agents),
                    'infected': sum(1 for a in self.simulation_agents if a.infected),
                    'mask_wearing': sum(1 for a in self.simulation_agents if a.mask_wearing),
                    'avg_comfort': self.current_metrics['comfort_index']
                }
            else:
                stats = {'total': 0, 'infected': 0, 'mask_wearing': 0, 'avg_comfort': 0}
                
            self.simulation_data['agent_stats'].append(stats)
            self.simulation_data['risk_metrics'].append(self.calculate_risk_metrics())
            self.simulation_data['energy_consumption'].append(self.calculate_energy_metrics())

    def get_visualization_data(self) -> Dict[str, Any]:
        """Prepara payload leve para o frontend (Streamlit)."""
        active_agents = [a for a in self.simulation_agents if a.pos is not None]
        
        return {
            'physics': self.physics.get_visualization_data(),
            'agents': {
                'positions': [a.pos for a in active_agents],
                'infected': [a.infected for a in active_agents],
                'activities': [a.current_activity.value for a in active_agents],
                'mask_wearing': [a.mask_wearing for a in active_agents],
                'comfort_levels': [getattr(a, 'comfort_level', 0.0) for a in active_agents]
            },
            'metrics': self.current_metrics,
            'time': self.time,
            'running': self.running,
            'paused': self.paused
        }

    def get_simulation_summary(self) -> Dict[str, Any]:
        """Gera relatório final consolidado."""
        total_exposure = sum(sum(exp['dose'] for exp in a.exposure_history) 
                             for a in self.simulation_agents if hasattr(a, 'exposure_history'))
        
        peak_infected = 0
        if self.simulation_data['agent_stats']:
            peak_infected = max(s['infected'] for s in self.simulation_data['agent_stats'])

        return {
            'scenario': self.scenario.name,
            'duration_hours': self.time / 3600,
            'total_agents': len(self.simulation_agents),
            'total_infections': sum(1 for a in self.simulation_agents if a.infection_start_time is not None),
            'peak_infected': peak_infected,
            'total_exposure': total_exposure,
            'average_co2': self.current_metrics['average_co2'],
            'total_energy_kwh': self.current_metrics['energy_consumption'],
            'interventions_applied': list(self.interventions_active.keys())
        }

    def export_simulation_data(self, format: str = 'json') -> Any:
        """Exporta dados completos da simulação."""
        # Constrói dicionário mestre com tipos nativos (para evitar erro de serialização NumPy)
        data = {
            'metadata': {
                'scenario': self.scenario.name,
                'building_type': str(self.scenario.building_type.value),
                'simulation_time_seconds': self.time,
                'total_agents': len(self.simulation_agents)
            },
            'metrics_final': self.current_metrics,
            'history': {
                'time': [float(t) for t in self.simulation_data['time']],
                # Simplificação: exporta apenas métricas chave para manter JSON leve
                'risk': self.simulation_data['risk_metrics'],
                'energy': self.simulation_data['energy_consumption']
            }
        }
        
        if format == 'json':
            # Custom encoder para lidar com tipos numpy se sobrarem
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8,
                        np.uint16, np.uint32, np.uint64)):
                        return int(obj)
                    elif isinstance(obj, (np.float_, np.float16, np.float32, 
                        np.float64)):
                        return float(obj)
                    elif isinstance(obj, (np.ndarray,)):
                        return obj.tolist()
                    return json.JSONEncoder.default(self, obj)
            
            return json.dumps(data, indent=2, cls=NumpyEncoder)
        
        return data
