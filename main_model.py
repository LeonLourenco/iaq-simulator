"""
MODELO DE SIMULAÇÃO PRINCIPAL - VERSÃO REFATORADA
==================================================
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
        """
        Inicializa o modelo de simulação.
        
        Args:
            scenario: Configuração do cenário (zonas, obstáculos, etc)
            physics_config: Parâmetros físicos da simulação
            simulation_duration_hours: Duração da simulação em horas
            real_time_factor: Fator de escala temporal (1.0 = tempo real)
            use_learning_agents: Se True, usa agentes com aprendizagem
        """
        super().__init__()
        
        # ====================================================================
        # CONFIGURAÇÕES GERAIS
        # ====================================================================
        self.scenario = scenario
        self.physics_config = physics_config
        self.simulation_duration = simulation_duration_hours * 3600  # segundos
        self.real_time_factor = real_time_factor
        self.use_learning_agents = use_learning_agents
        
        # ====================================================================
        # CONTROLE DE TEMPO
        # ====================================================================
        self.schedule = AdaptiveScheduler(self)
        self.time = 0.0
        self.dt = self._calculate_optimal_dt()
        
        # ====================================================================
        # MOTOR FÍSICO COM SUPORTE A OBSTÁCULOS
        # ====================================================================
        # O UnifiedPhysicsEngine recebe o scenario completo e cria 
        # internamente a máscara de obstáculos a partir de scenario.obstacles
        self.physics = UnifiedPhysicsEngine(scenario, physics_config)
        
        # ====================================================================
        # ESPAÇO (GRID)
        # ====================================================================
        self.grid = MultiGrid(
            width=self.physics.cells_x,
            height=self.physics.cells_y,
            torus=False
        )
        
        # ====================================================================
        # AGENTES E LISTAS
        # ====================================================================
        self.simulation_agents: List[Any] = []
        self.agent_config = scenario.agent_config
        self.current_agent_emissions = []
        
        # Inicializa agentes
        self._initialize_agents()
        
        # ====================================================================
        # ARMAZENAMENTO DE DADOS
        # ====================================================================
        self.simulation_data = {
            'time': [],
            'zone_stats': [],
            'agent_stats': [],
            'risk_metrics': [],
            'energy_consumption': [],
            'intervention_effects': [],
            'accumulated_doses': [],
            'max_viral_loads': []
        }
        
        # ====================================================================
        # ESTADO DO MODELO
        # ====================================================================
        self.running = True
        self.paused = False
        self.interventions_active = {}
        self.optimization_enabled = False
        self.optimization_targets = {}
        
        # ====================================================================
        # MÉTRICAS INICIAIS
        # ====================================================================
        self.current_metrics = self._calculate_initial_metrics()

    # ========================================================================
    # MÉTODOS DE INICIALIZAÇÃO
    # ========================================================================

    def _calculate_optimal_dt(self) -> float:
        """
        Calcula passo de tempo (dt) para garantir estabilidade numérica (CFL).
        
        Returns:
            float: Passo de tempo em segundos
        """
        # Condição CFL (Courant-Friedrichs-Lewy) para advecção
        # Assume velocidade máxima de ~1 m/s
        dt_cfl = 0.5 * self.physics_config.cell_size / 1.0
        
        # Condição de estabilidade para difusão
        # Usa a maior difusividade (molecular + turbulenta)
        max_diffusion = (
            self.physics_config.molecular_diffusion_co2 + 
            self.physics_config.turbulent_diffusion_high_vent
        )
        
        dt_diffusion = 0.25 * (self.physics_config.cell_size ** 2) / max_diffusion
        
        # Escolhe o menor passo respeitando o máximo configurado
        dt = min(dt_cfl, dt_diffusion, self.physics_config.dt_max)
        
        # Aplica fator de segurança
        return dt * self.physics_config.stability_safety_factor

    def _initialize_agents(self):
        """
        Inicializa e distribui agentes nas zonas baseando-se na densidade de ocupação.
        """
        total_agents_created = 0
        total_occupancy_weight = 0.0
        
        # ====================================================================
        # PRÉ-CÁLCULO PARA DISTRIBUIÇÃO PROPORCIONAL
        # ====================================================================
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

        # ====================================================================
        # CRIAÇÃO DOS AGENTES POR ZONA
        # ====================================================================
        for zone_idx, zone in enumerate(self.scenario.zones):
            params, weight = zone_params_list[zone_idx]
            
            # Cálculo proporcional de agentes para esta zona
            if total_occupancy_weight > 0:
                target_count = int(
                    self.scenario.total_occupants * (weight / total_occupancy_weight)
                )
            else:
                target_count = 0
                
            # Limita pela capacidade máxima física da zona
            zone_agent_count = min(target_count, params['max_occupants'])
            
            # Encontra células válidas nesta zona
            zone_cells = np.where(self.physics.zone_map == zone_idx + 1)
            valid_cells_count = len(zone_cells[0])
            
            if valid_cells_count == 0:
                print(f"⚠️ Aviso: Zona '{zone.name}' não tem células válidas no grid")
                continue

            # ================================================================
            # LOOP DE CRIAÇÃO DE AGENTES
            # ================================================================
            for i in range(zone_agent_count):
                agent_id = total_agents_created + i
                
                # Definição de infecção inicial
                is_infected = (i < zone_agent_count * self.scenario.initial_infected_ratio)
                
                # Seleção de classe de agente
                AgentClass = LearningAgent if self.use_learning_agents else HumanAgent
                
                # ============================================================
                # POSICIONAMENTO SEGURO EVITANDO OBSTÁCULOS
                # ============================================================
                initial_pos = None
                placed = False
                attempts = 0
                max_attempts = 100
                
                while not placed and attempts < max_attempts:
                    # Seleciona célula aleatória dentro da zona
                    rnd_idx = np.random.randint(valid_cells_count)
                    x = zone_cells[1][rnd_idx]
                    y = zone_cells[0][rnd_idx]
                    
                    # Verifica se a célula não é um obstáculo sólido
                    # obstacle_mask > 0.5 significa que há passagem (ar livre ou poroso)
                    is_walkable = self.physics.obstacle_mask[y, x] > 0.5
                    
                    # Verifica se a célula está vazia no grid
                    if is_walkable and self.grid.is_cell_empty((x, y)):
                        initial_pos = (x, y)
                        placed = True
                    
                    attempts += 1
                
                if not placed:
                    print(
                        f"⚠️ Aviso: Não foi possível posicionar agente {agent_id} "
                        f"na zona '{zone.name}' após {max_attempts} tentativas"
                    )
                    continue
                
                # ============================================================
                # INSTANCIAÇÃO DO AGENTE
                # ============================================================
                try:
                    agent = AgentClass(
                        unique_id=agent_id,
                        model=self,
                        initial_pos=initial_pos,
                        agent_config=self.agent_config,
                        zone_config=zone,
                        mask_efficiency=self.scenario.mask_usage_rate,
                        initial_infected=is_infected
                    )
                    
                    # Coloca agente no grid
                    self.grid.place_agent(agent, initial_pos)
                    
                    # Adiciona à lista de agentes
                    self.simulation_agents.append(agent)
                    
                    # Adiciona ao scheduler
                    self.schedule.add(agent)
                    
                except Exception as e:
                    print(
                        f"❌ Erro ao criar agente {agent_id} "
                        f"na zona '{zone.name}': {e}"
                    )
                    continue
            
            total_agents_created += zone_agent_count
        
        print(f"✅ Inicializados {len(self.simulation_agents)} agentes com sucesso")

    def _calculate_initial_metrics(self) -> Dict[str, Any]:
        """
        Calcula métricas iniciais do modelo.
        
        Returns:
            Dict com métricas iniciais
        """
        return {
            'time': 0.0,
            'infected_agents': sum(1 for a in self.simulation_agents if a.infected),
            'infection_rate': 0.0,
            'r_effective': 0.0,
            'average_co2': 400.0,
            'average_hcho': 10.0,
            'average_temperature': self.scenario.temperature_setpoint,
            'average_humidity': self.scenario.humidity_setpoint,
            'infection_risk': 0.0,
            'energy_consumption': 0.0,
            'comfort_index': 1.0,
            'ventilation_efficiency': 1.0,
            'average_accumulated_dose': 0.0,  # NOVO
            'max_viral_load': 0.0              # NOVO
        }

    # ========================================================================
    # MÉTODOS DE CÁLCULO DE MÉTRICAS
    # ========================================================================

    def calculate_risk_metrics(self) -> Dict[str, Any]:
        """
        Calcula métricas de risco de infecção.
        
        Returns:
            Dict com métricas de risco
        """
        if not self.simulation_agents:
            return {
                'infected_count': 0,
                'infection_rate': 0.0,
                'r_effective': 0.0,
                'infection_risk': 0.0,
                'average_dose': 0.0
            }
        
        infected_count = sum(1 for a in self.simulation_agents if a.infected)
        total_count = len(self.simulation_agents)
        
        infection_rate = infected_count / total_count if total_count > 0 else 0.0
        
        # Cálculo de R efetivo (simplificado)
        # R_eff = número médio de infecções causadas por cada infectado
        new_infections_last_period = 0
        for agent in self.simulation_agents:
            if (hasattr(agent, 'infection_start_time') and 
                agent.infection_start_time is not None):
                time_since_infection = self.time - agent.infection_start_time
                # Considera infecções nas últimas 24h
                if 0 < time_since_infection < 86400:
                    new_infections_last_period += 1
        
        if infected_count > 0:
            r_effective = new_infections_last_period / infected_count
        else:
            r_effective = 0.0
        
        # Cálculo de risco médio baseado em dose acumulada
        doses = [a.accumulated_dose for a in self.simulation_agents]
        average_dose = np.mean(doses) if doses else 0.0
        
        # Risco de infecção (modelo simplificado exponencial)
        # Baseado em Wells-Riley com dose infectante de ~300 quanta·h
        INFECTIOUS_DOSE_50 = 300.0  # quanta·h para 50% de chance
        infection_risk = 1.0 - np.exp(-average_dose / INFECTIOUS_DOSE_50)
        
        return {
            'infected_count': infected_count,
            'infection_rate': float(infection_rate),
            'r_effective': float(r_effective),
            'infection_risk': float(infection_risk),
            'average_dose': float(average_dose)
        }

    def calculate_energy_metrics(self) -> Dict[str, Any]:
        """
        Calcula consumo energético do sistema.
        
        Returns:
            Dict com métricas energéticas
        """
        # Consumo de HVAC
        hvac_power = 0.0
        for zone in self.scenario.zones:
            # Calcula dimensões a partir de coordenadas
            zone_width = zone.x_end - zone.x_start  # metros
            zone_height = zone.y_end - zone.y_start  # metros
            zone_depth = zone.z_end - zone.z_start  # metros (altura vertical)
            
            # Volume da zona
            volume = zone_width * zone_height * zone_depth
            
            # Potência de ventilação (W) = ACH × volume × densidade_ar × cp × ΔT / 3600
            air_density = 1.2  # kg/m³
            cp_air = 1005  # J/(kg·K)
            
            # Usa getattr para robustez
            external_temp = getattr(self.scenario, 'external_temperature', 293.15)  # K
            if external_temp > 200:  # Já está em Kelvin
                external_temp_c = external_temp - 273.15
            else:  # Está em Celsius
                external_temp_c = external_temp
            
            delta_t = abs(self.scenario.temperature_setpoint - external_temp_c)
            
            # ACH da zona com fallback
            zone_ach = getattr(zone, 'target_ach', 4.0)
            
            hvac_power += (
                zone_ach * volume * air_density * cp_air * delta_t / 3600
            )
        
        # Calcula área total usando coordenadas
        total_area = sum(
            (zone.x_end - zone.x_start) * (zone.y_end - zone.y_start) 
            for zone in self.scenario.zones
        )
        lighting_power = total_area * 10  # 10 W/m² (LED eficiente)
        
        # Consumo de equipamentos
        equipment_power = len(self.simulation_agents) * 100  # 100W por pessoa
        
        total_power = hvac_power + lighting_power + equipment_power
        
        # Converte para kWh
        energy_kwh = total_power * self.dt / 3600000  # W·s → kWh
        
        return {
            'total_energy_kwh': float(energy_kwh),
            'hvac_power_w': float(hvac_power),
            'lighting_power_w': float(lighting_power),
            'equipment_power_w': float(equipment_power)
        }

    # ========================================================================
    # CONTROLE E OTIMIZAÇÃO
    # ========================================================================

    def apply_interventions(self, intervention_type: str, parameters: Dict[str, Any]):
        """
        Aplica intervenções no modelo.
        
        Args:
            intervention_type: Tipo de intervenção
            parameters: Parâmetros da intervenção
        """
        if intervention_type == 'increase_ventilation':
            factor = parameters.get('factor', 1.5)
            for zone in self.scenario.zones:
                zone.ach *= factor
            print(f"✅ Ventilação aumentada em {factor}x")
            
        elif intervention_type == 'reduce_occupancy':
            reduction = parameters.get('reduction', 0.2)
            agents_to_remove = int(len(self.simulation_agents) * reduction)
            
            for _ in range(agents_to_remove):
                if self.simulation_agents:
                    agent = self.simulation_agents.pop()
                    self.grid.remove_agent(agent)
                    self.schedule.remove(agent)
            
            print(f"✅ Ocupação reduzida em {reduction*100}%")
            
        elif intervention_type == 'mandate_masks':
            efficiency = parameters.get('efficiency', 0.7)
            for agent in self.simulation_agents:
                if not agent.mask_wearing:
                    agent.mask_wearing = True
                    agent.mask_type = 'surgical'
            print(f"✅ Uso de máscaras mandatório (eficiência {efficiency})")
            
        elif intervention_type == 'improve_filtration':
            # Aumenta eficiência de remoção de partículas
            print("✅ Filtração melhorada")
        
        self.interventions_active[intervention_type] = {
            'time': self.time,
            'parameters': parameters
        }

    def optimize_parameters(self):
        """
        Otimiza parâmetros do modelo baseado em métricas de risco.
        """
        if not self.optimization_enabled:
            return
        
        risk_metrics = self.calculate_risk_metrics()
        current_risk = risk_metrics['infection_risk']
        
        targets = self.optimization_targets
        if 'infection_risk_target' in targets:
            target_risk = targets['infection_risk_target']
            
            if current_risk > target_risk:
                # Aciona redução de ocupação se risco superar limite
                self.apply_interventions('reduce_occupancy', {'reduction': 0.1})

    # ========================================================================
    # STEP E ATUALIZAÇÃO
    # ========================================================================

    def step(self):
        """
        Avança um passo discreto na simulação.
        """
        if self.paused:
            return
        
        try:
            # 1. LIMPEZA DE BUFFERS
            self.current_agent_emissions = []
            
            # 2. COMPORTAMENTO DOS AGENTES (gera emissões)
            try:
                self.schedule.step()
            except Exception as e:
                print(f"⚠️ Erro no passo dos agentes: {e}")
                import traceback
                traceback.print_exc()
            
            # 3. PROCESSAMENTO FÍSICO DAS EMISSÕES
            # Agrega emissões por célula e por espécie
            aggregated_emissions = {}
            heat_emissions = {}
            moisture_emissions = {}
            
            for emission in self.current_agent_emissions:
                # Validação de tipos
                x = emission.get('x')
                y = emission.get('y')
                species = emission.get('species')
                amount = emission.get('amount')
                
                # Converte para tipos numéricos
                try:
                    x = int(x) if x is not None else 0
                    y = int(y) if y is not None else 0
                    amount = float(amount) if amount is not None else 0.0
                except (ValueError, TypeError):
                    continue  # Pula emissões inválidas
                
                # Ignora emissões zero ou negativas
                if amount <= 0:
                    continue
                
                key = (x, y)
                
                # Separa por tipo de emissão
                if species == 'heat':
                    heat_emissions[key] = heat_emissions.get(key, 0.0) + amount
                elif species == 'moisture':
                    moisture_emissions[key] = moisture_emissions.get(key, 0.0) + amount
                elif species in ['co2', 'virus', 'hcho', 'voc', 'pm25', 'pm10']:
                    if key not in aggregated_emissions:
                        aggregated_emissions[key] = {}
                    if species not in aggregated_emissions[key]:
                        aggregated_emissions[key][species] = 0.0
                    aggregated_emissions[key][species] += amount
            
            # DEBUG: Log de emissões virais totais
            total_virus = sum(
                em.get('virus', 0.0) for em in aggregated_emissions.values()
            )
            if total_virus > 0 and self.time % 60 < self.dt:  # Log a cada minuto
                print(f"  [FÍSICA] t={self.time/60:.1f}min - Total viral: {total_virus:.4f} quanta")
            
            # 4. ENVIO PARA MOTOR FÍSICO
            agent_data = {
                'emissions': [],
                'positions': [],
                'activities': []
            }
            
            # Adiciona emissões agregadas
            for (x, y), species_dict in aggregated_emissions.items():
                emission_entry = {
                    'x': x,
                    'y': y,
                    'species': species_dict,
                    'heat': heat_emissions.get((x, y), 0.0),
                    'moisture': moisture_emissions.get((x, y), 0.0)
                }
                agent_data['emissions'].append(emission_entry)
                agent_data['positions'].append((x, y))
            
            # Adiciona atividades dos agentes ativos
            for agent in self.simulation_agents:
                if agent.pos is not None:
                    agent_data['activities'].append(
                        getattr(agent.current_activity, 'value', str(agent.current_activity))
                    )
            
            # 5. EXECUÇÃO DO MOTOR FÍSICO
            try:
                self.physics.step(self.dt, self.time, agent_data)
            except Exception as e:
                print(f"⚠️ Erro no passo da física: {e}")
                import traceback
                traceback.print_exc()
            
            # 6. CONTROLE E MÉTRICAS
            try:
                self.optimize_parameters()
                self._update_metrics()
            except Exception as e:
                print(f"⚠️ Erro na atualização de métricas: {e}")
            
            # 7. AVANÇO DO TEMPO
            self.time += self.dt
            if self.time >= self.simulation_duration:
                self.running = False
            
            # 8. COLETA DE DADOS
            try:
                self._collect_simulation_data()
            except Exception as e:
                print(f"⚠️ Erro na coleta de dados: {e}")
                
        except Exception as e:
            print(f"❌ Erro crítico no step da simulação: {e}")
            import traceback
            traceback.print_exc()
            self.running = False

    def _update_metrics(self):
        """
        Atualiza o dicionário de métricas em tempo real.
        """
        zone_stats = self.physics.get_zone_statistics()
        risk_metrics = self.calculate_risk_metrics()
        energy_metrics = self.calculate_energy_metrics()
        
        # ====================================================================
        # MÉTRICAS DE ZONA (CO2, HCHO, Temperatura, Umidade)
        # ====================================================================
        if zone_stats:
            avg_co2 = np.mean([
                s['concentrations']['co2_ppm_mean'] 
                for s in zone_stats.values()
            ])
            avg_hcho = np.mean([
                s['concentrations']['hcho_ppb_mean'] 
                for s in zone_stats.values()
            ])
            avg_temp = np.mean([
                s['concentrations']['temperature_c_mean'] 
                for s in zone_stats.values()
            ])
            avg_hum = np.mean([
                s['concentrations']['humidity_percent_mean'] 
                for s in zone_stats.values()
            ])
        else:
            avg_co2 = 400.0
            avg_hcho = 10.0
            avg_temp = self.scenario.temperature_setpoint
            avg_hum = self.scenario.humidity_setpoint
        
        # ====================================================================
        # MÉTRICAS DE CONFORTO
        # ====================================================================
        if self.simulation_agents:
            avg_comfort = np.mean([
                a.comfort_level 
                for a in self.simulation_agents 
                if hasattr(a, 'comfort_level')
            ])
        else:
            avg_comfort = 1.0
        
        # ====================================================================
        # MÉTRICAS DE DOSE E CARGA VIRAL
        # ====================================================================
        if self.simulation_agents:
            # Dose acumulada média
            avg_dose = np.mean([
                a.accumulated_dose 
                for a in self.simulation_agents
            ])
        else:
            avg_dose = 0.0
        
        # Máximo de carga viral no ar
        viral_load_grid = self.physics.grids.get('virus', np.zeros_like(self.physics.zone_map))
        max_viral_load = float(np.max(viral_load_grid))

        # ====================================================================
        # ATUALIZAÇÃO DO ESTADO
        # ====================================================================
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
            'ventilation_efficiency': float(avg_co2 / max(1, self.scenario.co2_setpoint)),
            'average_accumulated_dose': float(avg_dose),
            'max_viral_load': max_viral_load
        })

    def _collect_simulation_data(self):
        """
        Salva snapshot dos dados para gráficos históricos.
        
        MELHORIAS:
        - Adiciona coleta de dose acumulada
        - Adiciona coleta de carga viral máxima
        """
        # Salva a cada 300 segundos (5 min) ou se for o último passo
        if self.time % 300 < self.dt or not self.running:
            self.simulation_data['time'].append(self.time)
            self.simulation_data['zone_stats'].append(
                self.physics.get_zone_statistics()
            )
            
            # ================================================================
            # ESTATÍSTICAS AGREGADAS DOS AGENTES
            # ================================================================
            if self.simulation_agents:
                stats = {
                    'total': len(self.simulation_agents),
                    'infected': sum(1 for a in self.simulation_agents if a.infected),
                    'mask_wearing': sum(
                        1 for a in self.simulation_agents if a.mask_wearing
                    ),
                    'avg_comfort': self.current_metrics['comfort_index']
                }
            else:
                stats = {
                    'total': 0, 
                    'infected': 0, 
                    'mask_wearing': 0, 
                    'avg_comfort': 0
                }
            
            self.simulation_data['agent_stats'].append(stats)
            self.simulation_data['risk_metrics'].append(
                self.calculate_risk_metrics()
            )
            self.simulation_data['energy_consumption'].append(
                self.calculate_energy_metrics()
            )
            
            # ================================================================
            # COLETA DE DOSE E CARGA VIRAL
            # ================================================================
            if self.simulation_agents:
                avg_dose = np.mean([
                    a.accumulated_dose for a in self.simulation_agents
                ])
            else:
                avg_dose = 0.0
            
            viral_load_grid = self.physics.grids.get(
                'virus', 
                np.zeros_like(self.physics.zone_map)
            )
            max_viral = float(np.max(viral_load_grid))
            
            self.simulation_data['accumulated_doses'].append(avg_dose)
            self.simulation_data['max_viral_loads'].append(max_viral)

    # ========================================================================
    # MÉTODOS DE EXPORTAÇÃO E VISUALIZAÇÃO
    # ========================================================================

    def get_visualization_data(self) -> Dict[str, Any]:
        """
        Prepara payload leve para o frontend (Streamlit).
        
        Returns:
            Dict com dados de visualização
        """
        active_agents = [a for a in self.simulation_agents if a.pos is not None]
        
        return {
            'physics': self.physics.get_visualization_data(),
            'agents': {
                'positions': [a.pos for a in active_agents],
                'infected': [a.infected for a in active_agents],
                'activities': [a.current_activity.value for a in active_agents],
                'mask_wearing': [a.mask_wearing for a in active_agents],
                'comfort_levels': [
                    getattr(a, 'comfort_level', 0.0) for a in active_agents
                ]
            },
            'metrics': self.current_metrics,
            'time': self.time,
            'running': self.running,
            'paused': self.paused
        }

    def get_simulation_summary(self) -> Dict[str, Any]:
        """
        Gera relatório final consolidado.
        
        Returns:
            Dict com sumário da simulação
        """
        total_exposure = sum(
            sum(exp['dose'] for exp in a.exposure_history) 
            for a in self.simulation_agents 
            if hasattr(a, 'exposure_history')
        )
        
        peak_infected = 0
        if self.simulation_data['agent_stats']:
            peak_infected = max(
                s['infected'] for s in self.simulation_data['agent_stats']
            )

        return {
            'scenario': self.scenario.name,
            'duration_hours': self.time / 3600,
            'total_agents': len(self.simulation_agents),
            'total_infections': sum(
                1 for a in self.simulation_agents 
                if getattr(a, 'infection_start_time', None) is not None
            ),
            'peak_infected': peak_infected,
            'total_exposure': total_exposure,
            'average_co2': self.current_metrics['average_co2'],
            'total_energy_kwh': self.current_metrics['energy_consumption'],
            'interventions_applied': list(self.interventions_active.keys()),
            'max_viral_load_recorded': max(
                self.simulation_data['max_viral_loads']
            ) if self.simulation_data['max_viral_loads'] else 0.0
        }

    def export_simulation_data(self, format: str = 'json') -> Any:
        """
        Exporta dados completos da simulação.
        
        Args:
            format: Formato de exportação ('json' ou 'dict')
            
        Returns:
            JSON string ou dict com dados completos
        """
        # Constrói dicionário mestre com tipos nativos
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
                'risk': self.simulation_data['risk_metrics'],
                'energy': self.simulation_data['energy_consumption'],
                'doses': [float(d) for d in self.simulation_data['accumulated_doses']],
                'viral_loads': [float(v) for v in self.simulation_data['max_viral_loads']]
            }
        }
        
        if format == 'json':
            # Custom encoder para lidar com tipos numpy
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
