import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
import sys
import os

# Adicionar caminho dos módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar módulos do simulador
import config_final as cfg
from main_model import IAQSimulationModel
from unified_physics import UnifiedPhysicsEngine

# Configuração da página
st.set_page_config(
    page_title="Simulador IAQ - Dashboard",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E88E5, #4CAF50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border-left: 5px solid #1E88E5;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
    .agent-card {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        border: 2px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    .infected-badge {
        background: #ff5252;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .susceptible-badge {
        background: #4CAF50;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .recovered-badge {
        background: #9E9E9E;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

@st.cache_data
def get_obstacle_style(obstacle_type: str) -> dict:
    """
    Retorna o estilo de visualização para cada tipo de obstáculo.
    
    Args:
        obstacle_type: Tipo do obstáculo (WALL, FURNITURE, etc.)
    
    Returns:
        Dicionário com fillcolor e opacity
    """
    styles = {
        cfg.ObstacleType.WALL.value: {
            'fillcolor': 'rgba(50, 50, 50, 1.0)',  # Preto/cinza escuro
            'opacity': 1.0,
            'line_color': 'black',
            'line_width': 2
        },
        cfg.ObstacleType.FURNITURE.value: {
            'fillcolor': 'rgba(139, 90, 43, 0.6)',  # Marrom madeira
            'opacity': 0.6,
            'line_color': 'rgba(90, 60, 30, 1.0)',
            'line_width': 1
        },
        cfg.ObstacleType.PARTITION.value: {
            'fillcolor': 'rgba(150, 150, 150, 0.5)',  # Cinza claro
            'opacity': 0.5,
            'line_color': 'gray',
            'line_width': 1
        },
        cfg.ObstacleType.EQUIPMENT.value: {
            'fillcolor': 'rgba(70, 130, 180, 0.5)',  # Azul metálico
            'opacity': 0.5,
            'line_color': 'steelblue',
            'line_width': 1
        }
    }
    
    # Retorna estilo padrão se tipo desconhecido
    return styles.get(obstacle_type, {
        'fillcolor': 'rgba(128, 128, 128, 0.5)',
        'opacity': 0.5,
        'line_color': 'gray',
        'line_width': 1
    })


def add_obstacles_to_figure(fig: go.Figure, obstacles: list, cell_size: float = 0.1):
    """
    Adiciona obstáculos ao mapa de calor como retângulos.
    
    Args:
        fig: Figura Plotly
        obstacles: Lista de obstáculos (dicionários ou objetos Obstacle)
        cell_size: Tamanho da célula para conversão de coordenadas
    """
    if not obstacles:
        return
    
    for obs in obstacles:
        # Extrair dados do obstáculo (suporta dict ou objeto)
        if isinstance(obs, dict):
            x = obs['x']
            y = obs['y']
            width = obs['width']
            height = obs['height']
            obs_type = obs['type']
        else:
            x = obs.x
            y = obs.y
            width = obs.width
            height = obs.height
            obs_type = obs.obstacle_type.value if hasattr(obs.obstacle_type, 'value') else obs.obstacle_type
        
        # Obter estilo
        style = get_obstacle_style(obs_type)
        
        # Adicionar retângulo
        fig.add_shape(
            type="rect",
            x0=x / cell_size,
            y0=y / cell_size,
            x1=(x + width) / cell_size,
            y1=(y + height) / cell_size,
            fillcolor=style['fillcolor'],
            opacity=style['opacity'],
            line=dict(
                color=style['line_color'],
                width=style['line_width']
            ),
            layer="above"
        )


def get_agent_data(model) -> pd.DataFrame:
    """
    Extrai dados dos agentes do modelo para análise.
    
    Args:
        model: Modelo de simulação
    
    Returns:
        DataFrame com informações dos agentes
    """
    if model is None or not hasattr(model, 'schedule'):
        return pd.DataFrame()
    
    agents_data = []
    
    for agent in model.schedule.agents:
        agent_info = {
            'id': agent.unique_id,
            'x': agent.pos[0] if agent.pos else None,
            'y': agent.pos[1] if agent.pos else None,
            'infected': agent.infected,
            'activity': agent.current_activity.value if hasattr(agent.current_activity, 'value') else str(agent.current_activity),
            'mask_wearing': agent.mask_wearing,
            'accumulated_dose': getattr(agent, 'accumulated_dose', 0),
            'viral_load': getattr(agent, 'viral_load', 0),
            'exposure_history_length': len(getattr(agent, 'exposure_history', []))
        }
        agents_data.append(agent_info)
    
    return pd.DataFrame(agents_data)


def plot_agent_exposure_timeline(agent, model) -> go.Figure:
    """
    Cria gráfico de timeline da exposição do agente.
    
    Args:
        agent: Agente a ser analisado
        model: Modelo de simulação
    
    Returns:
        Figura Plotly com timeline de exposição
    """
    if not hasattr(agent, 'exposure_history') or not agent.exposure_history:
        fig = go.Figure()
        fig.add_annotation(
            text="Sem histórico de exposição disponível",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    # Extrair dados do histórico
    times = [record['time'] / 3600 for record in agent.exposure_history]  # Converter para horas
    doses = [record['accumulated_dose'] for record in agent.exposure_history]
    viral_loads = [record['viral_load_cell'] for record in agent.exposure_history]
    
    # Criar figura com dois eixos Y
    fig = make_subplots(
        specs=[[{"secondary_y": True}]],
        subplot_titles=(f"Timeline de Exposição - Agente {agent.unique_id}",)
    )
    
    # Linha da dose acumulada
    fig.add_trace(
        go.Scatter(
            x=times,
            y=doses,
            name="Dose Acumulada (quanta)",
            line=dict(color='red', width=3),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.1)'
        ),
        secondary_y=False
    )
    
    # Área da carga viral na célula
    fig.add_trace(
        go.Scatter(
            x=times,
            y=viral_loads,
            name="Carga Viral na Célula (quanta/m³)",
            line=dict(color='gray', width=1, dash='dot'),
            fill='tozeroy',
            fillcolor='rgba(128, 128, 128, 0.2)'
        ),
        secondary_y=True
    )
    
    # Marcar momento da infecção (se aplicável)
    if agent.infected and hasattr(agent, 'infection_start_time') and agent.infection_start_time is not None:
        infection_time_h = agent.infection_start_time / 3600
        
        # Encontrar dose no momento da infecção
        infection_dose = 0
        for i, t in enumerate(times):
            if t >= infection_time_h:
                infection_dose = doses[i] if i < len(doses) else doses[-1]
                break
        
        fig.add_vline(
            x=infection_time_h,
            line_dash="dash",
            line_color="orange",
            annotation_text="⚠️ Infecção",
            annotation_position="top"
        )
        
        fig.add_trace(
            go.Scatter(
                x=[infection_time_h],
                y=[infection_dose],
                mode='markers',
                marker=dict(size=15, color='orange', symbol='star'),
                name='Momento da Infecção',
                showlegend=True
            ),
            secondary_y=False
        )
    
    # Configurar layout
    fig.update_xaxes(title_text="Tempo (horas)")
    fig.update_yaxes(title_text="Dose Acumulada (quanta)", secondary_y=False)
    fig.update_yaxes(title_text="Carga Viral na Célula (quanta/m³)", secondary_y=True)
    
    fig.update_layout(
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


# ============================================================================
# INICIALIZAÇÃO DO ESTADO DA SESSÃO
# ============================================================================

def init_session_state():
    default_state = {
        'simulation_model': None,
        'simulation_running': False,
        'simulation_paused': False,
        'simulation_data': [],
        'selected_scenario': 'office',
        'simulation_history': [],
        'active_interventions': [],
        'current_step': 0,
        'total_steps': 0,
        'dt_seconds': 60,
        'selected_agent_id': None  # NOVO: para inspeção de agente
    }
    
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================================================================
# CABEÇALHO
# ============================================================================

st.markdown('<h1 class="main-header">🏢 Simulador IAQ - Dashboard Avançado</h1>', unsafe_allow_html=True)

# ============================================================================
# BARRA LATERAL
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Configuração do Cenário")
    
    # Seleção de cenário
    scenario_options = {
        "🏫 Escola (Sala de Aula)": "school",
        "🏢 Escritório (Open Space)": "office", 
        "💪 Academia (Fitness Center)": "gym",
    }
    
    selected_scenario_name = st.selectbox(
        "Tipo de Edificação",
        list(scenario_options.keys()),
        index=list(scenario_options.values()).index(st.session_state.selected_scenario)
    )
    st.session_state.selected_scenario = scenario_options[selected_scenario_name]
    
    # Parâmetros da simulação
    tab_params = st.tabs(["👥 Ocupação", "💨 Ventilação", "🌡️ Ambiente", "⚙️ Avançado"])
    
    with tab_params[0]:
        total_occupants = st.slider("Número total de ocupantes", 1, 500, 50, 5)
        initial_infected_ratio = st.slider("Taxa inicial de infectados (%)", 0.0, 50.0, 5.0, 0.5) / 100.0
        mask_usage_rate = st.slider("Uso inicial de máscaras (%)", 0.0, 100.0, 20.0, 5.0)
    
    with tab_params[1]:
        ventilation_strategy = st.selectbox(
            "Estratégia de ventilação",
            ["demand_controlled", "constant_volume", "natural"],
            index=0
        )
        ach_target = st.slider("ACH Alvo (Trocas/h)", 0.5, 20.0, 4.0, 0.5)
        co2_setpoint = st.number_input("Setpoint de CO₂ (ppm)", 400, 2000, 800, 50)
    
    with tab_params[2]:
        temperature_setpoint = st.slider("Temperatura (°C)", 18.0, 28.0, 22.0, 0.5)
        humidity_setpoint = st.slider("Umidade (%)", 30.0, 80.0, 50.0, 5.0)
        
        st.markdown("---")
        outdoor_temp = st.slider("Temp. Externa (°C)", 0.0, 40.0, 25.0, 1.0)
        outdoor_humidity = st.slider("Umid. Externa (%)", 10.0, 100.0, 60.0, 5.0)
    
    with tab_params[3]:
        cell_size = st.selectbox("Tamanho da célula (m)", [0.1, 0.2, 0.5], index=0)
        simulation_hours = st.slider("Duração (horas)", 1, 24, 4, 1)
        time_step = st.selectbox("Passo de tempo (s)", [1, 5, 10, 30, 60], index=2)
        
        st.caption("Física:")
        enable_kalman = st.checkbox("Filtro de Kalman (Estimação)", value=True)
        enable_plume_correction = st.checkbox("Correção Pluma Térmica", value=True)

    # Controles de simulação
    st.markdown("---")
    st.markdown("### 🎮 Controles")
    
    col_start, col_pause, col_reset = st.columns(3)
    
    def initialize_simulation():
        try:
            # 1. Obter cenário base
            if st.session_state.selected_scenario == 'school':
                scenario = cfg.create_school_scenario()
            elif st.session_state.selected_scenario == 'gym':
                scenario = cfg.create_gym_scenario()
            elif st.session_state.selected_scenario == 'office':
                scenario = cfg.create_office_scenario()
            else:
                scenario = cfg.create_office_scenario()
            
            # 2. Atualizar ventilação
            scenario.ventilation.ach = ach_target
            
            # 3. Config física
            physics_config = {
                'cell_size': cell_size,
                'diffusion_coefficient': 0.00001,
                'dt': time_step
            }
            
            # 4. Criar Modelo
            st.session_state.simulation_model = IAQSimulationModel(
                scenario=scenario,
                num_agents=total_occupants,
                initial_infected_ratio=initial_infected_ratio,
                dt=time_step
            )
            
            # Resetar contadores
            st.session_state.dt_seconds = time_step
            st.session_state.total_steps = int(simulation_hours * 3600 / time_step)
            st.session_state.current_step = 0
            st.session_state.simulation_data = []
            st.session_state.simulation_history = []
            st.session_state.active_interventions = []
            st.session_state.selected_agent_id = None
            
            return True
            
        except Exception as e:
            st.error(f"Erro na inicialização: {e}")
            import traceback
            st.code(traceback.format_exc())
            return False

    with col_start:
        if st.button("▶️ Iniciar", type="primary", disabled=st.session_state.simulation_running):
            if initialize_simulation():
                st.session_state.simulation_running = True
                st.session_state.simulation_paused = False
                st.rerun()
    
    with col_pause:
        pause_label = "▶️ Retomar" if st.session_state.simulation_paused else "⏸️ Pausar"
        if st.button(pause_label, disabled=not st.session_state.simulation_running):
            st.session_state.simulation_paused = not st.session_state.simulation_paused
            st.rerun()
    
    with col_reset:
        if st.button("🔄 Reiniciar"):
            st.session_state.simulation_running = False
            st.session_state.simulation_paused = False
            st.session_state.simulation_model = None
            st.session_state.selected_agent_id = None
            st.rerun()

# ============================================================================
# ÁREA PRINCIPAL
# ============================================================================

if st.session_state.simulation_running and st.session_state.simulation_model is not None:
    model = st.session_state.simulation_model
    
    # Métricas
    progress = st.session_state.current_step / max(st.session_state.total_steps, 1)
    time_elapsed = model.time if hasattr(model, 'time') else st.session_state.current_step * st.session_state.dt_seconds
    
    st.progress(progress, text=f"Progresso: {progress*100:.1f}% | Tempo: {time_elapsed/3600:.2f}h")
    
    # KPIs
    metrics = getattr(model, 'current_metrics', {})
    
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    with kpi1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        co2 = metrics.get('average_co2', 0)
        st.metric("CO₂ Médio", f"{co2:.0f} ppm")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with kpi2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        infected_count = metrics.get('infected_count', 0)
        total_agents = metrics.get('total_agents', 1)
        st.metric("Infectados", f"{infected_count}/{total_agents}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with kpi3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        risk = metrics.get('infection_risk', 0)
        st.metric("Risco Médio", f"{risk*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    with kpi4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        r_eff = metrics.get('r_effective', 0)
        st.metric("R-Efetivo", f"{r_eff:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with kpi5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        energy = metrics.get('energy_consumption', 0)
        st.metric("Energia", f"{energy:.2f} kWh")
        st.markdown('</div>', unsafe_allow_html=True)

    # Abas de Visualização
    tab_charts, tab_map, tab_inspector, tab_zones, tab_export = st.tabs([
        "📈 Gráficos Temporais", 
        "🗺️ Mapa de Calor", 
        "🔍 Inspeção de Agente",  # NOVA ABA
        "🏢 Detalhe Zonas", 
        "💾 Exportar"
    ])

    # --- ABA 1: GRÁFICOS ---
    with tab_charts:
        if len(st.session_state.simulation_history) > 1:
            df_hist = pd.DataFrame([
                {'time_h': h['time']/3600, **h['metrics']} 
                for h in st.session_state.simulation_history
            ])
            
            fig = make_subplots(
                rows=2, cols=2, 
                subplot_titles=("CO₂ (ppm)", "Risco de Infecção", "HCHO (ppb)", "Temperatura e Umidade")
            )
            
            # CO2
            fig.add_trace(
                go.Scatter(x=df_hist['time_h'], y=df_hist['average_co2'], 
                          name="CO₂", line=dict(color='orange')), 
                row=1, col=1
            )
            
            # Risco
            fig.add_trace(
                go.Scatter(x=df_hist['time_h'], y=df_hist['infection_risk'], 
                          name="Risco", line=dict(color='red')), 
                row=1, col=2
            )
            
            # HCHO
            if 'average_hcho' in df_hist.columns:
                fig.add_trace(
                    go.Scatter(x=df_hist['time_h'], y=df_hist['average_hcho'], 
                              name="HCHO", line=dict(color='purple')), 
                    row=2, col=1
                )
            
            # Temp/Hum
            if 'average_temperature' in df_hist.columns:
                fig.add_trace(
                    go.Scatter(x=df_hist['time_h'], y=df_hist['average_temperature'], 
                              name="Temp (°C)"), 
                    row=2, col=2
                )
            if 'average_humidity' in df_hist.columns:
                fig.add_trace(
                    go.Scatter(x=df_hist['time_h'], y=df_hist['average_humidity']*100, 
                              name="Umid (%)", line=dict(dash='dot')), 
                    row=2, col=2
                )
            
            fig.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Iniciando coleta de dados...")

    # --- ABA 2: MAPA DE CALOR COM OBSTÁCULOS ---
    with tab_map:
        st.markdown("### 🗺️ Mapa de Concentração de CO₂")
        
        if st.session_state.simulation_data:
            last_frame = st.session_state.simulation_data[-1]
            grids = last_frame.get('physics', {}).get('grids', {})
            
            if 'co2_ppm' in grids:
                co2_grid = np.array(grids['co2_ppm'])
                
                # Mapa de Calor 2D
                fig_map = go.Figure(data=go.Heatmap(
                    z=co2_grid,
                    colorscale='Jet',
                    colorbar=dict(title='CO₂ (ppm)'),
                    hovertemplate='X: %{x}<br>Y: %{y}<br>CO₂: %{z:.0f} ppm<extra></extra>'
                ))
                
                # ========================================================
                # ADICIONAR OBSTÁCULOS (CORREÇÃO DE INTEGRAÇÃO)
                # ========================================================
                # Tenta acessar via modelo ou cenário salvo
                scenario_obj = getattr(model, 'scenario', None)
                
                if scenario_obj and hasattr(scenario_obj, 'obstacles'):
                    obstacles = scenario_obj.obstacles
                    
                    for obs in obstacles:
                        # 1. Extração de dados (Híbrido Dict/Objeto)
                        if isinstance(obs, dict):
                            # Modo legado (Dict)
                            x0, y0 = obs['x'], obs['y']
                            w, h = obs['width'], obs['height']
                            otype = obs['type']
                        else:
                            # Modo novo (Objeto)
                            x0, y0 = obs.x, obs.y
                            w, h = obs.width, obs.height
                            # Tenta pegar o valor string do Enum se for Enum
                            raw_type = getattr(obs, 'obstacle_type', getattr(obs, 'type', 'wall'))
                            otype = raw_type.value if hasattr(raw_type, 'value') else str(raw_type)

                        # Converter de metros para células (indices do grid)
                        # Nota: O Heatmap usa índices, então precisamos dividir pelo cell_size
                        x0_idx = x0 / cell_size
                        y0_idx = y0 / cell_size
                        w_idx = w / cell_size
                        h_idx = h / cell_size
                        
                        x1_idx = x0_idx + w_idx
                        y1_idx = y0_idx + h_idx
                        
                        # 2. Definição de Estilo
                        if 'wall' in str(otype).lower() or 'partition' in str(otype).lower():
                            color = "black"
                            opacity = 1.0
                            line_width = 2
                        elif 'furniture' in str(otype).lower():
                            color = "#8B4513" # SaddleBrown
                            opacity = 0.6
                            line_width = 1
                        else:
                            color = "gray"
                            opacity = 0.5
                            line_width = 1
                        
                        # 3. Desenho
                        fig_map.add_shape(
                            type="rect",
                            x0=x0_idx, y0=y0_idx, x1=x1_idx, y1=y1_idx,
                            line=dict(color=color, width=line_width),
                            fillcolor=color,
                            opacity=opacity
                        )
                
                # Adicionar Agentes
                agents = last_frame.get('agents', {})
                if agents.get('positions'):
                    pos = np.array(agents['positions'])
                    inf = np.array(agents['infected'])
                    
                    if len(pos) > 0:
                        # Converter posições de metros para células
                        pos_cells = pos / cell_size
                        
                        fig_map.add_trace(go.Scatter(
                            x=pos_cells[:, 0], 
                            y=pos_cells[:, 1],
                            mode='markers',
                            marker=dict(
                                color=['#FF5252' if i else '#4CAF50' for i in inf],
                                size=10,
                                line=dict(width=2, color='white'),
                                symbol='circle'
                            ),
                            name='Ocupantes',
                            hovertemplate='Agente<br>X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>'
                        ))
                
                fig_map.update_layout(
                    title="Mapa de Concentração de CO₂, Obstáculos e Ocupantes",
                    height=700,
                    yaxis=dict(scaleanchor="x", scaleratio=1),
                    xaxis_title="Posição X (células)",
                    yaxis_title="Posição Y (células)"
                )
                
                st.plotly_chart(fig_map, use_container_width=True)
                
                # Legenda
                st.markdown("""
                **Legenda:**
                - 🔴 **Vermelho**: Ocupantes infectados
                - 🟢 **Verde**: Ocupantes saudáveis
                - ⬛ **Preto**: Paredes (obstáculos sólidos)
                - 🟤 **Marrom**: Móveis (semi-transparente)
                - Fundo colorido: Concentração de CO₂
                """)
            else:
                st.warning("Dados de grid ainda não gerados.")
        else:
            st.info("Execute a simulação para visualizar o mapa.")

    # --- ABA 3: INSPEÇÃO DE AGENTE (PRONTUÁRIO MÉDICO) ---
    with tab_inspector:
        st.markdown("### 🔍 Inspeção de Agente - Prontuário Médico")
        
        if hasattr(model, 'schedule') and model.schedule.agents:
            # Obter lista de agentes
            agents_df = get_agent_data(model)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("#### Selecionar Agente")
                
                # Selectbox para escolher agente
                agent_ids = agents_df['id'].tolist()
                selected_idx = st.selectbox(
                    "ID do Agente",
                    range(len(agent_ids)),
                    format_func=lambda x: f"Agente {agent_ids[x]}"
                )
                
                selected_agent_id = agent_ids[selected_idx]
                
                # Encontrar o agente selecionado
                selected_agent = None
                for agent in model.schedule.agents:
                    if agent.unique_id == selected_agent_id:
                        selected_agent = agent
                        break
                
                if selected_agent:
                    # Card com status atual
                    st.markdown('<div class="agent-card">', unsafe_allow_html=True)
                    
                    # Badge de status
                    if selected_agent.infected:
                        status_badge = '<span class="infected-badge">INFECTADO</span>'
                    else:
                        status_badge = '<span class="susceptible-badge">SUSCETÍVEL</span>'
                    
                    st.markdown(f"**Status:** {status_badge}", unsafe_allow_html=True)
                    st.markdown(f"**Atividade:** {selected_agent.current_activity.value if hasattr(selected_agent.current_activity, 'value') else selected_agent.current_activity}")
                    st.markdown(f"**Máscara:** {'✅ Sim' if selected_agent.mask_wearing else '❌ Não'}")
                    
                    if hasattr(selected_agent, 'pos') and selected_agent.pos:
                        st.markdown(f"**Posição:** ({selected_agent.pos[0]:.2f}, {selected_agent.pos[1]:.2f}) m")
                    
                    st.markdown(f"**Dose Acumulada:** {getattr(selected_agent, 'accumulated_dose', 0):.4f} quanta")
                    
                    if selected_agent.infected:
                        st.markdown(f"**Carga Viral:** {getattr(selected_agent, 'viral_load', 0):.2f}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Estatísticas do histórico
                    if hasattr(selected_agent, 'exposure_history'):
                        st.markdown("---")
                        st.markdown("**📊 Estatísticas de Exposição**")
                        n_exposures = len(selected_agent.exposure_history)
                        st.metric("Registros de Exposição", n_exposures)
                        
                        if n_exposures > 0:
                            max_viral_load = max([r['viral_load_cell'] for r in selected_agent.exposure_history])
                            st.metric("Carga Viral Máxima", f"{max_viral_load:.4f} quanta/m³")
            
            with col2:
                if selected_agent:
                    st.markdown("#### 📈 Timeline de Exposição")
                    
                    # Plotar gráfico de timeline
                    fig_timeline = plot_agent_exposure_timeline(selected_agent, model)
                    st.plotly_chart(fig_timeline, use_container_width=True)
                    
                    # Tabela detalhada do histórico (últimos 10 registros)
                    if hasattr(selected_agent, 'exposure_history') and selected_agent.exposure_history:
                        st.markdown("#### 📋 Histórico Recente de Exposição")
                        
                        recent_history = selected_agent.exposure_history[-10:]  # Últimos 10
                        
                        history_df = pd.DataFrame([
                            {
                                'Tempo (h)': f"{r['time']/3600:.2f}",
                                'Posição': f"({r['x']:.1f}, {r['y']:.1f})" if r['x'] is not None else "N/A",
                                'Dose (quanta)': f"{r['inhaled_dose']:.6f}",
                                'Dose Acum.': f"{r['accumulated_dose']:.4f}",
                                'Carga Viral': f"{r['viral_load_cell']:.4f}",
                                'Atividade': r['activity']
                            }
                            for r in recent_history
                        ])
                        
                        st.dataframe(history_df, use_container_width=True)
                else:
                    st.info("Selecione um agente para visualizar o histórico.")
        else:
            st.info("Nenhum agente disponível. Inicie a simulação primeiro.")

    # --- ABA 4: ZONAS ---
    with tab_zones:
        if st.session_state.simulation_data:
            last_frame = st.session_state.simulation_data[-1]
            zone_stats = last_frame.get('physics', {}).get('zone_stats', {})
            
            if zone_stats:
                zone_rows = []
                for zid, data in zone_stats.items():
                    concs = data.get('concentrations', {})
                    zone_rows.append({
                        "Zona": data.get('name', zid),
                        "Ocupantes": f"{data.get('current_occupants', 0)}/{data.get('max_occupants', 0)}",
                        "CO₂ (ppm)": f"{concs.get('co2_ppm_mean', 0):.0f}",
                        "Temp (°C)": f"{concs.get('temperature_c_mean', 0):.1f}",
                        "ACH (Trocas/h)": f"{data.get('ach_actual', 0):.2f}"
                    })
                
                st.dataframe(pd.DataFrame(zone_rows), use_container_width=True)
                
                zones = [z['Zona'] for z in zone_rows]
                co2_vals = [float(z['CO₂ (ppm)']) for z in zone_rows]
                fig_bar = go.Figure([go.Bar(x=zones, y=co2_vals, marker_color='orange')])
                fig_bar.update_layout(title="Comparativo de CO₂ por Zona", yaxis_title="ppm")
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Dados de zona não disponíveis.")

    # --- ABA 5: EXPORTAR ---
    with tab_export:
        st.markdown("### 📥 Download dos Dados")
        col_json, col_csv = st.columns(2)
        
        with col_csv:
            if st.button("📊 Gerar CSV Histórico"):
                if st.session_state.simulation_history:
                    flat_data = []
                    for h in st.session_state.simulation_history:
                        row = {'time_seconds': h['time']}
                        row.update(h['metrics'])
                        flat_data.append(row)
                    csv_df = pd.DataFrame(flat_data)
                    csv_str = csv_df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Baixar CSV",
                        data=csv_str,
                        file_name=f"iaq_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("Sem histórico para exportar.")
        
        with col_json:
            if st.button("📦 Gerar Dados dos Agentes"):
                if hasattr(model, 'schedule') and model.schedule.agents:
                    agents_export = []
                    for agent in model.schedule.agents:
                        agent_data = {
                            'id': agent.unique_id,
                            'infected': agent.infected,
                            'accumulated_dose': getattr(agent, 'accumulated_dose', 0),
                            'position': agent.pos if hasattr(agent, 'pos') else None,
                            'exposure_history': getattr(agent, 'exposure_history', [])
                        }
                        agents_export.append(agent_data)
                    
                    json_str = json.dumps(agents_export, indent=2)
                    st.download_button(
                        label="⬇️ Baixar JSON Agentes",
                        data=json_str,
                        file_name=f"agents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    st.warning("Nenhum agente disponível.")

elif not st.session_state.simulation_running:
    # Tela Inicial
    st.info("👈 Configure o cenário na barra lateral e clique em '▶️ Iniciar' para começar.")
    
    st.markdown("""
    ### 🚀 Visão Geral do Simulador IAQ Avançado
    
    Este sistema simula a física de ambientes internos integrada com agentes comportamentais.
    
    **✨ Novas Funcionalidades:**
    * **🗺️ Visualização de Obstáculos:** Paredes e móveis são exibidos no mapa de calor
    * **🔍 Inspeção de Agentes:** Prontuário médico detalhado com timeline de exposição
    * **📊 Histórico Completo:** Rastreamento de dose viral, posição e atividade ao longo do tempo
    
    **Capacidades Principais:**
    * **Motor Físico:** Simula CO₂, VOCs, Vírus e Térmica usando advecção-difusão
    * **Agentes Inteligentes:** Ocupantes que respiram, se movem e podem transmitir infecções
    * **Física de Colisão:** Agentes respeitam obstáculos físicos (paredes, mesas)
    * **Intervenções:** Teste o impacto de máscaras, ventilação e redução de ocupação
    """)
    
    # Demonstração visual
    st.markdown("---")
    st.markdown("### 📚 Legenda de Obstáculos")
    
    col_legend1, col_legend2, col_legend3, col_legend4 = st.columns(4)
    
    with col_legend1:
        st.markdown("⬛ **Parede**")
        st.caption("Obstáculo sólido impermeável")
    
    with col_legend2:
        st.markdown("🟤 **Móveis**")
        st.caption("Mesas, cadeiras (semi-transparente)")
    
    with col_legend3:
        st.markdown("⬜ **Divisória**")
        st.caption("Separações leves")
    
    with col_legend4:
        st.markdown("🔵 **Equipamento**")
        st.caption("Máquinas, dispositivos")

# ============================================================================
# LÓGICA DE EXECUÇÃO DA SIMULAÇÃO
# ============================================================================

if st.session_state.simulation_running and not st.session_state.simulation_paused:
    model = st.session_state.simulation_model
    
    if model is not None and hasattr(model, 'running') and model.running:
        try:
            # Executa um passo da simulação
            model.step()
            st.session_state.current_step += 1
            
            # Coleta dados APÓS o passo
            viz_data = model.get_visualization_data() if hasattr(model, 'get_visualization_data') else {}
            st.session_state.simulation_data.append(viz_data)
            
            # Atualiza histórico de métricas
            current_metrics = getattr(model, 'current_metrics', {})
            st.session_state.simulation_history.append({
                'time': getattr(model, 'time', st.session_state.current_step * st.session_state.dt_seconds),
                'metrics': current_metrics.copy() 
            })
            
            # Verifica se terminou
            if st.session_state.current_step >= st.session_state.total_steps:
                st.session_state.simulation_running = False
                st.success("✅ Simulação Finalizada!")
            
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Erro durante a execução: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.session_state.simulation_paused = True
    
    elif model is None:
        st.session_state.simulation_running = False
        st.error("⚠️ Modelo perdido. Reinicie a simulação.")
    
    else:
        st.session_state.simulation_running = False
        st.success("✅ Simulação Finalizada!")
