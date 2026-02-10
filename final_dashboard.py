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
</style>
""", unsafe_allow_html=True)

# Inicialização do estado da sessão
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
        'dt_seconds': 60
    }
    
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Cabeçalho
st.markdown('<h1 class="main-header">🏢 Simulador IAQ - Dashboard</h1>', unsafe_allow_html=True)

# --- BARRA LATERAL ---
with st.sidebar:
    st.markdown("### ⚙️ Configuração do Cenário")
    
    # Seleção de cenário
    scenario_options = {
        "🏫 Escola (Sala de Aula)": "school",
        "🏢 Escritório (Open Space)": "office", 
        "💪 Academia (Fitness Center)": "gym",
        "🏥 Hospital (Quarto)": "hospital",
        "🏠 Residencial (Casa)": "residential"
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
            scenario = cfg.get_scenario_config(st.session_state.selected_scenario)
            
            # 2. Atualizar com inputs da UI
            scenario.total_occupants = total_occupants
            scenario.initial_infected_ratio = initial_infected_ratio
            scenario.agent_config.mask_wearing_prob = mask_usage_rate / 100.0
            
            # Ajuste de zonas
            for zone in scenario.zones:
                zone.target_ach = ach_target
                
            scenario.co2_setpoint = co2_setpoint
            scenario.temperature_setpoint = temperature_setpoint
            scenario.humidity_setpoint = humidity_setpoint
            scenario.overall_ventilation_strategy = ventilation_strategy
            
            # 3. Config física
            physics_config = cfg.PhysicsConfig(
                cell_size=cell_size,
                kalman_enabled=enable_kalman,
                pem_correction_active=enable_plume_correction,
                kalman_update_interval=60
            )
            
            # 4. Criar Modelo
            st.session_state.simulation_model = IAQSimulationModel(
                scenario=scenario,
                physics_config=physics_config,
                simulation_duration_hours=simulation_hours,
            )
            
            # Definir condições externas no motor físico
            st.session_state.simulation_model.physics.set_external_conditions(
                temperature_c=outdoor_temp,
                humidity_percent=outdoor_humidity,
                co2_ppm=400
            )
            
            # Resetar contadores
            st.session_state.dt_seconds = time_step
            st.session_state.total_steps = int(simulation_hours * 3600 / time_step)
            st.session_state.current_step = 0
            st.session_state.simulation_data = []
            st.session_state.simulation_history = []
            st.session_state.active_interventions = []
            
            return True
            
        except Exception as e:
            st.error(f"Erro na inicialização: {e}")
            return False

    with col_start:
        if st.button("▶️ Iniciar", type="primary", disabled=st.session_state.simulation_running):
            if initialize_simulation():
                st.session_state.simulation_running = True
                st.session_state.simulation_paused = False
                st.rerun()
    
    with col_pause:
        pause_label = "Retomar" if st.session_state.simulation_paused else "Pausar"
        if st.button(f"⏸️ {pause_label}", disabled=not st.session_state.simulation_running):
            st.session_state.simulation_paused = not st.session_state.simulation_paused
            st.rerun()
            
    with col_reset:
        if st.button("🔄 Reset"):
            st.session_state.simulation_running = False
            st.session_state.simulation_paused = False
            st.session_state.simulation_model = None
            st.session_state.simulation_data = []
            st.rerun()

    if st.session_state.simulation_running and st.session_state.simulation_model:
        st.markdown("---")
        st.markdown("### 🛡️ Intervenções")
        
        intervention_map = {
            "Aumentar Ventilação (50%)": ("increase_ventilation", {"factor": 1.5}),
            "Máscaras Obrigatórias": ("mask_mandate", {"compliance": 0.95}),
            "Reduzir Ocupação (30%)": ("reduce_occupancy", {"reduction": 0.3})
        }
        
        sel_intervention = st.selectbox("Ação", list(intervention_map.keys()))
        
        if st.button("Aplicar Intervenção"):
            tipo, params = intervention_map[sel_intervention]
            st.session_state.simulation_model.apply_interventions(tipo, params)
            st.session_state.active_interventions.append({
                "nome": sel_intervention,
                "step": st.session_state.current_step
            })
            st.success(f"{sel_intervention} aplicada!")

# --- PARTE 1: VISUALIZAÇÃO ---

if st.session_state.simulation_model is not None:
    model = st.session_state.simulation_model
    
    # Barra de Progresso
    if st.session_state.total_steps > 0:
        prog = min(st.session_state.current_step / st.session_state.total_steps, 1.0)
        st.progress(prog)
        st.caption(f"Tempo Simulado: {model.time/3600:.2f}h / {st.session_state.dt_seconds}s por update")

    # Métricas Atuais (KPIs)
    metrics = model.current_metrics
    
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    with kpi1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("CO₂ Médio", f"{metrics.get('average_co2', 400):.0f} ppm", 
                 delta=f"{metrics.get('average_co2', 400) - 400:.0f} vs ext", delta_color="inverse")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with kpi2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        risk = metrics.get('infection_risk', 0) * 100
        st.metric("Risco Infecção", f"{risk:.2f}%", 
                 delta="Crítico" if risk > 5 else "Baixo", delta_color="inverse")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with kpi3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        total_agents = len(model.simulation_agents) if hasattr(model, 'simulation_agents') else 0
        inf_agents = metrics.get('infected_agents', 0)
        st.metric("Agentes Infectados", f"{inf_agents}", f"Total: {total_agents}")
        st.markdown('</div>', unsafe_allow_html=True)

    with kpi4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        r_eff = metrics.get('r_effective', 0)
        st.metric("R-Efetivo", f"{r_eff:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with kpi5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        energy = metrics.get('energy_consumption', 0)
        st.metric("Energia Acumulada", f"{energy:.2f} kWh")
        st.markdown('</div>', unsafe_allow_html=True)

    # Abas de Visualização
    tab_charts, tab_map, tab_zones, tab_export = st.tabs(["📈 Gráficos Temporais", "🗺️ Mapa de Calor", "🏢 Detalhe Zonas", "💾 Exportar"])

    # --- ABA 1: GRÁFICOS ---
    with tab_charts:
        if len(st.session_state.simulation_history) > 1:
            df_hist = pd.DataFrame([
                {'time_h': h['time']/3600, **h['metrics']} 
                for h in st.session_state.simulation_history
            ])
            
            fig = make_subplots(rows=2, cols=2, subplot_titles=("CO₂ (ppm)", "Risco de Infecção", "HCHO (ppb)", "Temperatura e Umidade"))
            
            # CO2
            fig.add_trace(go.Scatter(x=df_hist['time_h'], y=df_hist['average_co2'], name="CO₂", line=dict(color='orange')), row=1, col=1)
            # Risco
            fig.add_trace(go.Scatter(x=df_hist['time_h'], y=df_hist['infection_risk'], name="Risco", line=dict(color='red')), row=1, col=2)
            # HCHO
            fig.add_trace(go.Scatter(x=df_hist['time_h'], y=df_hist['average_hcho'], name="HCHO", line=dict(color='purple')), row=2, col=1)
            
            # Temp/Hum
            if 'average_temperature' in df_hist.columns:
                fig.add_trace(go.Scatter(x=df_hist['time_h'], y=df_hist['average_temperature'], name="Temp (°C)"), row=2, col=2)
            if 'average_humidity' in df_hist.columns:
                fig.add_trace(go.Scatter(x=df_hist['time_h'], y=df_hist['average_humidity']*100, name="Umid (%)", line=dict(dash='dot')), row=2, col=2)
            
            fig.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Iniciando coleta de dados...")

    # --- ABA 2: MAPA DE CALOR ---
    with tab_map:
        if st.session_state.simulation_data:
            last_frame = st.session_state.simulation_data[-1]
            grids = last_frame.get('physics', {}).get('grids', {})
            
            if 'co2_ppm' in grids:
                co2_grid = np.array(grids['co2_ppm'])
                
                # Mapa de Calor 2D
                fig_map = go.Figure(data=go.Heatmap(
                    z=co2_grid,
                    colorscale='Jet',
                    colorbar=dict(title='CO₂ (ppm)')
                ))
                
                # Adicionar Agentes
                agents = last_frame.get('agents', {})
                if agents.get('positions'):
                    pos = np.array(agents['positions'])
                    inf = np.array(agents['infected'])
                    
                    if len(pos) > 0:
                        fig_map.add_trace(go.Scatter(
                            x=pos[:, 0], 
                            y=pos[:, 1],
                            mode='markers',
                            marker=dict(
                                color=['red' if i else '#00FF00' for i in inf],
                                size=8,
                                line=dict(width=1, color='black')
                            ),
                            name='Ocupantes'
                        ))
                
                fig_map.update_layout(
                    title="Mapa de Concentração de CO₂ e Ocupantes",
                    height=600,
                    yaxis=dict(scaleanchor="x", scaleratio=1)
                )
                st.plotly_chart(fig_map, width="stretch")
                
                st.caption("Nota: Cores dos pontos indicam saúde (Verde=Saudável, Vermelho=Infectado). Fundo indica CO₂.")
            else:
                st.warning("Dados de grid ainda não gerados.")

    # --- ABA 3: ZONAS ---
    with tab_zones:
        if st.session_state.simulation_data:
            last_frame = st.session_state.simulation_data[-1]
            zone_stats = last_frame.get('physics', {}).get('zone_stats', {})
            
            if zone_stats:
                zone_rows = []
                for zid, data in zone_stats.items():
                    concs = data.get('concentrations', {})
                    zone_rows.append({
                        "Zona": data['name'],
                        "Ocupantes": f"{data.get('max_occupants', 0)} (Max)",
                        "CO₂ (ppm)": f"{concs.get('co2_ppm_mean', 0):.0f}",
                        "Temp (°C)": f"{concs.get('temperature_c_mean', 0):.1f}",
                        "ACH (Trocas/h)": f"{data.get('ach_actual', 0):.2f}"
                    })
                
                st.dataframe(pd.DataFrame(zone_rows), width="stretch")
                
                zones = [z['Zona'] for z in zone_rows]
                co2_vals = [float(z['CO₂ (ppm)']) for z in zone_rows]
                fig_bar = go.Figure([go.Bar(x=zones, y=co2_vals, marker_color='orange')])
                fig_bar.update_layout(title="Comparativo de CO₂ por Zona", yaxis_title="ppm")
                st.plotly_chart(fig_bar, width="stretch")

    # --- ABA 4: EXPORTAR ---
    with tab_export:
        st.markdown("### 📥 Download dos Dados")
        col_json, col_csv = st.columns(2)
        
        with col_json:
            if st.button("Gerar JSON Completo"):
                json_str = model.export_simulation_data('json')
                st.download_button(
                    label="⬇️ Baixar JSON",
                    data=json_str,
                    file_name=f"iaq_sim_{datetime.now().strftime('%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col_csv:
            if st.button("Gerar CSV Histórico"):
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
                        file_name=f"iaq_history_{datetime.now().strftime('%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("Sem histórico para exportar.")

elif not st.session_state.simulation_running:
    # Tela Inicial
    st.info("👈 Configure o cenário na barra lateral e clique em 'Iniciar' para começar.")
    
    st.markdown("""
    ### 🚀 Visão Geral do Simulador
    
    Este sistema simula a física de ambientes internos integrada com agentes comportamentais.
    
    **Capacidades:**
    * **Motor Físico:** Simula CO₂, VOCs, Vírus e Térmica usando advecção-difusão.
    * **Agentes:** Ocupantes que respiram, se movem e podem transmitir infecções.
    * **Intervenções:** Teste o impacto de máscaras, ventilação e redução de ocupação em tempo real.
    """)

# --- PARTE 2: LÓGICA DE EXECUÇÃO ---

if st.session_state.simulation_running and not st.session_state.simulation_paused:
    model = st.session_state.simulation_model
    
    if model is not None and model.running:
        try:
            # Executa a simulação
            target_time = model.time + st.session_state.dt_seconds
            
            while model.time < target_time and model.running:
                model.step()
                
            st.session_state.current_step += 1
            
            # Coleta dados APÓS o passo
            viz_data = model.get_visualization_data()
            st.session_state.simulation_data.append(viz_data)
            
            st.session_state.simulation_history.append({
                'time': model.time,
                'metrics': model.current_metrics.copy() 
            })
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Erro durante a execução: {e}")
            st.session_state.simulation_paused = True
    elif model is None:
        st.session_state.simulation_running = False
        st.error("Modelo perdido. Reinicie a simulação.")
    else:
        st.session_state.simulation_running = False
        st.success("Simulação Finalizada!")
