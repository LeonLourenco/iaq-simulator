import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time

# Importações do Projeto
from config import (
    create_school_scenario, 
    create_office_scenario, 
    create_gym_scenario,
    ScenarioConfig,
    AgentState
)
from model import IAQModel

# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="Simulador IAQ & Epidemiológico",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS Customizados para deixar com cara de "Dashboard Científico"
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #0E1117; font-weight: 700; }
    .sub-header { font-size: 1.5rem; color: #262730; }
    .metric-card { 
        background-color: #f0f2f6; 
        border-left: 5px solid #ff4b4b; 
        padding: 15px; 
        border-radius: 5px; 
        margin-bottom: 10px;
    }
    .stProgress > div > div > div > div { background-color: #ff4b4b; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# GERENCIAMENTO DE ESTADO (SESSION STATE)
# ============================================================================
def init_session_state():
    """Inicializa variáveis persistentes."""
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'simulation_done' not in st.session_state:
        st.session_state.simulation_done = False
    if 'spatial_history' not in st.session_state:
        st.session_state.spatial_history = [] # Snapshots para o player

init_session_state()

# ============================================================================
# SIDEBAR - CONFIGURAÇÃO
# ============================================================================
def render_sidebar():
    with st.sidebar:
        st.title("⚙️ Parâmetros")
        
        # 1. Seleção de Cenário
        st.subheader("Ambiente")
        scenario_type = st.selectbox(
            "Tipo de Cenário",
            ["Escola (Sala de Aula)", "Escritório (Open Space)", "Academia (Crossfit)"]
        )
        
        # 2. Parâmetros Físicos e Sociais
        st.subheader("População & Ventilação")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            occupants = st.number_input("Ocupantes", min_value=5, max_value=200, value=30, step=5)
        with col_s2:
            infected = st.number_input("Infectados (I0)", min_value=1, max_value=10, value=1)
            
        ach = st.slider("Trocas de Ar (ACH)", 0.5, 15.0, 4.0, 0.5, help="Maior = Mais ar fresco")
        duration = st.slider("Duração (Horas)", 1, 12, 4)
        
        st.subheader("Intervenções")
        mask_compliance = st.slider("Uso de Máscaras (%)", 0, 100, 20, help="% da população usando máscara") / 100.0
        mask_eff_type = st.selectbox("Tipo de Máscara", ["Pano (30%)", "Cirúrgica (50%)", "N95 (95%)"], index=0)
        
        mask_efficiency_map = {"Pano (30%)": 0.3, "Cirúrgica (50%)": 0.5, "N95 (95%)": 0.95}
        mask_eff = mask_efficiency_map[mask_eff_type]

        st.markdown("---")
        
        # Botão de Execução
        if st.button("▶️ Rodar Simulação", type="primary", use_container_width=True):
            run_simulation_logic(
                scenario_type, occupants, infected, ach, duration, mask_compliance, mask_eff
            )

# ============================================================================
# LÓGICA DE SIMULAÇÃO
# ============================================================================
def run_simulation_logic(s_type, occupants, infected, ach, duration, mask_comp, mask_eff):
    """Configura e executa o modelo, armazenando resultados no Session State."""
    
    # 1. Factory de Cenário
    if "Escola" in s_type:
        config = create_school_scenario(occupants, infected, ach)
    elif "Escritório" in s_type:
        config = create_office_scenario(occupants, infected, ach)
    else:
        config = create_gym_scenario(occupants, infected, ach)
        
    # Sobrescreve parâmetros com inputs da UI
    config.duration_hours = duration
    config.agents.mask_compliance = mask_comp
    config.agents.mask_efficiency = mask_eff
    
    # 2. Instancia Modelo
    model = IAQModel(config)
    
    # 3. Loop de Execução com Barra de Progresso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Prepara armazenamento de snapshots espaciais (para não estourar memória, salva a cada 10 min)
    spatial_snapshots = []
    total_steps = int(duration * 60) # steps de 1 min
    snapshot_interval = 10 # minutos
    
    start_time = time.time()
    
    try:
        while model.running:
            model.step()
            
            # Atualiza UI
            curr_progress = min(model.step_count / total_steps, 1.0)
            progress_bar.progress(curr_progress)
            
            if model.step_count % 10 == 0:
                status_text.caption(f"Simulando: {model.time/3600:.1f}h / {duration}h")
            
            # Salva Snapshot Espacial (Grid Viral + Posição Agentes)
            if model.step_count % snapshot_interval == 0:
                # Copia dados leves para visualização posterior
                agents_data = []
                for a in model.schedule.agents:
                    agents_data.append({
                        "x": a.pos[0], "y": a.pos[1], "state": a.state.name
                    })
                
                spatial_snapshots.append({
                    "time_h": model.time / 3600.0,
                    "virus_grid": model.physics.get_virus_snapshot(), # numpy copy
                    "agents": agents_data
                })
                
    except Exception as e:
        st.error(f"Erro durante a simulação: {e}")
        return

    # 4. Finalização
    elapsed = time.time() - start_time
    status_text.success(f"Simulação concluída em {elapsed:.2f}s!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    
    # Persiste no Session State
    st.session_state.model = model
    st.session_state.spatial_history = spatial_snapshots
    st.session_state.simulation_done = True

# ============================================================================
# VISUALIZAÇÕES
# ============================================================================

def plot_sir_curve(history):
    """Gráfico de Linhas Interativo (Plotly)."""
    df = pd.DataFrame(history)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['time_hours'], y=df['S'], name='Suscetíveis', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['time_hours'], y=df['I'], name='Infectados', line=dict(color='red', width=3)))
    fig.add_trace(go.Scatter(x=df['time_hours'], y=df['R'], name='Recuperados', line=dict(color='green')))
    
    fig.update_layout(
        title="Dinâmica Epidemiológica (SIR)",
        xaxis_title="Tempo (Horas)",
        yaxis_title="Pessoas",
        hovermode="x unified",
        template="plotly_white",
        height=400
    )
    return fig

def plot_spatial_replay(model):
    """
    Heatmap Animado (Slider) da Concentração Viral + Agentes.
    Usa os snapshots salvos para performance.
    """
    snapshots = st.session_state.spatial_history
    if not snapshots:
        st.warning("Sem dados espaciais disponíveis.")
        return

    # Slider para selecionar o tempo
    times = [s['time_h'] for s in snapshots]
    max_time = max(times)
    selected_time = st.slider("Navegar no Tempo (Horas)", 0.0, max_time, max_time, step=0.1)
    
    # Encontra o snapshot mais próximo
    idx = min(range(len(times)), key=lambda i: abs(times[i]-selected_time))
    snap = snapshots[idx]
    
    # 1. Heatmap Viral
    # Transpose porque numpy é (y,x) e plotly gosta de (x,y)
    virus_matrix = snap['virus_grid'].T 
    
    fig = go.Figure()
    
    # Adiciona Heatmap
    fig.add_trace(go.Heatmap(
        z=virus_matrix,
        colorscale='Reds',
        zmin=0,
        zmax=np.max([s['virus_grid'].max() for s in snapshots]), # Escala fixa
        opacity=0.7,
        name="Vírus (quanta/m³)"
    ))
    
    # 2. Scatter dos Agentes
    agents = snap['agents']
    
    # Separa por cor/estado
    colors = {'SUSCEPTIBLE': 'green', 'INFECTED': 'red', 'RECOVERED': 'gray'}
    
    for state, color in colors.items():
        xs = [a['x'] for a in agents if a['state'] == state]
        ys = [a['y'] for a in agents if a['state'] == state]
        
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode='markers',
                marker=dict(color=color, size=10, line=dict(width=1, color='black')),
                name=state.title()
            ))

    # Layout Físico
    fig.update_layout(
        title=f"Distribuição Viral e Posição dos Agentes - T={selected_time:.1f}h",
        xaxis=dict(title="Largura (Células)", showgrid=False),
        yaxis=dict(title="Profundidade (Células)", showgrid=False, scaleanchor="x"),
        width=700,
        height=600,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_risk_analysis(model):
    """Histograma de Doses e Métricas Avançadas."""
    agents = model.schedule.agents
    doses = [a.accumulated_dose for a in agents if a.state != AgentState.INFECTED] # Doses dos não-pacientes zero
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.histogram(
            x=doses, 
            nbins=20, 
            labels={'x': 'Dose Acumulada (quanta)', 'y': 'Contagem'},
            title="Distribuição de Risco (Dose Inalada)",
            color_discrete_sequence=['#ff4b4b']
        )
        # Linha do ID50
        fig.add_vline(x=50, line_dash="dash", line_color="black", annotation_text="ID50 (Risco Médio)")
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("### Estatísticas de Risco")
        if doses:
            avg_dose = np.mean(doses)
            max_dose = np.max(doses)
            high_risk_count = sum(1 for d in doses if d > 50) # Acima do ID50
            
            st.metric("Dose Média", f"{avg_dose:.2f} q")
            st.metric("Dose Máxima", f"{max_dose:.2f} q")
            st.metric("Pessoas em Alto Risco (>ID50)", f"{high_risk_count}")
        else:
            st.info("Sem dados de dose suficientes.")

# ============================================================================
# MAIN
# ============================================================================
def main():
    st.markdown('<h1 class="main-header">🦠 Simulador de Transmissão Aérea</h1>', unsafe_allow_html=True)
    st.caption("Universidade Federal Rural de Pernambuco (UFRPE) - Epidemiologia Computacional")
    
    render_sidebar()
    
    if st.session_state.simulation_done and st.session_state.model:
        model = st.session_state.model
        
        # --- CARDS DE MÉTRICAS (KPIs) ---
        counts = model.get_state_counts()
        initial_pop = model.config.agents.total_occupants
        attack_rate = (counts['INFECTED'] + counts['RECOVERED'] - model.config.agents.initial_infected) / initial_pop
        
        # Layout de 4 colunas
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Taxa de Ataque", f"{attack_rate*100:.1f}%", help="% de novos infectados")
        c2.metric("Infectados Ativos", counts['INFECTED'])
        c3.metric("Suscetíveis Restantes", counts['SUSCEPTIBLE'])
        c4.metric("Pico de Vírus (Ar)", f"{np.max([s['virus_grid'].max() for s in st.session_state.spatial_history]):.1f} q/m³")
        
        # --- ABAS DE ANÁLISE ---
        tab1, tab2, tab3 = st.tabs(["📊 Dinâmica SIR", "🗺️ Mapa Espacial (CFD)", "🔬 Análise de Risco"])
        
        with tab1:
            st.plotly_chart(plot_sir_curve(model.metrics_history), use_container_width=True)
            
        with tab2:
            st.markdown("Visualize como o vírus se espalhou pelo ambiente físico e como os agentes se moveram.")
            plot_spatial_replay(model)
            
        with tab3:
            plot_risk_analysis(model)
            
        # Download Data
        st.divider()
        csv = pd.DataFrame(model.metrics_history).to_csv(index=False).encode('utf-8')
        st.download_button("📥 Baixar Dados da Simulação (CSV)", csv, "iaq_simulation_data.csv", "text/csv")
        
    else:
        # TELA DE BOAS VINDAS
        st.info("👈 Configure o cenário na barra lateral e clique em 'Rodar Simulação' para começar.")
        
        st.markdown("""
        ### Sobre o Projeto
        Este simulador utiliza **Modelagem Baseada em Agentes (ABM)** acoplada a **Dinâmica de Fluidos Computacional (CFD)** simplificada para estimar o risco de transmissão aérea de doenças (como COVID-19 ou Influenza) em ambientes internos.
        
        **Modelos utilizados:**
        * **Epidemiologia:** Wells-Riley & SIR espacial.
        * **Física:** Advecção-Difusão (Lei de Fick).
        * **Comportamento:** Máquina de estados para movimentação humana.
        """)

if __name__ == "__main__":
    main()