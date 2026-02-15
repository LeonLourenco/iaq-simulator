"""
Dashboard Interativo para Simulador IAQ & Epidemiológico.

Funcionalidades:
1. Execução "Live": Visualização em tempo real enquanto o modelo calcula.
2. Análise "Post-Mortem": Abas detalhadas, Replay temporal, Análise de Risco e KPIs.
3. Controles Completos: Cenários, Máscaras, Ventilação e Duração.

Autor: Leon Lourenço da Silva Santos
Disciplina: Epidemiologia Computacional - UFRPE
"""

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
    AgentState
)
from model import IAQModel

# ============================================================================
# 1. CONFIGURAÇÃO DA PÁGINA E ESTILOS
# ============================================================================
st.set_page_config(
    page_title="Simulador IAQ - UFRPE",
    page_icon="☣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Ajuste de margens para aproveitar melhor a tela */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    /* Destaque para métricas */
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. GERENCIAMENTO DE ESTADO (SESSION STATE)
# ============================================================================
def init_session_state():
    """Inicializa variáveis de estado persistentes."""
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'simulation_done' not in st.session_state:
        st.session_state.simulation_done = False
    if 'spatial_history' not in st.session_state:
        st.session_state.spatial_history = []

init_session_state()

# ============================================================================
# 3. FUNÇÕES DE VISUALIZAÇÃO (GRÁFICOS)
# ============================================================================

def create_map_figure(virus_grid, agents_data, title_prefix="Tempo Real"):
    """
    Gera o mapa combinado (Heatmap + Scatter) usado tanto no Live quanto no Replay.
    """
    # Prepara dados dos agentes
    xs, ys, colors, symbols, texts = [], [], [], [], []
    
    # Mapeamento de cores e símbolos (Alto contraste para fundo escuro)
    style_map = {
        "SUSCEPTIBLE": ("#00FF00", "circle"),   # Verde Neon
        "INFECTED":    ("#FF0000", "x"),        # Vermelho Puro
        "RECOVERED":   ("#00FFFF", "square")    # Ciano
    }

    for agent in agents_data:
        # Suporta tanto objeto Agent quanto dict (do snapshot)
        is_obj = hasattr(agent, 'pos')
        x = (agent.pos[0] if is_obj else agent['x']) + 0.5
        y = (agent.pos[1] if is_obj else agent['y']) + 0.5
        state = agent.state.name if is_obj else agent['state']
        
        # Tooltip rico
        dose = agent.accumulated_dose if is_obj else agent.get('dose', 0)
        uid = agent.unique_id if is_obj else agent.get('id', '?')
        
        c, s = style_map.get(state, ("white", "circle"))
        
        xs.append(x)
        ys.append(y)
        colors.append(c)
        symbols.append(s)
        texts.append(f"<b>ID:</b> {uid}<br><b>Estado:</b> {state}<br><b>Dose:</b> {dose:.3f} q")

    # --- LÓGICA DE ESCALA DINÂMICA ---
    # Calcula o máximo atual da matriz para ajustar a sensibilidade
    current_max = np.max(virus_grid)
    
    # Define o teto da escala:
    # Se o máximo for muito baixo (< 0.5), forçamos 0.5 para não visualizar ruído numérico.
    # Caso contrário, usamos o próprio máximo da simulação atual.
    z_limit = current_max if current_max > 0.5 else 0.5

    fig = go.Figure()

    # Camada 1: Ar (Heatmap)
    fig.add_trace(go.Heatmap(
        z=virus_grid.T, 
        colorscale='Magma',  
        zmin=0, 
        zmax=z_limit,
        opacity=0.7,       
        showscale=True,
        zsmooth='best',
        colorbar=dict(
            title=dict(text="q/m³", side="right"),
            thickness=15, 
            len=0.7
        ),
        hovertemplate='Concentração: %{z:.2f} q/m³<extra></extra>'
    ))

    # Camada 2: Agentes
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode='markers',
        marker=dict(
            size=14, 
            color=colors, 
            symbol=symbols, 
            line=dict(width=1, color='white')
        ),
        text=texts,
        hoverinfo='text',
        name='Agentes'
    ))

    # Layout
    fig.update_layout(
        title=dict(
            text=f"{title_prefix} (Escala máx: {z_limit:.2f} q/m³)", 
            x=0.02, 
            y=0.98, 
            font=dict(color="white", size=16)
        ),
        xaxis=dict(showgrid=False, visible=False, range=[0, virus_grid.shape[0]]),
        yaxis=dict(showgrid=False, visible=False, scaleanchor="x", range=[0, virus_grid.shape[1]]),
        margin=dict(l=5, r=5, t=35, b=5),
        height=520,
        paper_bgcolor='#262730',
        plot_bgcolor='#000000',
        showlegend=False
    )
    return fig

def plot_risk_histogram(model):
    """Histograma de Risco baseado em doses acumuladas."""
    agents = model.schedule.agents
    doses = [
        a.accumulated_dose 
        for a in agents 
        if a.unique_id >= model.config.agents.initial_infected
    ]
    
    if not doses: 
        return None

    fig = px.histogram(
        x=doses, 
        nbins=20,
        labels={'x': 'Dose Inalada (quanta)', 'y': 'Nº de Pessoas'},
        title="Distribuição de Risco Populacional",
        color_discrete_sequence=['#EF553B']
    )
    
    fig.add_vline(
        x=50.0, 
        line_dash="dash", 
        line_color="black", 
        annotation_text="ID50 (Risco Médio)",
        annotation_position="top"
    )
    
    fig.update_layout(
        height=320, 
        margin=dict(l=20, r=20, t=50, b=20),
        template="plotly_white"
    )
    return fig

def create_sir_chart(history_df):
    """Gráfico de evolução SIR otimizado."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=history_df['time_hours'], 
        y=history_df['S'], 
        name='Suscetíveis', 
        line=dict(color='#00CC96', width=2),
        fill='tozeroy',
        fillcolor='rgba(0,204,150,0.1)'
    ))
    
    fig.add_trace(go.Scatter(
        x=history_df['time_hours'], 
        y=history_df['I'], 
        name='Infectados', 
        line=dict(color='#EF553B', width=3),
        fill='tozeroy',
        fillcolor='rgba(239,85,59,0.1)'
    ))
    
    fig.add_trace(go.Scatter(
        x=history_df['time_hours'], 
        y=history_df['R'], 
        name='Recuperados', 
        line=dict(color='#636EFA', width=2),
        fill='tozeroy',
        fillcolor='rgba(99,110,250,0.1)'
    ))
    
    fig.update_layout(
        height=420, 
        template="plotly_white", 
        xaxis_title="Tempo (Horas)", 
        yaxis_title="Número de Pessoas",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
    )
    return fig

# ============================================================================
# 4. APLICAÇÃO PRINCIPAL
# ============================================================================
def main():
    # --- HEADER ---
    st.markdown('<h1 class="main-header">🦠 Simulador IAQ: Análise de Risco Viral em Ambientes Fechados</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Modelagem Baseada em Agentes (ABM) + Dinâmica de Fluidos Computacional (CFD) | UFRPE</p>', unsafe_allow_html=True)

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("⚙️ Configuração da Simulação")
        
        # 1. Cenário
        st.subheader("🏢 Ambiente")
        scenario_type = st.selectbox(
            "Tipo de Cenário", 
            ["Escola (4h)", "Escritório (8h)", "Academia (2h)"],
            help="Cada cenário tem comportamentos e layouts específicos"
        )
        
        col1, col2 = st.columns(2)
        occupants = col1.number_input("Ocupantes", 10, 100, 30, step=5)
        infected = col2.number_input("Infectados (I₀)", 1, 10, 1)
        
        # 2. Física
        st.subheader("🌬️ Ventilação & Tempo")
        ach = st.slider(
            "Trocas de Ar (ACH)", 
            0.5, 12.0, 4.0, 0.5,
            help="0.5 = Janelas fechadas | 4 = Ventilação normal | 12 = Hospitalar"
        )
        
        # Duração customizável
        default_dur = {"Escola": 4, "Escritório": 8, "Academia": 2}
        dur_default = default_dur.get(scenario_type.split()[0], 4)
        duration = st.number_input("Duração (Horas)", 1, 24, dur_default)
        
        # 3. Intervenções
        st.subheader("😷 Medidas de Proteção")
        mask_compliance = st.slider("Adesão a Máscaras (%)", 0, 100, 50) / 100.0
        mask_type = st.selectbox(
            "Tipo de Máscara", 
            ["Pano (30%)", "Cirúrgica (50%)", "N95 (95%)"], 
            index=1
        )
        mask_eff_map = {"Pano (30%)": 0.3, "Cirúrgica (50%)": 0.5, "N95 (95%)": 0.95}
        mask_efficiency = mask_eff_map[mask_type]
        
        st.divider()
        
        # Controle de velocidade
        speed = st.select_slider(
            "Velocidade Visual", 
            options=["🐢 Lento", "▶️ Normal", "⚡ Turbo"], 
            value="▶️ Normal"
        )
        sleep_map = {"🐢 Lento": 0.2, "▶️ Normal": 0.05, "⚡ Turbo": 0.0}
        sleep_time = sleep_map[speed]
        
        st.divider()
        start_btn = st.button("▶️ INICIAR SIMULAÇÃO", type="primary", use_container_width=True)

    # ========================================================================
    # LÓGICA DE EXECUÇÃO (LOOP VIVO + PERSISTÊNCIA)
    # ========================================================================
    if start_btn:
        # 1. Configuração do Modelo
        if "Escola" in scenario_type:
            config = create_school_scenario(occupants, infected, ach)
        elif "Escritório" in scenario_type:
            config = create_office_scenario(occupants, infected, ach)
        else:
            config = create_gym_scenario(occupants, infected, ach)
            
        # Sobrescreve com valores da UI
        config.duration_hours = duration
        config.agents.mask_compliance = mask_compliance
        config.agents.mask_efficiency = mask_efficiency
        
        model = IAQModel(config)
        
        # Preparação Visual
        progress_bar = st.progress(0, text="Inicializando...")
        status_text = st.empty()
        
        col_live_map, col_live_metrics = st.columns([2, 1])
        with col_live_map:
            st.caption("🗺️ Mapa em Tempo Real")
            map_placeholder = st.empty()
        with col_live_metrics:
            st.caption("📊 Métricas Instantâneas")
            kpi_placeholder = st.empty()
            st.caption("⚠️ Alertas de Transmissão")
            log_placeholder = st.empty()
            
        # Armazenamento para Replay
        snapshots = []
        total_steps = int(duration * 3600 / 10.0)  # dt=10s
        
        # Intervalo dinâmico
        snapshot_interval = max(5, min(50, total_steps // 40))
        
        # 2. LOOP DE SIMULAÇÃO (EXECUÇÃO LIVE)
        start_time_wall = time.time()
        
        try:
            while model.running:
                model.step()
                
                # Frequência de atualização visual
                update_freq = 10 if "Turbo" in speed else 2
                
                if model.step_count % update_freq == 0:
                    # A. Renderizar Mapa
                    fig = create_map_figure(
                        model.physics.virus_grid, 
                        model.schedule.agents, 
                        title_prefix=f"⏱️ Tempo: {model.time/3600:.2f}h / {duration}h"
                    )
                    map_placeholder.plotly_chart(fig, key=f"live_{model.step_count}")
                    
                    # B. Métricas Rápidas
                    counts = model.get_state_counts()
                    new_cases = counts['INFECTED'] + counts['RECOVERED'] - infected
                    max_conc = model.physics.virus_grid.max()
                    
                    kpi_placeholder.markdown(f"""
                    <div style='background: #f0f2f6; padding: 15px; border-radius: 8px; border-left: 4px solid #ff4b4b;'>
                        <h4 style='margin: 0 0 10px 0; color: #333;'>Status Atual</h4>
                        <p style='margin: 5px 0;'><b>Novos Casos:</b> <span style='color: #ff4b4b; font-size: 1.3em;'>{new_cases}</span></p>
                        <p style='margin: 5px 0;'><b>Infectados Ativos:</b> {counts['INFECTED']}</p>
                        <p style='margin: 5px 0;'><b>Pico Viral:</b> {max_conc:.2f} q/m³</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # C. Log de Alertas
                    if model.infection_log:
                        last = model.infection_log[-1]
                        time_diff = model.time/3600 - last['time_h']
                        
                        if time_diff < 0.1:  # Últimos 6 minutos
                            log_placeholder.error(
                                f"☣️ **CONTÁGIO DETECTADO!**\n\n"
                                f"- Agente: `{last['victim_id']}`\n"
                                f"- Dose Crítica: `{last['dose_at_infection']:.2f} q`\n"
                                f"- Local: {last['location']}"
                            )
                        else:
                            log_placeholder.success("✅ Nenhuma transmissão recente")
                    else:
                        log_placeholder.info("🔍 Monitorando transmissões...")
                    
                    # D. Barra de Progresso
                    prog = min(model.step_count / total_steps, 1.0)
                    progress_bar.progress(
                        prog, 
                        text=f"Simulando: {prog*100:.1f}% completo"
                    )
                    
                    # Sleep para visualização
                    if sleep_time > 0: 
                        time.sleep(sleep_time)
                
                # 3. Salvar Snapshot para Replay
                if model.step_count % snapshot_interval == 0:
                    agents_snap = [
                        {
                            'x': a.pos[0], 
                            'y': a.pos[1], 
                            'state': a.state.name, 
                            'dose': a.accumulated_dose, 
                            'id': a.unique_id
                        } 
                        for a in model.schedule.agents
                    ]
                    snapshots.append({
                        'time_h': model.time / 3600.0,
                        'virus_grid': np.copy(model.physics.virus_grid),
                        'agents': agents_snap
                    })
        
        except Exception as e:
            st.error(f"❌ Erro durante a simulação: {e}")
            st.stop()

        # Fim do Loop
        elapsed = time.time() - start_time_wall
        status_text.success(f"✅ Simulação concluída em **{elapsed:.1f}s** ({model.step_count} passos)")
        time.sleep(1.5)
        
        # Limpeza de placeholders
        progress_bar.empty()
        status_text.empty()
        map_placeholder.empty()
        kpi_placeholder.empty()
        log_placeholder.empty()
        
        # Salva no Session State
        st.session_state.model = model
        st.session_state.spatial_history = snapshots
        st.session_state.simulation_done = True
        st.rerun()

    # ========================================================================
    # ANÁLISE PÓS-SIMULAÇÃO (RESULTADOS PERSISTENTES)
    # ========================================================================
    if st.session_state.simulation_done and st.session_state.model:
        model = st.session_state.model
        snapshots = st.session_state.spatial_history
        
        st.divider()
        st.markdown("## 📈 Resultados da Simulação")
        
        # --- 1. KPI CARDS ---
        counts = model.get_state_counts()
        initial_infected = model.config.agents.initial_infected
        total_occupants = model.config.agents.total_occupants
        new_cases = counts['INFECTED'] + counts['RECOVERED'] - initial_infected
        
        # Cálculo seguro da taxa de ataque
        susceptible_pop = total_occupants - initial_infected
        if susceptible_pop > 0:
            attack_rate = (new_cases / susceptible_pop) * 100
        else:
            attack_rate = 0
            
        peak_viral_load = np.max([s['virus_grid'].max() for s in snapshots]) if snapshots else 0
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "Taxa de Ataque", 
            f"{attack_rate:.1f}%", 
            delta=f"{new_cases} casos",
            delta_color="inverse",
            help="Proporção de suscetíveis que se infectaram"
        )
        k2.metric(
            "Total de Novos Casos", 
            f"{new_cases}",
            help="Infecções secundárias"
        )
        k3.metric(
            "Pico de Carga Viral", 
            f"{peak_viral_load:.1f} q/m³",
            help="Concentração máxima detectada no ar"
        )
        k4.metric(
            "Duração Real", 
            f"{model.time/3600:.2f}h",
            delta=f"{model.step_count} steps",
            help="Tempo total simulado"
        )
        
        # --- 2. SISTEMA DE ABAS ---
        tab1, tab2, tab3, tab4 = st.tabs([
            "🗺️ Replay Espacial", 
            "📊 Dinâmica SIR", 
            "🔬 Análise de Risco", 
            "📋 Rastreamento de Contatos"
        ])
        
        # ABA 1: REPLAY TEMPORAL
        with tab1:
            st.markdown("### 🎬 Replay da Evolução Viral no Ambiente")
            
            if snapshots:
                times = [s['time_h'] for s in snapshots]
                max_t = max(times)
                
                col_slider, col_info = st.columns([3, 1])
                
                with col_slider:
                    t_selected = st.slider(
                        "Linha do Tempo (Horas)", 
                        0.0, max_t, max_t, 
                        step=0.05,
                        format="%.2f h"
                    )
                
                # Busca snapshot mais próximo
                idx = min(range(len(times)), key=lambda i: abs(times[i]-t_selected))
                snap = snapshots[idx]
                
                with col_info:
                    st.metric("Snapshot", f"{idx+1}/{len(snapshots)}")
                    st.metric("Tempo Exato", f"{snap['time_h']:.3f}h")
                
                fig_replay = create_map_figure(
                    snap['virus_grid'], 
                    snap['agents'], 
                    title_prefix=f"📍 Replay T={snap['time_h']:.2f}h"
                )
                st.plotly_chart(fig_replay, use_container_width=True)
                
                # Estatísticas do frame
                st.caption("**Dados do Frame Selecionado:**")
                frame_stats = {
                    "S": sum(1 for a in snap['agents'] if a['state'] == 'SUSCEPTIBLE'),
                    "I": sum(1 for a in snap['agents'] if a['state'] == 'INFECTED'),
                    "R": sum(1 for a in snap['agents'] if a['state'] == 'RECOVERED'),
                    "Conc. Média": f"{snap['virus_grid'].mean():.3f} q/m³"
                }
                st.json(frame_stats)
            else:
                st.warning("⚠️ Histórico espacial insuficiente para replay.")

        # ABA 2: CURVAS EPIDEMIOLÓGICAS
        with tab2:
            st.markdown("### 📈 Evolução Epidemiológica (Modelo SIR)")
            
            hist_df = pd.DataFrame(model.metrics_history)
            fig_sir = create_sir_chart(hist_df)
            st.plotly_chart(fig_sir, use_container_width=True)
            
            # Tabela de resumo
            st.markdown("#### Resumo Estatístico")
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Tempo até 1º Caso", f"{hist_df[hist_df['I'] > initial_infected]['time_hours'].iloc[0]:.2f}h" if len(hist_df[hist_df['I'] > initial_infected]) > 0 else "N/A")
            
            with col_b:
                peak_i = hist_df['I'].max()
                peak_time = hist_df[hist_df['I'] == peak_i]['time_hours'].iloc[0]
                st.metric("Pico de Infectados", f"{peak_i} pessoas", delta=f"em {peak_time:.1f}h")
            
            with col_c:
                final_r = hist_df['R'].iloc[-1]
                st.metric("Recuperados Finais", f"{final_r}")

        # ABA 3: ANÁLISE DE RISCO
        with tab3:
            st.markdown("### 🔬 Análise de Risco por Exposição")
            
            fig_risk = plot_risk_histogram(model)
            if fig_risk:
                st.plotly_chart(fig_risk, use_container_width=True)
            else:
                st.info("Sem dados de dose acumulada para análise.")
            
            # Tabela de Risco por Perfil Comportamental
            st.markdown("#### 🎭 Risco por Perfil de Comportamento")
            
            risk_data = []
            for a in model.schedule.agents:
                behavior_name = getattr(a.behavior, '__class__', type('Unknown', (), {})).__name__
                
                risk_data.append({
                    "Perfil": behavior_name,
                    "Dose Acumulada": a.accumulated_dose,
                    "Estado Final": a.state.name,
                    "Infectado": "Sim" if a.state != AgentState.SUSCEPTIBLE else "Não"
                })
            
            df_risk = pd.DataFrame(risk_data)
            
            risk_summary = df_risk.groupby("Perfil").agg({
                "Dose Acumulada": "mean",
                "Infectado": lambda x: f"{(x=='Sim').sum()}/{len(x)}"
            }).round(2)
            
            risk_summary.columns = ["Dose Média (q)", "Taxa de Infecção"]
            st.dataframe(risk_summary, use_container_width=True)
            
            csv_risk = df_risk.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Baixar Dados de Risco (CSV)",
                csv_risk,
                "risk_analysis.csv",
                "text/csv"
            )

        # ABA 4: RASTREAMENTO DE CONTATOS
        with tab4:
            st.markdown("### 🕵️ Contact Tracing Log")
            
            if model.infection_log:
                df_log = pd.DataFrame(model.infection_log)
                
                df_display = df_log.rename(columns={
                    "time_h": "Hora da Infecção", 
                    "victim_id": "ID do Agente", 
                    "dose_at_infection": "Dose Crítica (q)", 
                    "location": "Local (x, y)"
                })
                
                df_display["Hora da Infecção"] = df_display["Hora da Infecção"].round(3)
                df_display["Dose Crítica (q)"] = df_display["Dose Crítica (q)"].round(2)
                
                st.dataframe(df_display, use_container_width=True, hide_index=True)
                
                st.markdown("#### 📊 Estatísticas de Transmissão")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total de Transmissões", len(df_log))
                col2.metric("Dose Média na Infecção", f"{df_log['dose_at_infection'].mean():.2f} q")
                col3.metric("Tempo Médio até Infecção", f"{df_log['time_h'].mean():.2f}h")
                
                csv_log = df_display.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Baixar Log de Rastreamento (CSV)",
                    csv_log,
                    "contact_tracing_log.csv",
                    "text/csv"
                )
            else:
                st.success("✅ **Cenário Seguro:** Nenhuma transmissão secundária foi registrada!")

        # Botão de Reset
        st.divider()
        col_reset, col_download = st.columns([1, 3])
        
        with col_reset:
            if st.button("🔄 Nova Simulação", type="primary", use_container_width=True):
                st.session_state.simulation_done = False
                st.session_state.model = None
                st.session_state.spatial_history = []
                st.rerun()
        
        with col_download:
            csv_full = pd.DataFrame(model.metrics_history).to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Baixar Série Temporal Completa (CSV)",
                csv_full,
                "simulation_timeseries.csv",
                "text/csv",
                use_container_width=True
            )

    # ========================================================================
    # TELA INICIAL (BEM-VINDA)
    # ========================================================================
    else:
        st.info("👈 **Configure os parâmetros na barra lateral e clique em 'INICIAR SIMULAÇÃO'.**")
        
        st.markdown("""
        ### 🧪 Sobre o Simulador
        
        Ferramenta para análise de risco de transmissão aérea de patógenos (ex: SARS-CoV-2) em ambientes internos, combinando **Dinâmica de Fluidos (CFD)** e **Modelagem Baseada em Agentes (ABM)**.
        
        **Destaques:**
        * **Física:** Dispersão de aerossóis, ventilação e decaimento viral.
        * **Comportamento:** Agentes autônomos com perfis de movimentação e respiração.
        * **Epidemiologia:** Cálculo probabilístico de infecção (Wells-Riley) em tempo real.
        
        ---
        **Autor:** Leon Lourenço da Silva Santos | **Disciplina:** Epidemiologia Computacional (UFRPE)
        """)

if __name__ == "__main__":
    main()