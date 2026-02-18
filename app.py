"""
IAQ Simulator - Interface Principal (Streamlit)
---------------------------------------------------
Este arquivo é o ponto de entrada da aplicação web. Ele gerencia:
1. A configuração dos parâmetros da simulação (Barra Lateral).
2. A execução do motor de simulação (IAQSimulator).
3. A visualização interativa dos dados (Plotly).
4. A exportação de vídeos e relatórios (Matplotlib/Pandas).

Autor: Leon Lourenço (UFRPE)
Licença: MIT
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import imageio.v2 as imageio
import pandas as pd
import io
import os
import json
import time

# Importa o motor de simulação e constantes
from src.simulation_engine import IAQSimulator, OBJ_WALL, OBJ_DESK, OBJ_AC, OBJ_WINDOW, OBJ_DOOR

# CONSTANTE: Caminho do arquivo de resultados
RESULT_FILE = os.path.join("results", "simulation_result.npz")

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="IAQ Simulator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏥 IAQ Simulator: Análise Epidemiológica e Ambiental")
st.markdown("""
**Simulação Híbrida LBM-ABM (Lattice Boltzmann + Agent-Based Model)**

Este sistema simula a dinâmica de transmissão aérea seguindo modelos compartimentais estocásticos em grade (Autômatos Celulares),
atendendo aos requisitos de **Epidemiologia Computacional** e **Física de Fluidos**.
""")

# ==============================================================================
# FUNÇÃO DE GERAÇÃO DE VÍDEO
# ==============================================================================
def generate_video_optimized(hist_data, mode="Risco Viral"):
    """
    Gera um vídeo MP4 da simulação usando Matplotlib e ImageIO.
    """
    output_filename = "simulation_video.mp4"
    fps = 20
    max_duration_sec = 90
    
    # Extração de dados (já garantimos que hist_data são arrays numpy puros)
    virus_data = hist_data['virus']
    co2_data = hist_data['co2']
    layout = hist_data['layout']
    pos_data = hist_data['pos']
    meta = hist_data['meta']
    ux_data = hist_data['ux']
    uy_data = hist_data['uy']
    agent_stats = hist_data['agent_stats']
    
    total_frames = len(virus_data)
    target_frames = fps * max_duration_sec
    skip = max(1, int(total_frames / target_frames))
    
    cmap_wall = ListedColormap(['black'])
    cmap_desk = ListedColormap(['#8B4513']) 
    cmap_ac = ListedColormap(['#0000CD'])   
    cmap_window = ListedColormap(['#00FFFF']) 
    cmap_door = ListedColormap(['#FF8C00']) 
    
    if mode == "Risco Viral":
        g_max = np.max(virus_data) * 0.8 + 0.001
        g_min = 0
        norm = Normalize(vmin=g_min, vmax=g_max)
        cmap_fluid = 'inferno'
    else:
        vals = co2_data + 400.0
        g_max = np.max(vals)
        g_min = 400.0
        norm = Normalize(vmin=g_min, vmax=g_max)
        cmap_fluid = 'jet'

    writer = imageio.get_writer(output_filename, fps=fps, codec='libx264', format='FFMPEG')
    
    step_arrow = 5
    ny, nx = ux_data[0].shape
    step_slice = int(step_arrow)
    X, Y = np.meshgrid(np.arange(0, nx, step_slice), np.arange(0, ny, step_slice))
    
    prog_bar = st.progress(0, text=f"Renderizando vídeo ({mode})...")
    frames_to_render = range(0, total_frames, skip)
    total_render = len(frames_to_render)
    
    for count, i in enumerate(frames_to_render):
        # FIX: Dimensões fixas 800x480 (Divisível por 16)
        fig = plt.figure(figsize=(8, 4.8), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        
        if mode == "Risco Viral":
            frame_data = virus_data[i]
        else:
            frame_data = co2_data[i] + 400.0
            
        ax.imshow(frame_data.astype(np.float32), cmap=cmap_fluid, norm=norm, origin='lower', interpolation='bilinear')
        
        ax.imshow(np.ma.masked_where(layout != OBJ_WALL, layout), cmap=cmap_wall, origin='lower', alpha=1.0)
        ax.imshow(np.ma.masked_where(layout != OBJ_DESK, layout), cmap=cmap_desk, origin='lower', alpha=0.8)
        ax.imshow(np.ma.masked_where(layout != OBJ_AC, layout), cmap=cmap_ac, origin='lower', alpha=1.0)
        ax.imshow(np.ma.masked_where(layout != OBJ_WINDOW, layout), cmap=cmap_window, origin='lower', alpha=0.5)
        ax.imshow(np.ma.masked_where(layout != OBJ_DOOR, layout), cmap=cmap_door, origin='lower', alpha=0.8)
        
        u_sub = ux_data[i][::step_slice, ::step_slice].astype(np.float32)
        v_sub = uy_data[i][::step_slice, ::step_slice].astype(np.float32)
        ax.quiver(X, Y, u_sub, v_sub, color='cyan', alpha=0.5, scale=5, width=0.005)
        
        agents_pos = pos_data[i]
        stats = agent_stats[i]
        colors = ['red' if s[0] == 1 else ('orange' if s[0]==2 else 'lime') for s in stats]
        
        if len(agents_pos) > 0:
            ax.scatter(agents_pos[:, 0], agents_pos[:, 1], c=colors, s=80, edgecolors='black', zorder=10)
        
        min_per_frame = meta['save_interval'] / meta['steps_per_min']
        current_time = i * min_per_frame
        
        info_text = f"Tempo: {current_time:.1f} min\nModo: {mode}\nMax: {np.max(frame_data):.0f}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, color='white', 
                verticalalignment='top', fontsize=10, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
        
        ax.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='black')
        plt.close(fig)
        
        buf.seek(0)
        writer.append_data(imageio.imread(buf))
        buf.close()
        
        prog_bar.progress((count + 1) / total_render)
        
    writer.close()
    prog_bar.empty()
    return output_filename

# ==============================================================================
# CARREGAMENTO DO CENÁRIO PADRÃO (JSON)
# ==============================================================================
try:
    with open("scenarios/school.json", "r") as f:
        default_cfg = json.load(f)
except Exception:
    default_cfg = {
        "physics": {"width_m": 10.0, "height_m": 8.0},
        "agents": {"total": 25, "infected": 1, "rows": 4},
        "ventilation": {"ach_default": 4.5, "window_open": True}
    }

# ==============================================================================
# BARRA LATERAL (CONTROLES)
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Configuração do Cenário")
    
    # 1. Dimensões
    st.subheader("Arquitetura")
    c1, c2 = st.columns(2)
    with c1: 
        room_w = st.number_input("Largura (m)", 5.0, 20.0, 
                                 float(default_cfg['physics']['width_m']), format="%.1f")
    with c2: 
        room_h = st.number_input("Comprimento (m)", 5.0, 15.0, 
                                 float(default_cfg['physics']['height_m']), format="%.1f")
    
    layout_rows = st.slider("Fileiras de Carteiras", 2, 6, 
                            int(default_cfg['agents']['rows']))
    window_open = st.checkbox("Janelas Abertas", 
                              value=bool(default_cfg['ventilation']['window_open']), 
                              help="Habilita troca de ar com o exterior.")
    
    st.divider()
    
    # 2. Ventilação e Física
    st.subheader("💨 Ventilação (IAQ)")
    ach = st.slider("ACH (Trocas de Ar/Hora)", 0.0, 15.0, 
                    float(default_cfg['ventilation']['ach_default']), 
                    help="0.0 = Sala Hermética.\n4.5 = Escola Padrão.\n10+ = Hospitalar.")
    ac_power = st.slider("Potência do AC (Vento)", 5.0, 30.0, 15.0)
    
    st.divider()
    
    # 3. Tempo e Ocupantes
    st.subheader("⏱️ Simulação")
    hours = st.slider("Duração Real (Horas)", 1, 6, 4)
    
    precision = st.selectbox("Qualidade", 
                             ["Rápida (10 steps/min)", "Normal (30 steps/min)", "Alta (60 steps/min)"],
                             index=0)
    
    if "Rápida" in precision: spm = 10
    elif "Normal" in precision: spm = 30
    else: spm = 60
    
    st.caption(f"Processamento: **{int(hours * 60 * spm)} passos** físicos.")
    
    st.subheader("👥 Ocupantes")
    col_p1, col_p2 = st.columns(2)
    with col_p1: 
        n_agents = st.number_input("Total", 5, 50, int(default_cfg['agents']['total']))
    with col_p2: 
        n_infected = st.number_input("Infectados (I0)", 1, 10, int(default_cfg['agents']['infected']))
    
    col_b1, col_b2 = st.columns([2, 1])
    with col_b1:
        btn_run = st.button("🔴 RODAR SIMULAÇÃO", type="primary", width='stretch')
    with col_b2:
        # BOTÃO LIMPAR SEGURO
        if st.button("🗑️ Limpar", help="Apaga resultados anteriores", width='stretch'):
            try:
                # Tenta remover o arquivo. Se falhar, captura o erro sem crashar.
                if os.path.exists(RESULT_FILE):
                    os.remove(RESULT_FILE)
                    # Limpa o cache do Streamlit para forçar recarregamento
                    st.cache_data.clear()
                    st.rerun()
            except PermissionError:
                st.error("⚠️ O arquivo está em uso. Tente novamente em alguns segundos.")
            except Exception as e:
                st.error(f"Erro ao limpar: {e}")

# ==============================================================================
# LÓGICA DE EXECUÇÃO
# ==============================================================================
if btn_run:
    # 1. Cria dicionário com as configurações da UI (Overrides)
    ui_overrides = {
        "physics": {
            "width_m": room_w, 
            "height_m": room_h
        },
        "agents": {
            "total": n_agents, 
            "infected": n_infected, 
            "rows": layout_rows
        },
        "ventilation": {
            "window_open": window_open
        }
    }
    
    # 2. Instancia o simulador injetando os overrides
    sim = IAQSimulator("scenarios/school.json", config_overrides=ui_overrides) 
    
    with st.spinner(f"Processando {hours} horas de dinâmica de fluidos e agentes..."):
        try:
            # Garante que o arquivo antigo seja fechado/removido antes de salvar o novo
            if os.path.exists(RESULT_FILE):
                try:
                    os.remove(RESULT_FILE)
                except:
                    pass # Se não der pra remover, o savez_compressed tenta sobrescrever
            
            hist_result = sim.run_simulation(
                total_hours=hours,
                ach_target=ach,
                ac_power=ac_power,
                window_open=window_open,
                steps_per_min=spm
            )
            # Garante que o diretório existe antes de salvar
            os.makedirs("results", exist_ok=True)
            np.savez_compressed(RESULT_FILE, **hist_result)
            st.success("✅ Simulação concluída!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Erro durante a simulação: {e}")

# ==============================================================================
# VISUALIZAÇÃO DOS RESULTADOS
# ==============================================================================
if os.path.exists(RESULT_FILE):
    try:
        # --- CARREGAMENTO SEGURO (CONTEXT MANAGER) ---
        # Isso garante que o arquivo seja fechado IMEDIATAMENTE após a leitura
        with np.load(RESULT_FILE, allow_pickle=True) as data_file:
            # Carrega tudo para a memória RAM e desconecta do arquivo
            virus_hist = np.array(data_file['virus'])
            co2_hist = np.array(data_file['co2'])
            ux_hist = np.array(data_file['ux'])
            uy_hist = np.array(data_file['uy'])
            pos_hist = np.array(data_file['pos'])
            layout = np.array(data_file['layout'])
            agent_stats = np.array(data_file['agent_stats'])
            infected_total = np.array(data_file['infected_total'])
            
            # Trata metadados (escalar ou array 0-d)
            raw_meta = data_file['meta']
            meta = raw_meta.item() if raw_meta.shape == () else raw_meta
        
        # --- PROCESSAMENTO ---
        # Agora estamos trabalhando apenas com a RAM, o arquivo 'simulation_result.npz' está livre.
        spm = float(meta['steps_per_min'])
        save_int = float(meta['save_interval'])
        min_per_frame = save_int / spm
        total_frames = len(virus_hist)
        
        # Agrupa dados em um dict para passar para a função de vídeo
        data_dict = {
            'virus': virus_hist, 'co2': co2_hist, 'ux': ux_hist, 'uy': uy_hist,
            'pos': pos_hist, 'layout': layout, 'agent_stats': agent_stats, 'meta': meta
        }
        
        metrics = []
        for t in range(total_frames):
            infected_count = infected_total[t]
            real_total_agents = len(pos_hist[t])
            susceptible_count = real_total_agents - infected_count
            co2_mean = np.mean(co2_hist[t]) + 400.0
            metrics.append({
                "Tempo (min)": t * min_per_frame,
                "Suscetíveis (S)": susceptible_count,
                "Infectados (I)": infected_count,
                "CO2 Médio (ppm)": co2_mean,
                "Carga Viral Máx": np.max(virus_hist[t])
            })
        df_metrics = pd.DataFrame(metrics)
        
        # --- TABs ---
        tab_viz, tab_graphs, tab_export = st.tabs(["👁️ Visualização Espacial", "📈 Curvas Epidemiológicas", "📥 Exportação"])
        
        # --- TAB 1: MAPA DE CALOR E VETORES ---
        with tab_viz:
            c_ctrl, c_view = st.columns([1, 3])
            
            with c_ctrl:
                st.markdown("### Controles")
                view_mode = st.radio("Camada:", ["Risco Viral", "CO2 (ppm)"])
                show_wind = st.checkbox("Mostrar Vento (Setas)", value=True)
                frame_idx = st.slider("Tempo (Frames)", 0, total_frames-1, 0)
                st.metric("Tempo Simulado", f"{frame_idx * min_per_frame:.1f} min")
                
                if "CO2" in view_mode:
                    curr_data = co2_hist[frame_idx] + 400.0
                    st.metric("CO2 Sala", f"{np.mean(curr_data):.0f} ppm")
                else:
                    st.metric("Carga Viral Máx", f"{np.max(virus_hist[frame_idx]):.2f} q")

            with c_view:
                if "CO2" in view_mode:
                    z_data = co2_hist[frame_idx] + 400.0
                    colors = 'Jet'
                    z_min, z_max = 400, np.max(co2_hist) + 400
                    hover_fluid = "<b>Ar (CO2)</b><br>Conc: %{z:.0f} ppm<extra></extra>"
                else:
                    z_data = virus_hist[frame_idx]
                    colors = 'Inferno'
                    z_min, z_max = 0, np.max(virus_hist)
                    hover_fluid = "<b>Ar (Vírus)</b><br>Conc: %{z:.2f} q/m³<extra></extra>"

                fig = go.Figure()
                
                fig.add_trace(go.Heatmap(z=z_data, colorscale=colors, zmin=z_min, zmax=z_max, zsmooth='best', hovertemplate=hover_fluid))
                
                if show_wind:
                    step_q = 6
                    ny, nx = z_data.shape
                    y_grid, x_grid = np.mgrid[0:ny:step_q, 0:nx:step_q]
                    u = ux_hist[frame_idx][::step_q, ::step_q]
                    v = uy_hist[frame_idx][::step_q, ::step_q]
                    quiver = ff.create_quiver(x_grid.flatten(), y_grid.flatten(), u.flatten(), v.flatten(),
                                              scale=35, arrow_scale=0.4, line=dict(color='cyan', width=1), opacity=0.5, hoverinfo='skip')
                    fig.add_traces(quiver.data)
                
                objects_map = {
                    OBJ_WALL: {"name": "Parede", "color": "black", "symbol": "square"},
                    OBJ_DESK: {"name": "Mesa", "color": "brown", "symbol": "square"},
                    OBJ_AC: {"name": "AC (Inlet)", "color": "blue", "symbol": "square"},
                    OBJ_WINDOW: {"name": "Janela", "color": "cyan", "symbol": "line-ns"},
                    OBJ_DOOR: {"name": "Porta", "color": "orange", "symbol": "square"}
                }
                
                for obj_id, props in objects_map.items():
                    dy, dx = np.where(layout == obj_id)
                    if len(dx) > 0:
                        fig.add_trace(go.Scatter(x=dx, y=dy, mode='markers',
                                                 marker=dict(symbol=props["symbol"], color=props["color"], size=8),
                                                 name=props["name"], hovertemplate=f"<b>{props['name']}</b><br>Obstáculo<extra></extra>"))

                ag = pos_hist[frame_idx]
                stats = agent_stats[frame_idx]
                c_ag = ['lime' if s[0]==0 else ('red' if s[0]==1 else 'orange') for s in stats]
                hover_ag = []
                status_map_inv = {0: "Suscetível", 1: "INFECTADO", 2: "Assintomático"}
                for i, s in enumerate(stats):
                    st_str = status_map_inv.get(int(s[0]), "Desconhecido")
                    hover_ag.append(f"<b>Aluno #{i}</b><br>Status: {st_str}<br>Emissão CO2: {s[1]:.1f}<br>Emissão Vírus: {s[2]:.2f}")

                fig.add_trace(go.Scatter(x=ag[:,0], y=ag[:,1], mode='markers', 
                                        marker=dict(color=c_ag, size=12, line=dict(width=2, color='black')),
                                        text=hover_ag, hoverinfo='text', name="Ocupantes"))
                
                fig.update_layout(height=500, margin=dict(l=0,r=0,t=0,b=0), xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, width='stretch')

        # --- TAB 2: GRÁFICOS SIR ---
        with tab_graphs:
            st.subheader("Análise de Dinâmica Epidemiológica")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Curva S-I (Suscetíveis vs Infectados)")
                fig_sir = px.line(df_metrics, x="Tempo (min)", y=["Suscetíveis (S)", "Infectados (I)"],
                                  color_discrete_map={"Suscetíveis (S)": "green", "Infectados (I)": "red"})
                fig_sir.update_layout(height=350)
                st.plotly_chart(fig_sir, width='stretch')
            with c2:
                st.markdown("#### Qualidade do Ar (CO2)")
                fig_co2 = px.line(df_metrics, x="Tempo (min)", y="CO2 Médio (ppm)")
                fig_co2.add_hline(y=1000, line_dash="dash", line_color="red", annotation_text="Limite Crítico")
                fig_co2.update_layout(height=350)
                st.plotly_chart(fig_co2, width='stretch')

        # --- TAB 3: EXPORTAÇÃO ---
        with tab_export:
            c_exp1, c_exp2 = st.columns(2)
            with c_exp1:
                st.markdown("#### 🎥 Vídeo MP4")
                
                video_mode = st.selectbox("Selecione o Conteúdo do Vídeo:", 
                                         ["Risco Viral", "CO2"], 
                                         index=0)
                
                if st.button(f"🎬 Renderizar Vídeo ({video_mode})"):
                    with st.spinner("Gerando quadros do vídeo..."):
                        # Passa o dicionário 'data_dict' que já está na memória
                        v_path = generate_video_optimized(data_dict, mode=video_mode)
                    
                    with open(v_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Baixar Vídeo MP4",
                            data=f,
                            file_name=f"simulacao_{video_mode.lower().replace(' ', '_')}.mp4",
                            mime="video/mp4"
                        )
            with c_exp2:
                st.markdown("#### 📊 Relatório CSV")
                csv = df_metrics.to_csv(index=False).encode('utf-8')
                st.download_button("Baixar Dados Completos", csv, "relatorio_epidemiologico.csv", "text/csv")

    except Exception as e:
        st.error(f"Erro ao carregar visualização: {e}")
        st.code(f"Detalhes: {str(e)}")

else:
    st.info("👈 Configure os parâmetros na barra lateral e clique em **RODAR SIMULAÇÃO** para começar.")
    st.markdown("""
    ### Como usar:
    1. Ajuste as **Dimensões da Sala** e **Ocupação** na barra lateral.
    2. Defina o nível de ventilação (**ACH**) - *Dica: Use 0.0 para testar cenários críticos.*
    3. Escolha a **Qualidade** (Steps/min) para balancear precisão e velocidade.
    4. Analise os resultados nos gráficos interativos ou exporte o vídeo.
    """)