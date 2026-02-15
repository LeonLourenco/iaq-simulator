"""
Script de Geração de Ativos para Artigo Científico (Nature/Science Format).

Autor: Leon Lourenço da Silva Santos
Disciplina: Epidemiologia Computacional - UFRPE
Objetivo: Gerar embasamento quantitativo, gráficos de alta resolução e estatísticas
          para o artigo final do Simulador IAQ.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from scipy.ndimage import gaussian_filter

# Importações do Simulador
from config import create_school_scenario, create_office_scenario, ScenarioConfig
from model import IAQModel

# ============================================================================
# CONFIGURAÇÃO GERAL
# ============================================================================
OUTPUT_DIR = Path("results/paper_assets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Estilo dos gráficos (Padrão Acadêmico)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif' # Estilo LaTeX/Scientific

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("PaperGen")

class VirtualLaboratory:
    def __init__(self):
        self.stats_buffer = []

    def log_stat(self, experiment, key, value):
        """Registra uma estatística para o relatório de texto."""
        self.stats_buffer.append(f"[{experiment}] {key}: {value}")

    def save_stats(self):
        """Salva o resumo estatístico em texto."""
        with open(OUTPUT_DIR / "resumo_estatistico.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(self.stats_buffer))
        logger.info(f"Relatório estatístico salvo em {OUTPUT_DIR}")

    # ========================================================================
    # EXPERIMENTO 1: VALIDAÇÃO DA CURVA EPIDÊMICA (SIR)
    # ========================================================================
    def run_validation_experiment(self):
        logger.info(">>> Iniciando Exp 1: Validação Dinâmica SIR (Boarding School Style)...")
        
        # Cenário de alta transmissibilidade para gerar curva clara
        scenario = create_school_scenario(occupants=50, infected=1, ach=1.0)
        scenario.duration_hours = 12.0 # Estendido para ver evolução
        scenario.physics.room_width_m = 8.0 # Sala menor para garantir contágio
        scenario.physics.room_height_m = 8.0
        
        model = IAQModel(scenario)
        
        # Execução
        sim_data = []
        while model.running:
            model.step()
            counts = model.get_state_counts()
            sim_data.append({
                "Tempo (h)": model.time / 3600.0,
                "Suscetíveis": counts["SUSCEPTIBLE"],
                "Infectados": counts["INFECTED"],
                "Recuperados": counts["RECOVERED"]
            })
            if counts["INFECTED"] == 0 and model.time > 3600: break # Encerra se acabar o surto

        df = pd.DataFrame(sim_data)
        
        # Plotagem
        plt.figure(figsize=(8, 5))
        plt.plot(df["Tempo (h)"], df["Suscetíveis"], '--', label="Suscetíveis (S)", color="#2ca02c", linewidth=2)
        plt.plot(df["Tempo (h)"], df["Infectados"], '-', label="Infectados (I)", color="#d62728", linewidth=2.5)
        plt.plot(df["Tempo (h)"], df["Recuperados"], '-.', label="Recuperados (R)", color="#1f77b4", linewidth=2)
        
        plt.title("Dinâmica Epidemiológica em Ambiente Escolar (Simulado)")
        plt.xlabel("Tempo de Exposição (Horas)")
        plt.ylabel("Número de Indivíduos")
        plt.legend(frameon=True)
        plt.tight_layout()
        
        filename = OUTPUT_DIR / "fig1_curva_sir_validacao.png"
        plt.savefig(filename)
        logger.info(f"Figura 1 gerada: {filename}")

        # Estatísticas
        peak_infected = df["Infectados"].max()
        peak_time = df.loc[df["Infectados"].idxmax(), "Tempo (h)"]
        attack_rate = (50 - df.iloc[-1]["Suscetíveis"]) / 50.0
        
        self.log_stat("Exp 1 (Validação)", "Pico de Infectados", f"{peak_infected} alunos")
        self.log_stat("Exp 1 (Validação)", "Tempo do Pico", f"{peak_time:.1f} horas")
        self.log_stat("Exp 1 (Validação)", "Taxa de Ataque Final", f"{attack_rate*100:.1f}%")

    # ========================================================================
    # EXPERIMENTO 2: SENSIBILIDADE À VENTILAÇÃO (ACH)
    # ========================================================================
    def run_ventilation_sensitivity(self):
        logger.info(">>> Iniciando Exp 2: Impacto da Ventilação (ACH)...")
        
        ach_levels = [0.5, 2.0, 4.0, 6.0, 10.0]
        results = []
        
        # Roda 3 simulações por nível para robustez (média estocástica)
        n_trials = 3 
        
        for ach in ach_levels:
            for i in range(n_trials):
                # Setup consistente
                np.random.seed(42 + i) # Seeds diferentes para variabilidade controlada
                scenario = create_school_scenario(occupants=30, infected=1, ach=ach)
                scenario.duration_hours = 6.0
                
                model = IAQModel(scenario)
                while model.running:
                    model.step()
                
                final_counts = model.get_state_counts()
                attack_rate = (30 - final_counts["SUSCEPTIBLE"]) / 30.0
                
                # Coleta Dose Média dos Suscetíveis (Métrica mais sensível que binária I/S)
                doses = [a.accumulated_dose for a in model.schedule.agents if a.unique_id != 0]
                avg_dose = np.mean(doses) if doses else 0
                
                results.append({
                    "ACH": ach,
                    "Taxa de Ataque": attack_rate,
                    "Dose Média (quanta)": avg_dose,
                    "Trial": i
                })
        
        df = pd.DataFrame(results)
        
        # Gráfico de Linha com Banda de Confiança
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=df, x="ACH", y="Dose Média (quanta)", marker="o", color="navy", linewidth=2.5)
        plt.title("Redução da Dose Viral Inalada por Ventilação Mecânica")
        plt.xlabel("Trocas de Ar por Hora (ACH)")
        plt.ylabel("Dose Média Acumulada (quanta)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = OUTPUT_DIR / "fig2_sensibilidade_ach.png"
        plt.savefig(filename)
        
        # Estatísticas Comparativas
        dose_low = df[df["ACH"]==0.5]["Dose Média (quanta)"].mean()
        dose_high = df[df["ACH"]==10.0]["Dose Média (quanta)"].mean()
        reduction = (1 - dose_high/dose_low) * 100
        
        self.log_stat("Exp 2 (Ventilação)", "Dose Média (ACH 0.5)", f"{dose_low:.4f} q")
        self.log_stat("Exp 2 (Ventilação)", "Dose Média (ACH 10.0)", f"{dose_high:.4f} q")
        self.log_stat("Exp 2 (Ventilação)", "Eficácia da Redução", f"{reduction:.1f}%")

    # ========================================================================
    # EXPERIMENTO 3: ANÁLISE ESPACIAL (HEATMAP CFD)
    # ========================================================================
    def run_spatial_analysis(self):
        logger.info(">>> Iniciando Exp 3: Mapeamento Espacial de Risco...")
        
        # Cenário de Escritório com obstáculos
        scenario = create_office_scenario(occupants=20, infected=1, ach=2.0)
        scenario.duration_hours = 2.0 # Curto, apenas para gerar a mancha
        
        model = IAQModel(scenario)
        
        # Avança até metade da simulação
        target_steps = int(scenario.duration_hours * 60 * 0.8) # 80% do tempo
        for _ in range(target_steps):
            model.step()
            
        # Extrai grid viral
        virus_grid = model.physics.get_virus_snapshot()
        
        # Suavização para melhor visualização (como interpolação visual)
        # O dado bruto é pixelado (células), o filtro gaussiano imita difusão visual
        virus_smooth = gaussian_filter(virus_grid, sigma=1.0)
        
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(virus_smooth.T, cmap="rocket_r", cbar_kws={'label': 'Concentração Viral (quanta/m³)'})
        
        # Overlay dos Agentes
        agents = model.schedule.agents
        x_s = [a.pos[0] + 0.5 for a in agents if a.state.name == "SUSCEPTIBLE"]
        y_s = [a.pos[1] + 0.5 for a in agents if a.state.name == "SUSCEPTIBLE"]
        x_i = [a.pos[0] + 0.5 for a in agents if a.state.name == "INFECTED"]
        y_i = [a.pos[1] + 0.5 for a in agents if a.state.name == "INFECTED"]
        
        ax.scatter(x_s, y_s, c='green', marker='o', s=100, edgecolors='white', label="Suscetível")
        ax.scatter(x_i, y_i, c='red', marker='X', s=150, edgecolors='white', label="Infectado")
        
        ax.invert_yaxis() # Plotly/Matplotlib axis match
        plt.title(f"Distribuição Espacial de Aerossóis (T={model.time/3600:.1f}h)")
        plt.xlabel("Dimensão X (Células)")
        plt.ylabel("Dimensão Y (Células)")
        plt.legend(loc="upper right")
        plt.tight_layout()
        
        filename = OUTPUT_DIR / "fig3_heatmap_espacial.png"
        plt.savefig(filename)
        
        max_conc = np.max(virus_grid)
        self.log_stat("Exp 3 (Espacial)", "Concentração Máxima Local", f"{max_conc:.2f} quanta/m³")

    # ========================================================================
    # EXPERIMENTO 4: EFICÁCIA DE MÁSCARAS (INTERVENÇÃO)
    # ========================================================================
    def run_mask_intervention(self):
        logger.info(">>> Iniciando Exp 4: Comparativo de Máscaras...")
        
        scenarios_config = [
            ("Sem Máscara", 0.0, 0.0),
            ("Pano (30% eff)", 1.0, 0.3),
            ("N95 (95% eff)", 1.0, 0.95)
        ]
        
        data = []
        
        for label, compliance, eff in scenarios_config:
            # Roda 5 vezes cada para média
            for i in range(5):
                scenario = create_school_scenario(occupants=40, infected=1, ach=3.0)
                scenario.duration_hours = 6.0
                scenario.agents.mask_compliance = compliance
                scenario.agents.mask_efficiency = eff
                
                model = IAQModel(scenario)
                while model.running:
                    model.step()
                
                # Métrica: R Efetivo (Novos casos / 1 Infectado inicial)
                final = model.get_state_counts()
                new_cases = final["INFECTED"] + final["RECOVERED"] - 1
                data.append({"Cenário": label, "Novos Casos": new_cases})
        
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(7, 6))
        sns.barplot(data=df, x="Cenário", y="Novos Casos", palette="viridis", errorbar="sd")
        plt.title("Impacto do Uso de Máscaras na Transmissão")
        plt.ylabel("Novos Casos (Média)")
        plt.tight_layout()
        
        filename = OUTPUT_DIR / "fig4_comparativo_mascaras.png"
        plt.savefig(filename)
        
        # Dados para texto
        avg_no_mask = df[df["Cenário"]=="Sem Máscara"]["Novos Casos"].mean()
        avg_n95 = df[df["Cenário"]=="N95 (95% eff)"]["Novos Casos"].mean()
        
        self.log_stat("Exp 4 (Máscaras)", "Casos (Sem Máscara)", f"{avg_no_mask:.1f}")
        self.log_stat("Exp 4 (Máscaras)", "Casos (N95)", f"{avg_n95:.1f}")

def main():
    lab = VirtualLaboratory()
    
    try:
        lab.run_validation_experiment()
        lab.run_ventilation_sensitivity()
        lab.run_spatial_analysis()
        lab.run_mask_intervention()
        
        lab.save_stats()
        print("\n" + "="*50)
        print(f"✅ GERAÇÃO DE DADOS CONCLUÍDA!")
        print(f"📂 Arquivos salvos em: {OUTPUT_DIR.absolute()}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Erro fatal na geração de dados: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()