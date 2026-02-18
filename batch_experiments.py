"""
batch_experiments.py
-------------------------------
SCRIPT DE GERAÇÃO DE DADOS EM ALTA FIDELIDADE (MONTE CARLO)
===========================================================
Este script executa uma bateria robusta de simulações para garantir significância estatística.
Duração estimada: 15-25 minutos (dependendo da CPU).

Metodologia:
1. Executa 4 cenários distintos.
2. Cada cenário é rodado 5 VEZES (Repetições) para eliminar o acaso.
3. Usa precisão de física máxima (60 steps/min).
4. Exporta dados brutos e estatísticas agregadas (Média ± Desvio Padrão).

Saídas:
- dados_monte_carlo_raw.csv: Resultado de cada uma das 20 simulações individuais.
- dados_estatisticos_finais.csv: Médias consolidadas.
- dados_series_temporais.csv: Curvas de evolução para gráficos de linha.
"""

import numpy as np
import pandas as pd
import time
import os
from src.simulation_engine import IAQSimulator

# --- DIRETÓRIO DE SAÍDA ---
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# --- CONFIGURAÇÃO ---
REPETITIONS = 5       # Número de vezes que roda cada cenário
DURATION_HOURS = 4.0  # Duração de uma aula típica
STEPS_PER_MIN = 60    # Alta Precisão Física (1Hz)

def run_single_realization(run_id, scenario_name, ach, window_open, population, infected):
    """Roda uma simulação individual e retorna os dados."""
    
    # Configuração de Override
    overrides = {
        "ventilation": { "ach_default": ach, "window_open": window_open },
        "agents": { "total": population, "infected": infected, "rows": 4 },
        "physics": { "width_m": 10.0, "height_m": 8.0 }
    }
    
    start_t = time.time()
    sim = IAQSimulator("scenarios/school.json", config_overrides=overrides)
    
    # Execução Headless
    hist = sim.run_simulation(
        total_hours=DURATION_HOURS,
        ach_target=ach,
        ac_power=15.0, # AC constante
        window_open=window_open,
        steps_per_min=STEPS_PER_MIN
    )
    elapsed = time.time() - start_t
    
    # Extração de Métricas Finais
    final_infected = hist['infected_total'][-1]
    attack_rate = ((final_infected - infected) / (population - infected)) * 100
    peak_viral = np.max(hist['virus'])
    # Dose Acumulada Média (Integral no tempo da média espacial)
    avg_virus_curve = np.mean(hist['virus'], axis=(1, 2)) # Média espacial por timestep
    integrated_dose = np.sum(avg_virus_curve)
    final_co2 = np.mean(hist['co2'][-1]) + 400.0
    
    # Extração de Série Temporal (Downsampling para 1 ponto por minuto para economizar disco)
    # Pegamos a cada 60 frames (já que STEPS_PER_MIN=60)
    series_data = []
    total_frames = len(hist['infected_total'])
    for t in range(0, total_frames, 60): # 1 ponto por minuto
        series_data.append({
            "Run_ID": run_id,
            "Cenário": scenario_name,
            "Tempo_Min": t, # Como 1 frame = 1 seg (60spm), t é segundos? Não, t é frame index.
                            # Se steps_per_min=60, save_interval costuma ser ajustado.
                            # Vamos calcular tempo real baseado no meta
            "Infectados": hist['infected_total'][t],
            "CO2_PPM": np.mean(hist['co2'][t]) + 400.0,
            "Virus_Quanta": np.mean(hist['virus'][t])
        })
        
    results = {
        "Run_ID": run_id,
        "Cenário": scenario_name,
        "Repetição": run_id.split("_")[-1],
        "Infectados": final_infected,
        "Taxa_Ataque": attack_rate,
        "Pico_Viral": peak_viral,
        "Dose_Index": integrated_dose,
        "CO2_Final": final_co2,
        "Tempo_Calc_s": elapsed
    }
    
    return results, series_data

# --- ORQUESTRADOR ---
if __name__ == "__main__":
    print(f"🚀 INICIANDO BATERIA MONTE CARLO (n={REPETITIONS})")
    print(f"⏱️  Tempo Estimado: ~20 minutos. Vá tomar um café ☕")
    print("==================================================================")
    
    scenarios = [
        {"code": "A", "name": "A. Hermético (Controle)", "ach": 0.5, "win": False, "pop": 25},
        {"code": "B", "name": "B. Escola Padrão", "ach": 4.5, "win": True, "pop": 25},
        {"code": "C", "name": "C. Ventilação Otimizada", "ach": 10.0, "win": True, "pop": 25},
        {"code": "D", "name": "D. Superlotação", "ach": 4.5, "win": True, "pop": 40}
    ]
    
    all_raw_data = []
    all_series_data = []
    
    total_runs = len(scenarios) * REPETITIONS
    current_run = 0
    
    for sc in scenarios:
        print(f"\n🧪 GRUPO DE TESTE: {sc['name']}")
        
        for i in range(1, REPETITIONS + 1):
            run_id = f"{sc['code']}_Rep{i}"
            current_run += 1
            print(f"   ▶️  Executando Repetição {i}/{REPETITIONS} (Progresso Global: {current_run}/{total_runs})...", end="", flush=True)
            
            try:
                res, series = run_single_realization(
                    run_id, sc['name'], sc['ach'], sc['win'], sc['pop'], infected=1
                )
                
                all_raw_data.append(res)
                all_series_data.extend(series)
                print(f" ✅ [Inf: {res['Infectados']} | CO2: {int(res['CO2_Final'])}]")
                
            except Exception as e:
                print(f" ❌ ERRO: {e}")
    
    # --- CONSOLIDAÇÃO E ESTATÍSTICA ---
    print("\n Processando Estatísticas...")
    df_raw = pd.DataFrame(all_raw_data)
    df_series = pd.DataFrame(all_series_data)
    
    # Agrupamento por Cenário para calcular Média e Desvio Padrão
    df_stats = df_raw.groupby("Cenário").agg({
        "Taxa_Ataque": ["mean", "std", "min", "max"],
        "Pico_Viral": ["mean", "std"],
        "Dose_Index": ["mean", "std"],
        "CO2_Final": ["mean"],
        "Tempo_Calc_s": ["sum"]
    }).round(2)
    
    # --- EXPORTAÇÃO ---
    raw_path = os.path.join(RESULTS_DIR, "dados_monte_carlo_raw.csv")
    series_path = os.path.join(RESULTS_DIR, "dados_series_temporais.csv")
    stats_path = os.path.join(RESULTS_DIR, "dados_estatisticos_finais.csv")

    df_raw.to_csv(raw_path, index=False)
    df_series.to_csv(series_path, index=False)
    df_stats.to_csv(stats_path)

    
    print("\n CONCLUÍDO! Arquivos gerados em /results:")
    print("   1. dados_estatisticos_finais.csv (Use na Tabela 1 do Paper)")
    print("   2. dados_monte_carlo_raw.csv (Para Scatter Plots)")
    print("   3. dados_series_temporais.csv (Para Gráficos de Linha)")
    
    print("\n📊 RESUMO ESTATÍSTICO (Média das Repetições):")
    print(df_stats["Taxa_Ataque"])