#!/usr/bin/env python3
"""
CLI para Simulador Epidemiológico IAQ (Indoor Air Quality).

Ferramenta de linha de comando para executar simulações ABM+CFD,
gerar relatórios JSON e visualizar a dinâmica SIR.

Uso Exemplo:
    python run.py --scenario school --occupants 30 --ach 6.0 --plot
"""

import argparse
import json
import sys
import logging
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Importações do Projeto
try:
    from src.config import (
        create_school_scenario, 
        create_office_scenario, 
        create_gym_scenario,
        ScenarioConfig
    )
    from src.model import IAQModel
except ImportError as e:
    print(f"Erro crítico de importação: {e}")
    print("Certifique-se de estar na raiz do projeto 'iaq-simulator'.")
    sys.exit(1)

# Configuração de Logs simples para o terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("IAQ-CLI")

def parse_arguments():
    """Configura e processa os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Simulador de Transmissão Aérea (ABM + CFD Simplificado)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Argumentos Obrigatórios
    parser.add_argument(
        '--scenario', 
        type=str, 
        required=True, 
        choices=['school', 'office', 'gym'],
        help="Tipo de cenário para simular."
    )

    # Parâmetros de Simulação
    parser.add_argument('--occupants', type=int, default=50, help="Total de ocupantes.")
    parser.add_argument('--infected', type=int, default=1, help="Infectados iniciais.")
    parser.add_argument('--ach', type=float, default=4.0, help="Trocas de ar por hora (ACH).")
    parser.add_argument('--duration', type=float, default=8.0, help="Duração simulada em horas.")

    # Saídas
    parser.add_argument('--output', type=str, help="Caminho para salvar relatório JSON.")
    parser.add_argument('--plot', action='store_true', help="Gerar gráfico PNG da dinâmica SIR.")

    return parser.parse_args()

def get_scenario_config(args) -> ScenarioConfig:
    """Fábrica de cenários baseada nos argumentos."""
    if args.scenario == 'school':
        config = create_school_scenario(
            occupants=args.occupants, 
            infected=args.infected, 
            ach=args.ach
        )
    elif args.scenario == 'office':
        config = create_office_scenario(
            occupants=args.occupants, 
            infected=args.infected, 
            ach=args.ach
        )
    elif args.scenario == 'gym':
        config = create_gym_scenario(
            occupants=args.occupants, 
            infected=args.infected, 
            ach=args.ach
        )
    else:
        raise ValueError(f"Cenário desconhecido: {args.scenario}")
    
    # Sobrescreve a duração padrão do preset com a do argumento
    config.duration_hours = args.duration
    return config

def run_simulation_loop(model: IAQModel):
    """Executa o loop principal com feedback visual."""
    logger.info(f"Iniciando simulação: {model.config.name}")
    logger.info(f"Física: {model.physics.width_cells}x{model.physics.height_cells} grid, ACH={model.config.ventilation.ach}")
    
    total_steps = int(model.config.duration_hours * 60) # Aproximado (steps de 1 min)
    last_hour_reported = -1

    while model.running:
        model.step()
        
        # Feedback de progresso a cada hora simulada
        current_hour = int(model.time / 3600)
        if current_hour > last_hour_reported:
            stats = model.get_state_counts()
            logger.info(f"Hora {current_hour}: S={stats['SUSCEPTIBLE']} I={stats['INFECTED']} R={stats['RECOVERED']}")
            last_hour_reported = current_hour

    logger.info("Simulação concluída.")

def analyze_results(model: IAQModel):
    """Calcula métricas finais da simulação."""
    history = pd.DataFrame(model.metrics_history)
    
    initial_pop = model.config.agents.total_occupants
    initial_infected = model.config.agents.initial_infected
    
    final_stats = history.iloc[-1]
    final_s = final_stats['S']
    final_i = final_stats['I']
    final_r = final_stats['R']
    
    # 1. Taxa de Ataque: (Total - Final Suscetíveis) / Total (excluindo paciente zero?)
    # Definição padrão: Novos casos / População em risco.
    # Aqui usamos simplificado: % da população que pegou a doença
    total_infected_during_sim = (initial_pop - final_s)
    attack_rate = (total_infected_during_sim - initial_infected) / (initial_pop - initial_infected) if (initial_pop - initial_infected) > 0 else 0.0
    
    # 2. R efetivo (Proxy): Novos Casos / Casos Iniciais
    # Uma aproximação para este surto específico
    secondary_cases = total_infected_during_sim - initial_infected
    r_effective = secondary_cases / initial_infected if initial_infected > 0 else 0
    
    # 3. Pico de Infectados
    peak_infected = history['I'].max()

    return {
        "attack_rate": attack_rate,
        "r_effective": r_effective,
        "peak_infected": int(peak_infected),
        "final_susceptible": int(final_s),
        "final_infected": int(final_i),
        "final_recovered": int(final_r),
        "history_df": history
    }

def save_json_results(args, config, results, history_df):
    """Salva os resultados em formato JSON estruturado."""
    output_data = {
        "scenario": args.scenario,
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "occupants": config.agents.total_occupants,
            "initial_infected": config.agents.initial_infected,
            "ach": config.ventilation.ach,
            "duration_hours": config.duration_hours
        },
        "results": {k: v for k, v in results.items() if k != "history_df"},
        "time_series": history_df[['time_hours', 'S', 'I', 'R']].to_dict(orient='records')
    }

    # Se não foi passado --output, gera nome padrão se necessário, 
    # mas o requisito diz "Se --output: salvar".
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4)
            logger.info(f"Relatório salvo em: {args.output}")
        except IOError as e:
            logger.error(f"Erro ao salvar JSON: {e}")

def generate_plot(args, history_df):
    """Gera gráfico SIR usando Matplotlib."""
    if not args.plot:
        return

    plt.figure(figsize=(10, 6))
    
    plt.plot(history_df['time_hours'], history_df['S'], label='Suscetíveis', color='blue', linestyle='--')
    plt.plot(history_df['time_hours'], history_df['I'], label='Infectados', color='red', linewidth=2)
    plt.plot(history_df['time_hours'], history_df['R'], label='Recuperados', color='green')
    
    plt.title(f"Dinâmica SIR - Cenário: {args.scenario.upper()} (ACH={args.ach})")
    plt.xlabel("Tempo (Horas)")
    plt.ylabel("Número de Pessoas")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = f"sir_{args.scenario}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename)
    logger.info(f"Gráfico salvo em: {filename}")
    plt.close() # Libera memória

def main():
    try:
        # 1. Parse Argumentos
        args = parse_arguments()
        
        # 2. Configurar Cenário
        config = get_scenario_config(args)
        
        # 3. Instanciar Modelo
        model = IAQModel(config)
        
        # 4. Executar Simulação
        run_simulation_loop(model)
        
        # 5. Analisar Resultados
        results = analyze_results(model)
        
        # 6. Exibir no Terminal
        print("\n" + "="*40)
        print(f" RESULTADOS: {args.scenario.upper()}")
        print("="*40)
        print(f" Tempo Simulado : {config.duration_hours} horas")
        print(f" Taxa de Ataque : {results['attack_rate']*100:.2f}%")
        print(f" R Efetivo (Re) : {results['r_effective']:.2f}")
        print(f" Pico Infectados: {results['peak_infected']} pessoas")
        print("-" * 40)
        print(f" Final S/I/R    : {results['final_susceptible']} / {results['final_infected']} / {results['final_recovered']}")
        print("="*40 + "\n")
        
        # 7. Exportações
        if args.output:
            save_json_results(args, config, results, results['history_df'])
            
        if args.plot:
            generate_plot(args, results['history_df'])

    except KeyboardInterrupt:
        print("\nSimulação interrompida pelo usuário.")
        sys.exit(0)
    except Exception as e:
        logger.exception("Erro inesperado durante a execução.")
        sys.exit(1)

if __name__ == "__main__":
    main()