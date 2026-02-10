#!/usr/bin/env python3
"""
Exemplo de Análise em Lote
Executa múltiplas simulações com diferentes parâmetros
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config_final as cfg
from main_model import IAQSimulationModel


def run_batch_analysis():
    """Executa múltiplas simulações com diferentes configurações."""
    
    print("📋 Executando análise em lote do Simulador IAQ")
    print("=" * 70)
    
    # Configurações a testar
    scenarios_to_test = ['school', 'office', 'gym']
    occupancy_levels = [20, 50, 100]
    ach_levels = [2.0, 4.0, 8.0]
    
    results = []
    total = len(scenarios_to_test) * len(occupancy_levels) * len(ach_levels)
    current = 0
    
    print(f"Total de simulações planejadas: {total}")
    print("-" * 70)
    
    for scenario_name in scenarios_to_test:
        for occupants in occupancy_levels:
            for ach in ach_levels:
                current += 1
                
                print(f"\n🎬 Simulação {current}/{total}")
                print(f"  Config: {scenario_name}, {occupants} pessoas, ACH={ach}")
                
                try:
                    scenario = cfg.get_scenario_config(scenario_name)
                    scenario.total_occupants = occupants
                    
                    for zone in scenario.zones:
                        zone.target_ach = ach
                    
                    physics_config = cfg.PhysicsConfig(
                        cell_size=0.2,
                        kalman_enabled=False
                    )
                    
                    model = IAQSimulationModel(
                        scenario=scenario,
                        physics_config=physics_config,
                        simulation_duration_hours=1.0,
                        real_time_factor=20.0,
                        use_learning_agents=False
                    )
                    
                    while model.running:
                        model.step()
                    
                    metrics = model.current_metrics
                    zone_stats = model.physics.get_zone_statistics()
                    
                    if zone_stats:
                        avg_co2 = np.mean([s['concentrations']['co2_ppm_mean'] 
                                         for s in zone_stats.values()])
                    else:
                        avg_co2 = metrics['average_co2']
                    
                    score = (
                        (1 - min(avg_co2 / 1500, 1)) * 0.4 +
                        (1 - metrics['infection_risk']) * 0.3 +
                        metrics['comfort_index'] * 0.3
                    )
                    
                    results.append({
                        'scenario': scenario_name,
                        'occupants': occupants,
                        'target_ach': ach,
                        'final_co2': avg_co2,
                        'infection_risk': metrics['infection_risk'],
                        'comfort_index': metrics['comfort_index'],
                        'energy_consumption': metrics['energy_consumption'],
                        'composite_score': score
                    })
                    
                    print(f"  ✅ Concluído | CO₂: {avg_co2:.0f}ppm | Score: {score:.3f}")
                    
                except Exception as e:
                    print(f"  ❌ Erro: {str(e)}")
    
    # Analisar resultados
    if not results:
        print("\n❌ Nenhuma simulação concluída")
        return
    
    df = pd.DataFrame(results)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("data", "results", f"batch_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    csv_file = os.path.join(results_dir, "batch_results.csv")
    df.to_csv(csv_file, index=False)
    
    print("\n" + "=" * 70)
    print("📊 ANÁLISE ESTATÍSTICA")
    print("=" * 70)
    
    best = df.loc[df['composite_score'].idxmax()]
    print(f"\n🏆 MELHOR CENÁRIO (Score: {best['composite_score']:.3f}):")
    print(f"  Cenário: {best['scenario']}")
    print(f"  Ocupantes: {best['occupants']}")
    print(f"  ACH: {best['target_ach']:.1f}")
    print(f"  CO₂: {best['final_co2']:.0f} ppm")
    
    print(f"\n💾 Resultados salvos em: {csv_file}")
    print("\n✅ Análise em lote concluída!")


if __name__ == "__main__":
    run_batch_analysis()
