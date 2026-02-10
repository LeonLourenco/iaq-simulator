#!/usr/bin/env python3
"""
Exemplo de Simulação Básica
Demonstração simples do simulador IAQ
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config_final as cfg
from main_model import IAQSimulationModel


def run_basic_simulation():
    """Executa uma simulação básica com configurações padrão."""
    
    print("🚀 Iniciando simulação básica do Simulador IAQ")
    print("=" * 60)
    
    # 1. Configurar cenário
    print("1. Configurando cenário...")
    scenario = cfg.get_scenario_config('office')
    scenario.total_occupants = 30
    scenario.initial_infected_ratio = 0.05
    
    # 2. Configurar física
    print("2. Configurando motor físico...")
    physics_config = cfg.PhysicsConfig(
        cell_size=0.2,
        kalman_enabled=False,
        pem_correction_active=True
    )
    
    # 3. Criar modelo
    print("3. Criando modelo de simulação...")
    model = IAQSimulationModel(
        scenario=scenario,
        physics_config=physics_config,
        simulation_duration_hours=2.0,
        real_time_factor=10.0,
        use_learning_agents=False
    )
    
    # 4. Executar simulação
    print("\n▶️  Executando simulação...")
    print("-" * 60)
    
    step_count = 0
    while model.running:
        model.step()
        step_count += 1
        
        if step_count % 100 == 0:
            progress = (model.time / (2.0 * 3600)) * 100
            print(f"\r⏱️  Progresso: {progress:5.1f}% | "
                  f"CO₂: {model.current_metrics['average_co2']:.0f} ppm | "
                  f"Infectados: {model.current_metrics['infected_agents']}", 
                  end="", flush=True)
    
    print("\n" + "=" * 60)
    print("✅ Simulação concluída!")
    
    # 5. Exibir resultados
    print("\n📊 RESULTADOS DA SIMULAÇÃO")
    print("=" * 60)
    
    metrics = model.current_metrics
    print(f"💨 CO₂ médio final: {metrics['average_co2']:.0f} ppm")
    print(f"🧪 HCHO médio final: {metrics['average_hcho']:.1f} ppb")
    print(f"🌡️  Temperatura média: {metrics['average_temperature']:.1f} °C")
    print(f"💧 Umidade média: {metrics['average_humidity']:.1f} %")
    print(f"🦠 Risco de infecção: {metrics['infection_risk']*100:.1f} %")
    print(f"😌 Índice de conforto: {metrics['comfort_index']*100:.1f} %")
    print(f"👥 Agentes infectados: {metrics['infected_agents']}")
    print(f"⚡ Consumo de energia: {metrics['energy_consumption']:.2f} kWh")
    
    # 6. Estatísticas por zona
    print("\n🏢 DESEMPENHO POR ZONA")
    print("=" * 60)
    
    zone_stats = model.physics.get_zone_statistics()
    for zone_id, stats in zone_stats.items():
        print(f"\n{stats['name']}:")
        print(f"  💨 CO₂: {stats['concentrations']['co2_ppm_mean']:.0f} ppm")
        print(f"  🌡️  Temperatura: {stats['concentrations']['temperature_c_mean']:.1f} °C")
    
    print("\n🎉 Simulação básica concluída com sucesso!")


if __name__ == "__main__":
    run_basic_simulation()
