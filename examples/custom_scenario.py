#!/usr/bin/env python3
"""
Exemplo de Cenário Personalizado
Cria e executa um cenário totalmente personalizado
"""

import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config_final as cfg
from main_model import IAQSimulationModel


def create_custom_scenario():
    """Cria e executa um cenário totalmente personalizado."""
    
    print("🛠️  Criando cenário personalizado")
    print("=" * 70)
    
    # Definir zonas personalizadas
    zones = [
        cfg.ZoneConfig(
            name="Sala Principal",
            zone_type=cfg.ZoneType.WORKSPACE,
            x=0, y=0, width=12, height=10,
            occupancy_density=0.8,
            target_ach=3.5,
            ventilation_efficiency=0.85,
            materials=["carpet", "gypsum_board", "wood"]
        ),
        cfg.ZoneConfig(
            name="Cozinha",
            zone_type=cfg.ZoneType.KITCHEN,
            x=12, y=0, width=8, height=6,
            occupancy_density=0.4,
            target_ach=8.0,
            ventilation_efficiency=0.7,
            materials=["tile", "stainless_steel"]
        )
    ]
    
    # Criar cenário personalizado
    scenario = cfg.BuildingScenario(
        name="MinhaCasaPersonalizada",
        building_type=cfg.BuildingType.RESIDENTIAL,
        total_width=20.0,
        total_height=15.0,
        floor_height=2.8,
        total_occupants=8,
        initial_infected_ratio=0.125,
        co2_setpoint=900,
        temperature_setpoint=23.0,
        humidity_setpoint=55.0,
        overall_ventilation_strategy="mixed_mode",
        zones=zones,
        agent_config=cfg.AgentConfig(
            intelligence_level="adaptive",
            movement_pattern="waypoint",
            mask_wearing_prob=0.5,
            compliance_rate=0.9
        )
    )
    
    # Criar modelo
    print("Criando modelo...")
    model = IAQSimulationModel(
        scenario=scenario,
        physics_config=cfg.PhysicsConfig(cell_size=0.15),
        simulation_duration_hours=6.0,
        real_time_factor=5.0,
        use_learning_agents=True
    )
    
    # Executar simulação
    print("\n▶️  Executando simulação...")
    
    interventions_applied = []
    step_count = 0
    
    while model.running:
        model.step()
        step_count += 1
        
        # Aplicar intervenções dinâmicas
        if step_count % 100 == 0:
            if model.time >= 3600 and "mask_mandate" not in interventions_applied:
                model.apply_interventions("mask_mandate", {"compliance": 0.95})
                interventions_applied.append("mask_mandate")
                print(f"\n  🕐 {model.time/3600:.1f}h: Máscaras obrigatórias")
            
            if model.current_metrics['average_co2'] > 1000:
                if "increase_ventilation" not in interventions_applied:
                    model.apply_interventions("increase_ventilation", {"factor": 1.3})
                    interventions_applied.append("increase_ventilation")
                    print(f"\n  🕐 {model.time/3600:.1f}h: Ventilação aumentada")
        
        if step_count % 500 == 0:
            progress = model.time / (6.0 * 3600) * 100
            print(f"\r⏱️  {progress:5.1f}% | "
                  f"CO₂: {model.current_metrics['average_co2']:.0f}ppm | "
                  f"Risco: {model.current_metrics['infection_risk']*100:5.1f}%", 
                  end="", flush=True)
    
    print("\n" + "=" * 70)
    print("✅ Simulação concluída!")
    
    # Exibir resultados
    print("\n📊 RESULTADOS")
    print("=" * 70)
    
    metrics = model.current_metrics
    print(f"CO₂ médio: {metrics['average_co2']:.0f} ppm")
    print(f"Risco de infecção: {metrics['infection_risk']*100:.1f}%")
    print(f"Conforto: {metrics['comfort_index']*100:.1f}%")
    print(f"Energia: {metrics['energy_consumption']:.2f} kWh")
    
    print(f"\n🛡️  Intervenções aplicadas: {len(interventions_applied)}")
    for i, interv in enumerate(interventions_applied, 1):
        print(f"  {i}. {interv}")
    
    # Salvar cenário
    results_dir = os.path.join("data", "results", "custom_scenario")
    os.makedirs(results_dir, exist_ok=True)
    
    scenario_dict = {
        'name': scenario.name,
        'building_type': scenario.building_type.value,
        'zones': [{'name': z.name, 'type': z.zone_type.value} for z in zones]
    }
    
    with open(os.path.join(results_dir, "custom_scenario.json"), 'w') as f:
        json.dump(scenario_dict, f, indent=2)
    
    print(f"\n💾 Cenário salvo em: {results_dir}")
    print("\n🎉 Cenário personalizado concluído!")


if __name__ == "__main__":
    create_custom_scenario()
