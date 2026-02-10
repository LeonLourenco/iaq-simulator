#!/usr/bin/env python3
"""
ARQUIVO PRINCIPAL DE EXECUÇÃO
Script para executar o simulador de forma independente (CLI) ou integrada (GUI).
"""

import sys
import os
import argparse
import json
import time
from datetime import datetime
import numpy as np

# Adicionar diretório atual ao path para garantir imports locais
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """Verifica se os arquivos essenciais do projeto existem."""
    # LISTA DE ARQUIVOS NECESSÁRIOS
    required = [
        'main_model.py', 
        'config_final.py', 
        'unified_physics.py', 
        'advanced_agents.py',
        'final_dashboard.py'  # <--- Nome corrigido aqui
    ]
    
    missing = [f for f in required if not os.path.exists(f)]
    
    if missing:
        print(f"❌ Erro Crítico: Arquivos faltando no diretório: {', '.join(missing)}")
        print("Certifique-se de estar executando o script na raiz do projeto.")
        sys.exit(1)

def main():
    """Função principal de execução."""
    parser = argparse.ArgumentParser(
        description="Simulador IAQ Avançado - Versão Final",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  %(prog)s --gui                          # Iniciar Interface Gráfica
  %(prog)s -s school -d 8 -v              # Escola, 8 horas, com gráficos
  %(prog)s -s office -o 100 -e dados.json # Escritório, 100 pessoas, exportar JSON
        """
    )
    
    # Argumentos
    parser.add_argument('--scenario', '-s', default='school',
                        choices=['school', 'office', 'gym', 'hospital', 'residential'],
                        help='Cenário a ser simulado')
    
    parser.add_argument('--duration', '-d', type=float, default=8.0,
                        help='Duração da simulação em horas')
    
    parser.add_argument('--occupants', '-o', type=int, default=50,
                        help='Número total de ocupantes')
    
    parser.add_argument('--infected-ratio', '-i', type=float, default=0.03,
                        help='Razão inicial de infectados (0.0 a 1.0)')
    
    parser.add_argument('--export', '-e', type=str,
                        help='Nome do arquivo JSON para exportação')
    
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Gerar gráficos PNG ao final da simulação')
    
    parser.add_argument('--gui', '-g', action='store_true',
                        help='Iniciar interface gráfica completa (Streamlit)')
    
    parser.add_argument('--batch', '-b', type=str,
                        help='Caminho para arquivo JSON de configuração batch')
    
    parser.add_argument('--output-dir', '-od', type=str, default='results',
                        help='Diretório para salvar resultados')
    
    parser.add_argument('--time-step', '-t', type=float, default=1.0,
                        help='Passo de tempo máximo da simulação (segundos)')
    
    parser.add_argument('--real-time-factor', '-rtf', type=float, default=1.0,
                        help='Fator de aceleração (apenas informativo para este script)')
    
    args = parser.parse_args()
    
    # --- MODO GUI ---
    if args.gui:
        check_dependencies()
        print("🚀 Iniciando Dashboard Interativo...")
        print(f"📂 Executando: streamlit run final_dashboard.py")
        os.system("streamlit run final_dashboard.py")
        return
    
    # --- MODO BATCH ---
    if args.batch:
        check_dependencies()
        run_batch_simulation(args.batch, args.output_dir)
        return
    
    # --- MODO CLI PADRÃO ---
    check_dependencies()
    
    print(f"""
╔{'═'*60}╗
║{'SIMULADOR IAQ - MODO CLI':^60}║
╚{'═'*60}╝
""")
    
    print(f"🔧 Configuração:")
    print(f"   Cenário: {args.scenario}")
    print(f"   Duração: {args.duration} horas")
    print(f"   Ocupantes: {args.occupants}")
    print(f"   Infectados: {args.infected_ratio*100:.1f}%")
    print()
    
    try:
        # Importações tardias (dentro do try para capturar erros de script)
        import config_final as cfg
        from main_model import IAQSimulationModel
        
        # 1. Configurar Cenário
        print("📋 Preparando ambiente...")
        scenario = cfg.get_scenario_config(args.scenario)
        scenario.total_occupants = args.occupants
        scenario.initial_infected_ratio = args.infected_ratio
        
        # 2. Configurar Física
        physics_config = cfg.PhysicsConfig(
            cell_size=0.5,           # Otimizado para CLI (mais rápido)
            dt_max=args.time_step,   # Passo de tempo definido pelo usuário
            stability_safety_factor=0.8,
            kalman_enabled=True,
            pem_correction_active=True
        )
        
        # 3. Inicializar Modelo
        print("🚀 Inicializando modelo...")
        model = IAQSimulationModel(
            scenario=scenario,
            physics_config=physics_config,
            simulation_duration_hours=args.duration,
            use_learning_agents=False
        )
        
        # Condições iniciais padrão
        model.physics.set_external_conditions(
            temperature_c=22.0, humidity_percent=50.0, co2_ppm=400
        )
        
        # 4. Loop de Execução
        print("▶️  Executando simulação...")
        print("=" * 60)
        
        start_time = time.time()
        simulation_time = 0
        total_steps_est = int(args.duration * 3600 / model.dt)
        step_count = 0
        
        while model.running:
            model.step()
            simulation_time = model.time
            step_count += 1
            
            # Atualização da barra de progresso (a cada ~100 passos ou 5% para não flodar o terminal)
            if step_count % max(1, int(total_steps_est/20)) == 0:
                elapsed_real = time.time() - start_time
                progress = min(simulation_time / (args.duration * 3600), 1.0)
                bars = int(progress * 40)
                
                print(f"\r⏱️  Tempo Sim: {simulation_time/3600:5.2f}h | "
                      f"[{'█'*bars}{'░'*(40-bars)}] {progress*100:5.1f}% | "
                      f"Real: {elapsed_real:.1f}s", end="", flush=True)
        
        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✅ Simulação finalizada em {elapsed_time:.2f} segundos.")
        
        # 5. Resultados e Exportação
        print("\n📊 Calculando estatísticas finais...")
        summary = model.get_simulation_summary()
        metrics = model.current_metrics
        
        print("\n📊 RESULTADOS PRINCIPAIS")
        print("-" * 30)
        print(f"📅 Cenário: {summary['scenario']}")
        print(f"👥 Total Agentes: {summary['total_agents']}")
        print(f"🤒 Infecções: {summary.get('total_infections', 0)}")
        print(f"💨 CO₂ Final: {metrics.get('average_co2', 0):.0f} ppm")
        print(f"⚠️  Risco Infecção: {metrics.get('infection_risk', 0)*100:.2f}%")
        print(f"⚡ Energia: {metrics.get('energy_consumption', 0):.2f} kWh")
        
        # Exportação JSON
        if args.export:
            os.makedirs(args.output_dir, exist_ok=True)
            # Garante extensão .json
            fname = args.export if args.export.endswith('.json') else f"{args.export}.json"
            json_path = os.path.join(args.output_dir, fname)
            
            with open(json_path, 'w') as f:
                f.write(model.export_simulation_data('json'))
            print(f"\n💾 JSON salvo em: {json_path}")
            
        # Exportação CSV Histórico (Automático se houver dados)
        if model.simulation_data['time']:
            import pandas as pd
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Construir lista plana de dicionários
            rows = []
            for i, t in enumerate(model.simulation_data['time']):
                row = {'time_s': t, 'time_h': t/3600}
                # Merge seguro de listas
                if i < len(model.simulation_data['risk_metrics']):
                    row.update(model.simulation_data['risk_metrics'][i])
                if i < len(model.simulation_data['energy_consumption']):
                    row.update(model.simulation_data['energy_consumption'][i])
                rows.append(row)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_path = os.path.join(args.output_dir, f"history_{args.scenario}_{timestamp}.csv")
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"💾 CSV Histórico salvo em: {csv_path}")
        
        # 6. Visualização
        if args.visualize:
            print("\n📈 Gerando gráficos...")
            generate_cli_visualizations(model, args.output_dir)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Simulação interrompida pelo usuário (Ctrl+C).")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro Fatal: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def run_batch_simulation(config_file: str, output_dir: str):
    """Executa múltiplas simulações em lote a partir de um JSON."""
    try:
        if not os.path.exists(config_file):
            print(f"❌ Arquivo de configuração não encontrado: {config_file}")
            return

        with open(config_file, 'r') as f:
            configs = json.load(f)
        
        print(f"📋 Iniciando Batch: {len(configs)} simulações")
        print("=" * 70)
        
        results = []
        
        # Imports locais
        import config_final as cfg
        from main_model import IAQSimulationModel
        
        for i, config in enumerate(configs):
            print(f"\n🎬 Simulação {i+1}/{len(configs)}")
            
            try:
                # Configuração
                scen_name = config.get('scenario', 'school')
                scenario = cfg.get_scenario_config(scen_name)
                scenario.total_occupants = config.get('occupants', 50)
                scenario.initial_infected_ratio = config.get('infected_ratio', 0.03)
                
                duration = config.get('duration', 8.0)
                
                # Física
                physics_config = cfg.PhysicsConfig(
                    cell_size=config.get('cell_size', 0.5),
                    dt_max=60.0,
                    kalman_enabled=True
                )
                
                # Modelo
                model = IAQSimulationModel(
                    scenario=scenario,
                    physics_config=physics_config,
                    simulation_duration_hours=duration
                )
                
                # Execução silenciosa
                start_t = time.time()
                while model.running:
                    model.step()
                elapsed = time.time() - start_t
                
                # Coleta
                summary = model.get_simulation_summary()
                results.append({
                    'config_id': i,
                    'scenario': scen_name,
                    'summary': summary,
                    'elapsed_time': elapsed,
                    'status': 'success'
                })
                print(f"   ✅ Sucesso ({elapsed:.1f}s) - Infecções: {summary.get('total_infections',0)}")
                
            except Exception as e:
                print(f"   ❌ Falha: {e}")
                results.append({'config_id': i, 'status': 'failed', 'error': str(e)})
        
        # Salvar Batch Result
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        print(f"\n💾 Relatório de Batch salvo em: {out_file}")

    except Exception as e:
        print(f"❌ Erro no Batch Runner: {e}")

def generate_cli_visualizations(model, output_dir: str):
    """Gera gráficos estáticos sem necessidade de interface gráfica."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # IMPORTANTE: Backend não-interativo para evitar erros de display
        import matplotlib.pyplot as plt
        
        os.makedirs(output_dir, exist_ok=True)
        data = model.simulation_data
        
        if not data['time']:
            print("   ⚠️ Sem dados temporais para gerar gráficos.")
            return
        
        # Preparação dos dados
        times_h = [t / 3600 for t in data['time']]
        
        # Coleta de métricas
        co2_vals = []
        if data['zone_stats']:
            for zs in data['zone_stats']:
                # Média simples de todas as zonas no instante t
                mean = np.mean([z['concentrations']['co2_ppm_mean'] for z in zs.values()])
                co2_vals.append(mean)
        
        risk_vals = []
        if data['risk_metrics']:
            risk_vals = [r['infection_risk'] * 100 for r in data['risk_metrics']]
        
        # Plotagem
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Subplot 1: CO2
        if co2_vals:
            ax1.plot(times_h, co2_vals, color='tab:orange', label='CO₂ Médio')
            ax1.axhline(y=model.scenario.co2_setpoint, color='red', linestyle='--', alpha=0.5, label='Setpoint')
            ax1.set_ylabel('Concentração (ppm)')
            ax1.set_title('Qualidade do Ar')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Risco
        if risk_vals:
            ax2.plot(times_h, risk_vals, color='tab:purple', label='Risco Acumulado')
            ax2.set_ylabel('Probabilidade (%)')
            ax2.set_xlabel('Tempo de Simulação (h)')
            ax2.set_title('Risco de Infecção (Wells-Riley)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        # Salvar
        img_path = os.path.join(output_dir, 'simulation_summary.png')
        plt.savefig(img_path, dpi=150)
        plt.close()
        
        print(f"   🖼️  Gráfico salvo: {img_path}")
        
    except ImportError:
        print("   ⚠️ Matplotlib não instalado (pip install matplotlib).")
    except Exception as e:
        print(f"   ⚠️ Erro na plotagem: {e}")

def generate_simulation_report(model, output_dir: str):
    """Gera um arquivo Markdown com o relatório final."""
    try:
        report_path = os.path.join(output_dir, 'simulation_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Relatório de Simulação IAQ\n\n")
            f.write(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            
            summary = model.get_simulation_summary()
            metrics = model.current_metrics
            
            f.write("## 1. Parâmetros\n")
            f.write(f"- **Cenário:** {summary['scenario']}\n")
            f.write(f"- **Duração:** {summary['duration_hours']:.2f} h\n")
            f.write(f"- **Ocupantes:** {summary['total_agents']}\n\n")
            
            f.write("## 2. Resultados Chave\n")
            f.write(f"- **CO₂ Médio Final:** {metrics.get('average_co2', 0):.0f} ppm\n")
            f.write(f"- **Risco de Infecção:** {metrics.get('infection_risk', 0)*100:.2f}%\n")
            f.write(f"- **Total Infectados:** {summary.get('total_infections', 0)}\n")
            f.write(f"- **Energia Consumida:** {metrics.get('energy_consumption', 0):.2f} kWh\n")
            
        print(f"   📝 Relatório MD salvo: {report_path}")
        
    except Exception as e:
        print(f"   ⚠️ Erro ao gerar relatório MD: {e}")

if __name__ == "__main__":
    main()