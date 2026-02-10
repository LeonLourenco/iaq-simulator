# Simulador IAQ

Simulador de Qualidade do Ar Interno (IAQ) integrado com física CFD, agentes inteligentes e otimização multiobjetivo.

## 🚀 Características Principais

- **Motor Físico Unificado**: Simulação CFD multi-espécies (CO₂, HCHO, vírus, calor, umidade)
- **Agentes Inteligentes**: Comportamento adaptativo com aprendizado por reforço
- **Dashboard Interativo**: Interface Streamlit com visualizações 3D em tempo real
- **Otimização Automática**: Balanceamento entre IAQ, conforto e eficiência energética
- **Cenários Configuráveis**: Escola, escritório, hospital, academia, residencial

## 📋 Pré-requisitos

- Python 3.8 ou superior
- 4GB RAM mínimo (8GB recomendado)
- 500MB de espaço em disco

## 🛠️ Instalação

```bash
# Clone o repositório
git clone https://github.com/LeonLourenco/iaq-simulator.git
cd iaq-simulator

# Instale as dependências
pip install -r requirements.txt
```

## 🏃‍♂️ Uso Rápido

### Interface Gráfica
```bash
streamlit run final_dashboard.py
```

### Linha de Comando
```bash
python run_simulation.py --scenario office --duration 8 --visualize
```

### Como Script Python
```python
from main_model import IAQSimulationModel
import config_final as cfg

scenario = cfg.get_scenario_config('office')
physics_config = cfg.PhysicsConfig()

model = IAQSimulationModel(scenario, physics_config)
while model.running:
    model.step()

print(f"CO₂ médio: {model.current_metrics['average_co2']:.0f} ppm")
```

## 📁 Estrutura do Projeto

```
iaq-simulator/
├── config_final.py          # Configurações e constantes
├── unified_physics.py       # Motor físico unificado
├── advanced_agents.py       # Agentes inteligentes
├── main_model.py           # Modelo principal de simulação
├── final_dashboard.py      # Dashboard Streamlit
├── run_simulation.py       # Interface linha de comando
├── requirements.txt        # Dependências
├── README.md              # Documentação
└── LICENSE                # Licença MIT

data/                      # Dados e configurações
├── scenarios/             # Cenários pré-definidos
├── materials/            # Propriedades de materiais
└── results/              # Resultados de simulação

examples/                  # Exemplos de uso
tests/                     # Testes unitários
docs/                      # Documentação detalhada
```

## 🧪 Testes

```bash
pytest tests/ -v
```

## 📊 Casos de Uso

1. **Projeto de Edifícios**: Otimização de sistemas HVAC
2. **Gestão de Pandemias**: Avaliação de intervenções
3. **Certificação Sustentável**: Análise para LEED/WELL
4. **Pesquisa Acadêmica**: Estudos de transmissão aérea
5. **Treinamento**: Educação em IAQ e controle de infecções

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🙏 Agradecimentos

- [Mesa Framework](https://mesa.readthedocs.io/) para simulação multiagente
- [Streamlit](https://streamlit.io/) para dashboard interativo
- [Plotly](https://plotly.com/python/) para visualizações

---
Desenvolvido com ❤️ para melhorar a qualidade do ar interno e a saúde dos ocupantes.
