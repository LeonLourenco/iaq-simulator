# Referência Rápida - Simulador IAQ

## Instalação Rápida
```bash
pip install -r requirements.txt
```

## Executar

### Dashboard
```bash
streamlit run final_dashboard.py
```

### CLI
```bash
python run_simulation.py --scenario office --duration 8
```

### Python
```python
from main_model import IAQSimulationModel
import config_final as cfg

model = IAQSimulationModel(
    scenario=cfg.get_scenario_config('office'),
    physics_config=cfg.PhysicsConfig()
)

while model.running:
    model.step()
```

## Cenários Disponíveis
- `school` - Sala de aula
- `office` - Escritório open space
- `gym` - Academia
- `hospital` - Quarto hospitalar
- `residential` - Residência

## Parâmetros Principais

### PhysicsConfig
- `cell_size` (0.1-1.0m): Resolução espacial
- `kalman_enabled` (bool): Filtro de Kalman
- `pem_correction_active` (bool): Correção de pluma

### BuildingScenario
- `total_occupants` (int): Número de pessoas
- `initial_infected_ratio` (0-1): Taxa de infectados
- `co2_setpoint` (ppm): Limite de CO₂
- `overall_ventilation_strategy`: demand_controlled, constant_volume, natural

### ZoneConfig
- `target_ach` (0.5-20): Trocas de ar por hora
- `occupancy_density` (pessoas/m²): Densidade
- `ventilation_efficiency` (0-1): Eficiência

## Intervenções
```python
# Máscaras
model.apply_interventions("mask_mandate", {"compliance": 0.9})

# Ventilação
model.apply_interventions("increase_ventilation", {"factor": 1.5})

# Ocupação
model.apply_interventions("reduce_occupancy", {"reduction": 0.3})
```

## Métricas
```python
metrics = model.current_metrics

print(metrics['average_co2'])        # CO₂ médio (ppm)
print(metrics['infection_risk'])     # Risco (0-1)
print(metrics['comfort_index'])      # Conforto (0-1)
print(metrics['energy_consumption']) # Energia (kWh)
```

## Exportar Dados
```python
# JSON
json_data = model.export_simulation_data('json')

# Resumo
summary = model.get_simulation_summary()

# Visualização
viz_data = model.get_visualization_data()
```

## Testes
```bash
# Executar todos
pytest

# Específicos
pytest tests/test_physics.py
pytest tests/test_agents.py
pytest tests/test_integration.py
```

## Valores Recomendados

### CO₂
- < 600 ppm: Excelente
- 600-800 ppm: Bom
- 800-1000 ppm: Moderado
- > 1000 ppm: Ruim

### ACH (Trocas/hora)
- 2-4: Mínimo para ocupação normal
- 6-8: Recomendado para saúde
- 10+: Espaços de alto risco

### Temperatura
- 20-24°C: Ideal
- 18-26°C: Aceitável

### Umidade
- 40-60%: Ideal
- 30-70%: Aceitável

## Troubleshooting

### Simulação lenta
- Aumente `cell_size` para 0.3-0.5
- Desative `kalman_enabled`
- Reduza número de ocupantes

### Erro de memória
- Aumente `cell_size`
- Reduza área simulada

### Resultados estranhos
- Verifique parâmetros de entrada
- Consulte logs
- Teste com cenário pré-configurado

## Ajuda
- Documentação: `docs/`
- Exemplos: `examples/`
- Issues: GitHub
