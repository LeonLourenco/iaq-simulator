# Resultados de Simulação

Este diretório armazena os resultados das simulações executadas.

## Estrutura

- `raw/`: Dados brutos em JSON/CSV
- `processed/`: Dados processados e agregados
- `visualizations/`: Gráficos e imagens geradas
- `reports/`: Relatórios em PDF/HTML

## Formato dos Arquivos

### JSON de Resultados
```json
{
  "metadata": {
    "scenario": "office",
    "timestamp": "2024-01-15T10:30:00",
    "duration_hours": 8.0,
    "occupants": 50
  },
  "final_metrics": {
    "average_co2": 650.5,
    "infection_risk": 0.12,
    "energy_consumption": 45.3
  },
  "time_series": [...],
  "zone_statistics": [...]
}
```

### CSV Histórico
Colunas: time_seconds, time_hours, average_co2, average_hcho, infected_agents, etc.

## Convenções de Nomeação

- `simulation_YYYYMMDD_HHMMSS_scenario_format.ext`
- Exemplo: `simulation_20240115_103000_office.json`

## Backup

Recomenda-se backup regular dos resultados importantes.
