"""
IAQ Simulator - Simulation Engine
=================================

Este módulo é o "Maestro" da simulação. Ele é responsável por:
1. Inicializar o ambiente físico (Geometria, Barreiras, Inlets/Outlets).
2. Gerenciar o loop principal de tempo.
3. Sincronizar a Física de Fluidos (LBM) com a Biologia (Agentes).
4. Coletar e comprimir dados para visualização.

Autor: Leon Lourenço (UFRPE)
Licença: MIT
"""

import numpy as np
import json
import os
import shutil
from src.lbm_core import lbm_step_classroom, scalar_transport
from src.agents import BioAgent

# --- Constantes de Objetos (Devem bater com app.py) ---
OBJ_EMPTY = 0
OBJ_WALL = 1
OBJ_DESK = 2
OBJ_AC = 3
OBJ_WINDOW = 4
OBJ_DOOR = 5

class IAQSimulator:
    def __init__(self, config_file="scenarios/school.json", config_overrides=None):
        """
        Inicializa o simulador carregando o cenário padrão.
        
        Args:
            config_file (str): Caminho para o JSON base.
            config_overrides (dict): Dicionário com valores da UI para sobrescrever o JSON.
        """
        # Carrega configuração base
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Arquivo de cenário não encontrado: {config_file}")
            
        with open(config_file, 'r') as f:
            self.cfg = json.load(f)
            
        # --- APLICAÇÃO DE CONFIGURAÇÕES DO PAINEL (OVERRIDES) ---
        # Isso garante que o slider da UI altere a física real antes de definir o Grid
        if config_overrides:
            if 'physics' in config_overrides:
                self.cfg['physics'].update(config_overrides['physics'])
            if 'agents' in config_overrides:
                self.cfg['agents'].update(config_overrides['agents'])
            if 'ventilation' in config_overrides:
                self.cfg['ventilation'].update(config_overrides['ventilation'])
            
        # Define Geometria do Grid (Escala: 1 célula = 0.1m)
        # Ex: 10.0m -> 100 células
        self.scale = 10.0 # células por metro
        self.NX = int(self.cfg['physics']['width_m'] * self.scale)
        self.NY = int(self.cfg['physics']['height_m'] * self.scale)
        
        # Inicializa Distribuição LBM (F) com ruído aleatório para quebrar simetria
        # 9 canais (D2Q9), Altura (NY), Largura (NX)
        self.omega = 1.0 / 0.54 # Tau = 0.54 (Viscosidade do ar ajustada para estabilidade)
        self.F = np.ones((9, self.NY, self.NX)) 
        for i in range(9): 
            self.F[i] += 0.01 * np.random.randn(self.NY, self.NX)
        
        # Constrói o layout estático com as novas dimensões
        self._build_environment()
        
    def _build_environment(self):
        """
        Constrói a matriz de layout e as máscaras físicas (paredes, inlets, outlets).
        Distribui as carteiras baseada na configuração de linhas.
        """
        self.layout = np.zeros((self.NY, self.NX), dtype=int)
        # Renomeado para base_walls_mask para permitir modificação dinâmica (janelas fechadas)
        self.base_walls_mask = np.zeros((self.NY, self.NX), dtype=bool)
        
        # 1. Paredes Externas (Borda)
        self.layout[0, :] = OBJ_WALL
        self.layout[-1, :] = OBJ_WALL
        self.layout[:, 0] = OBJ_WALL
        self.layout[:, -1] = OBJ_WALL
        
        # 2. Porta (Canto Inferior Esquerdo)
        # Porta de 1m (10 células)
        door_size = int(1.0 * self.scale)
        # Garante que a porta fique dentro dos limites da parede
        door_y = max(5, self.NY - door_size - 5)
        self.layout[door_y : door_y + door_size, 0] = OBJ_DOOR
        
        # 3. Janelas (Parede Direita)
        # Inicialmente fechadas na geometria, abertas na máscara se config permitir
        self.window_mask = np.zeros((self.NY, self.NX), dtype=bool)
        win_h = int(self.NY * 0.4) # Janela ocupa 40% da parede
        start_win = (self.NY - win_h) // 2
        
        # Desenha no layout visual
        self.layout[start_win : start_win + win_h, -1] = OBJ_WINDOW
        # Define máscara lógica
        self.window_mask[start_win : start_win + win_h, -1] = True
            
        # 4. Ar Condicionado (Split no Topo, deslocado do centro)
        ac_w = int(0.9 * self.scale) # 90cm
        ac_pos_x = int(self.NX * 0.3)
        # Proteção para não desenhar fora se a sala for muito estreita
        if ac_pos_x + ac_w < self.NX:
            self.layout[1:4, ac_pos_x : ac_pos_x + ac_w] = OBJ_AC
            self.ac_inlet_mask = (self.layout == OBJ_AC)
        else:
            self.ac_inlet_mask = np.zeros_like(self.layout, dtype=bool)
        
        # 5. Mesas / Carteiras (Distribuição Automática)
        self.desks_positions = []
        n_agents = self.cfg['agents']['total']
        n_rows = max(1, self.cfg['agents']['rows']) # Proteção div/0
        
        # Lógica de Grid para Mesas
        # Calcula colunas necessárias
        import math
        n_cols = math.ceil(n_agents / n_rows)
        
        # Margens e Espaçamento
        margin_x = int(self.NX * 0.15)
        margin_y = int(self.NY * 0.20)
        
        # Garante espaço mínimo para evitar passo zero
        avail_w = max(1, self.NX - 2 * margin_x)
        avail_h = max(1, self.NY - 2 * margin_y)
        
        step_x = avail_w // max(1, n_cols)
        step_y = avail_h // max(1, n_rows)
        
        # Tamanho da Mesa (60x40cm -> 6x4 células)
        desk_w, desk_h = 6, 4
        
        count = 0
        for r in range(n_rows):
            for c in range(n_cols):
                if count >= n_agents: break
                
                # Centro da mesa
                px = margin_x + c * step_x + step_x // 2
                py = margin_y + r * step_y + step_y // 2
                
                self.desks_positions.append((px, py))
                
                # Desenha mesa no layout (obstáculo físico)
                # Garante limites
                y_start, y_end = int(py), int(py) + desk_h
                x_start, x_end = int(px), int(px) + desk_w
                
                if y_end < self.NY - 1 and x_end < self.NX - 1:
                    self.layout[y_start:y_end, x_start:x_end] = OBJ_DESK
                
                count += 1
                
        # 6. Consolida Máscara de Obstáculos (Paredes + Mesas)
        # O fluido deve contornar as mesas
        self.base_walls_mask[self.layout == OBJ_WALL] = True
        self.base_walls_mask[self.layout == OBJ_DESK] = True
        # Porta fechada conta como parede para o fluido

    def run_simulation(self, total_hours, ach_target, ac_power, window_open, steps_per_min=10):
        """
        Executa o loop principal da simulação.
        """
        # --- Configuração de Diretório ---
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        
        # --- Configuração Temporal ---
        total_steps = int(total_hours * 60 * steps_per_min)
        # time_scale: Fator para ajustar velocidades biológicas em relação ao clock físico
        time_scale = steps_per_min / 10.0
        
        # --- Lógica Dinâmica de Paredes (Janelas) ---
        # Se janela fechada (window_open=False), ela vira parede sólida (Bounce-Back)
        current_walls_mask = self.base_walls_mask.copy()
        current_layout = self.layout.copy()
        
        if not window_open:
            current_walls_mask[self.window_mask] = True
            current_layout[self.window_mask] = OBJ_WALL # Atualiza visualização para parede
        
        # --- Inicialização dos Agentes ---
        agents = []
        total_agents = len(self.desks_positions)
        n_infected = self.cfg['agents']['infected']
        
        # Sorteia pacientes zero
        if total_agents > 0:
            # Proteção: não tentar infectar mais do que existe
            actual_infected = min(n_infected, total_agents)
            infected_indices = np.random.choice(total_agents, actual_infected, replace=False)
            
            for i, (dx, dy) in enumerate(self.desks_positions):
                status = "Infectado" if i in infected_indices else "Suscetivel"
                # Spawn na porta (entrada)
                start_x, start_y = 2.0, max(5.0, self.NY - 10.0)
                
                agent = BioAgent(
                    unique_id=i, 
                    pos_x=start_x, pos_y=start_y, 
                    desk_x=dx, desk_y=dy, 
                    room_dims=(self.NX, self.NY), 
                    time_scale=time_scale, 
                    initial_status=status
                )
                agents.append(agent)
            
        # --- Inicialização dos Campos Físicos ---
        C_virus = np.zeros((self.NY, self.NX)) # Quanta/m³
        C_co2 = np.zeros((self.NY, self.NX))   # Delta ppm (acima de 400)
        
        # Campos de Velocidade (Grid)
        u_grid = np.zeros((self.NY, self.NX))
        v_grid = np.zeros((self.NY, self.NX))
        
        # Configuração do Inlet (AC)
        vel_ac = min(0.15, ac_power * 0.005)
        
        # Cálculo do Decaimento (Baseado no ACH)
        # Se janela fechada, ventilação é mínima (infiltração)
        effective_ach = ach_target if window_open else 0.1
        decay_rate = effective_ach / (60.0 * steps_per_min)
            
        # --- Preparação para Salvar Dados (Downsampling) ---
        save_interval = max(1, int(30 * (steps_per_min / 60.0)))
        n_frames = total_steps // save_interval + 1
        
        # Alocação de Memória (Float32 para evitar overflow e garantir precisão)
        hist = {
            "virus": np.zeros((n_frames, self.NY, self.NX), dtype=np.float32),
            "co2": np.zeros((n_frames, self.NY, self.NX), dtype=np.float32),
            "ux": np.zeros((n_frames, self.NY, self.NX), dtype=np.float32),
            "uy": np.zeros((n_frames, self.NY, self.NX), dtype=np.float32),
            "pos": np.zeros((n_frames, len(agents), 2), dtype=np.float32),
            # Agent Stats: [Status(0,1,2), EmissaoCO2, EmissaoVirus]
            "agent_stats": np.zeros((n_frames, len(agents), 3), dtype=np.float32),
            "infected_total": np.zeros(n_frames, dtype=int),
            "layout": current_layout, # Usa layout atualizado (janela vs parede)
            "meta": {"steps_per_min": steps_per_min, "save_interval": save_interval}
        }
        
        # --- Spin-up (Aquecimento do Fluido) ---
        print(f"🔄 Spin-up LBM ({int(500 * time_scale)} steps)...")
        u_grid[self.ac_inlet_mask] = 0.0      # Vento X zero (sopra para baixo)
        v_grid[self.ac_inlet_mask] = -vel_ac  # Vento Y negativo
        
        # Se janela aberta, outlet passivo. Se fechada, parede (já no mask).
        outlets_passive = self.window_mask if window_open else np.zeros_like(self.window_mask)
        inlets_active = self.ac_inlet_mask.copy()
        
        for _ in range(int(500 * time_scale)):
            self.F, _, _ = lbm_step_classroom(
                self.F, current_walls_mask, inlets_active, outlets_passive, 
                u_grid, v_grid, self.omega
            )
            
        print("🚀 Iniciando Simulação Principal...")
        frame_idx = 0
        status_map = {"Suscetivel": 0, "Infectado": 1, "Assintomatico": 2}
        
        # --- LOOP PRINCIPAL ---
        for step in range(total_steps):
            
            # Reset Grids de Forçamento
            u_grid.fill(0.0); v_grid.fill(0.0)
            
            # AC Sempre ligado
            u_grid[self.ac_inlet_mask] = 0.0
            v_grid[self.ac_inlet_mask] = -vel_ac
            
            curr_inlets = self.ac_inlet_mask.copy()
            curr_outlets = np.zeros_like(self.window_mask)
            
            # Lógica Avançada de Respiração da Janela (Sopro/Exaustão)
            if window_open:
                # Ciclo Senoidal Lento (~2000 steps)
                wind_cycle = np.sin(step / (200.0 * time_scale)) 
                
                if wind_cycle > 0.2: 
                    # FASE 1: SOPRO (INLET) -> Vento entra (Rajada)
                    intensity = wind_cycle * 0.08 
                    u_grid[self.window_mask] = -intensity # Negativo = Esquerda (Entra)
                    curr_inlets = np.logical_or(curr_inlets, self.window_mask)
                    
                elif wind_cycle < -0.2:
                    # FASE 2: EXAUSTÃO ATIVA (INLET REVERSO) -> Vento sai
                    # Simula pressão negativa puxando ar para fora
                    intensity = abs(wind_cycle) * 0.08
                    u_grid[self.window_mask] = intensity # Positivo = Direita (Sai)
                    curr_inlets = np.logical_or(curr_inlets, self.window_mask)
                    
                else:
                    # FASE 3: TROCA PASSIVA (OUTLET) -> Pressão equilibra
                    curr_outlets = self.window_mask
            
            # 2. Atualização dos Agentes
            for ag in agents: 
                ag.step_behavior(step, total_steps, current_walls_mask, agents)
                
            # 3. Coleta de Emissões e Dados
            src_virus = np.zeros_like(C_virus)
            src_co2 = np.zeros_like(C_co2)
            inf_count = 0
            
            save_now = (step % save_interval == 0) and (frame_idx < n_frames)
            
            for i, ag in enumerate(agents):
                # Coleta estatísticas
                if save_now:
                    hist["pos"][frame_idx, i] = [ag.x, ag.y]
                    qv, qc = ag.get_emissions()
                    st_code = status_map.get(ag.status_saude, 0)
                    hist["agent_stats"][frame_idx, i] = [st_code, qc * time_scale, qv * time_scale]
                    if ag.status_saude != "Suscetivel": inf_count += 1
                
                # Injeta massa no grid
                ix, iy = int(ag.x), int(ag.y)
                if 0 < ix < self.NX and 0 < iy < self.NY:
                    qv, qc = ag.get_emissions()
                    src_virus[iy, ix] += qv
                    src_co2[iy, ix] += qc
                    
                    # 4. Exposição
                    ag.update_infection_risk(C_virus[iy, ix])

            # 5. Passo Físico (LBM + Escalares)
            # Resolve o fluxo de ar
            self.F, ux, uy = lbm_step_classroom(
                self.F, current_walls_mask, curr_inlets, curr_outlets, 
                u_grid, v_grid, self.omega
            )
            
            # Resolve transporte
            C_virus = scalar_transport(C_virus, ux, uy, src_virus, 0.01, decay_rate * 2.0)
            C_co2 = scalar_transport(C_co2, ux, uy, src_co2, 0.05, decay_rate)
            
            # 6. Salva Frame
            if save_now:
                hist["virus"][frame_idx] = C_virus.astype(np.float32)
                hist["co2"][frame_idx] = C_co2.astype(np.float32)
                hist["ux"][frame_idx] = ux.astype(np.float32)
                hist["uy"][frame_idx] = uy.astype(np.float32)
                hist["infected_total"][frame_idx] = inf_count
                frame_idx += 1
                
        # Corta o array para o tamanho real preenchido
        for key in hist:
            if isinstance(hist[key], np.ndarray) and len(hist[key]) > frame_idx:
                hist[key] = hist[key][:frame_idx]
                
        return hist