"""
IAQ Simulator - Agents Module
=============================

Este módulo define a entidade biológica (BioAgent).
Ele é responsável por:
1. Comportamento Social: Ir para a carteira, socializar no intervalo, distanciamento.
2. Fisiologia: Taxa de respiração, emissão de CO2 e Vírus (Quanta).
3. Epidemiologia: Modelo de Dose-Resposta (Wells-Riley) para infecção.

Autor: Leon Lourenço (UFRPE)
Licença: MIT
"""

import numpy as np

# --- CONSTANTES DE CALIBRAÇÃO (Baseadas em Literatura/Ajuste Fino) ---
# Velocidade média de caminhada (m/s convertida para células/step)
# Assumindo dt ~ 1s e célula ~ 0.1m -> 0.6 m/s = 6 células/s
BASE_SPEED = 0.6 

# Taxa de Emissão de CO2 (Arbitrária para gerar PPM realista no LBM)
# Valor alto (100.0) compensa a difusão numérica para atingir ~1000-2000 ppm
BASE_CO2_RATE = 100.0  

# Carga Viral (Quanta/h)
# Respirando (Aula) vs Falando (Intervalo)
BASE_QUANTA_BREATH = 10.0
BASE_QUANTA_SPEAK = 50.0

# Resistência Imunológica (k)
# Probabilidade P = 1 - exp(-Dose/k). Quanto maior k, mais difícil infectar.
# Calibrado para gerar ~30-50% de ataque em 4h sem ventilação.
IMMUNITY_FACTOR = 100.0 

class BioAgent:
    """
    Agente inteligente que simula um ocupante da sala de aula.
    """
    def __init__(self, unique_id, pos_x, pos_y, desk_x, desk_y, room_dims, time_scale=1.0, initial_status="Suscetivel"):
        """
        Inicializa o agente.
        
        Args:
            unique_id (int): Identificador único.
            pos_x, pos_y (float): Posição inicial.
            desk_x, desk_y (float): Posição da carteira (destino principal).
            room_dims (tuple): (NX, NY) Dimensões da sala.
            time_scale (float): Fator de ajuste temporal (dt).
            initial_status (str): "Suscetivel", "Infectado" ou "Assintomatico".
        """
        self.id = unique_id
        self.nx, self.ny = room_dims
        
        # Coordenadas Contínuas (Float)
        self.x = float(pos_x)
        self.y = float(pos_y)
        
        # Destinos
        self.desk_x = float(desk_x)
        self.desk_y = float(desk_y)
        self.target_x, self.target_y = self.desk_x, self.desk_y
        
        # Máquina de Estados Comportamental
        self.state_behav = "ENTRANDO" # ENTRANDO -> SENTADO <-> SOCIALIZANDO
        
        # Física
        self.speed = BASE_SPEED / time_scale
        self.time_scale = time_scale
        
        # Epidemiologia
        self.status_saude = initial_status
        self.mask_efficiency = 0.50 # Eficiência da máscara (se implementado uso universal)
        self.accumulated_dose = 0.0 # Dose viral inalada acumulada
        
        # Jitter (Movimento aleatório na carteira para não parecer estátua)
        self.jitter_intensity = 0.2

    def step_behavior(self, current_step, total_steps, walls_mask, all_agents):
        """
        Atualiza a lógica do agente para o passo atual.
        1. Decide o objetivo (Target).
        2. Executa movimento e colisão.
        """
        # --- 1. CÉREBRO: Decisão de Destino ---
        self._update_state_machine(current_step, total_steps)
        
        # --- 2. CORPO: Movimentação ---
        self._move_and_collide(walls_mask, all_agents)

    def _update_state_machine(self, step, total):
        """Define o estado comportamental baseado no tempo da aula."""
        
        # Estado 1: Entrando na sala
        if self.state_behav == "ENTRANDO":
            dist = np.hypot(self.x - self.desk_x, self.y - self.desk_y)
            # Se chegou perto da mesa (2 células), senta
            if dist < 2.0: 
                self.state_behav = "SENTADO"
            
        # Estado 2: Assistindo Aula
        elif self.state_behav == "SENTADO":
            # Define o intervalo (Break) entre 45% e 55% do tempo total
            start_break = 0.45 * total
            end_break = 0.55 * total
            
            if start_break < step < end_break:
                self.state_behav = "SOCIALIZANDO"
                self._pick_random_spot() # Escolhe um lugar para conversar
            else:
                # Mantém na mesa
                self.target_x, self.target_y = self.desk_x, self.desk_y
                
                # Micro-movimentos (simula inquietação)
                if np.random.random() < (0.05 / self.time_scale):
                    self.x += np.random.uniform(-self.jitter_intensity, self.jitter_intensity)
                    self.y += np.random.uniform(-self.jitter_intensity, self.jitter_intensity)

        # Estado 3: Intervalo / Socialização
        elif self.state_behav == "SOCIALIZANDO":
            # Fim do intervalo? Voltar para mesa.
            end_break = 0.55 * total
            if step > end_break:
                self.state_behav = "ENTRANDO" # Usa lógica de entrar para voltar à mesa
                self.target_x, self.target_y = self.desk_x, self.desk_y
            
            # Dinâmica de Grupo: Muda de lugar a cada X minutos simulados
            # 50 steps * time_scale
            change_freq = int(50 * self.time_scale)
            if step % change_freq == 0:
                self._pick_random_spot()

    def _pick_random_spot(self):
        """Escolhe um ponto aleatório na sala (longe das paredes) para socializar."""
        margin = 10 # Margem de segurança das paredes
        self.target_x = np.random.randint(margin, self.nx - margin)
        self.target_y = np.random.randint(margin, self.ny - margin)

    def _move_and_collide(self, walls, agents):
        """Executa movimento vetorial com repulsão social e colisão com paredes."""
        # Vetor para o alvo
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = np.hypot(dx, dy)
        
        vx, vy = 0.0, 0.0
        
        # Se está longe, move-se em direção ao alvo
        if dist > 0.5:
            vx = (dx / dist) * self.speed
            vy = (dy / dist) * self.speed

        # --- Força de Repulsão Social (Evita ficar em cima de outro aluno) ---
        for other in agents:
            if other.id != self.id:
                # Distância euclidiana
                d_ag = np.hypot(self.x - other.x, self.y - other.y)
                min_dist = 1.5 # Raio pessoal (1.5 células = 15cm na escala atual, evitar sobreposição exata)
                
                if d_ag < min_dist and d_ag > 0:
                    # Vetor de afastamento
                    push_x = self.x - other.x
                    push_y = self.y - other.y
                    # Força inversamente proporcional à distância
                    factor = (min_dist - d_ag) / min_dist
                    
                    vx += push_x * factor * (0.8 / self.time_scale)
                    vy += push_y * factor * (0.8 / self.time_scale)

        # --- Colisão com Paredes (Algoritmo "Slide") ---
        # Tenta mover em X e Y
        next_x = self.x + vx
        next_y = self.y + vy
        
        if self._is_valid_position(next_x, next_y, walls):
            self.x = next_x
            self.y = next_y
        else:
            # Se bloqueado diagonalmente, tenta mover apenas em X
            if self._is_valid_position(next_x, self.y, walls):
                self.x = next_x
            # Se bloqueado em X, tenta mover apenas em Y
            elif self._is_valid_position(self.x, next_y, walls):
                self.y = next_y
            # Se bloqueado em tudo, fica parado (paredes absorvem o movimento)

    def _is_valid_position(self, x, y, walls):
        """Verifica se a coordenada (x,y) está dentro da sala e fora de paredes."""
        ix, iy = int(x), int(y)
        # Limites do Grid
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            # Verifica máscara de obstáculos (True = Parede/Mesa)
            # Nota: Permitimos andar sobre "Mesa" (OBJ_DESK) se necessário, 
            # mas idealmente o walls_mask passado deve definir onde é proibido pisar.
            return not walls[iy, ix]
        return False

    def get_emissions(self):
        """
        Calcula a quantidade de contaminantes emitidos neste passo de tempo.
        Retorna: (Virus_Quanta, CO2_Mass)
        """
        # Emissão de CO2 é constante (metabolismo basal ajustado)
        co2 = BASE_CO2_RATE / self.time_scale
        
        virus = 0.0
        # Apenas infectados emitem vírus
        if self.status_saude == "Infectado":
            # Emite mais se estiver socializando (falando) do que sentado (respirando)
            if self.state_behav == "SOCIALIZANDO":
                base_v = BASE_QUANTA_SPEAK
            else:
                base_v = BASE_QUANTA_BREATH
            
            # Ajusta por eficiência da máscara e passo de tempo
            virus = (base_v / self.time_scale) * (1.0 - self.mask_efficiency) * 0.1
            
        return virus, co2

    def update_infection_risk(self, virus_conc_at_pos):
        """
        Calcula o risco de infecção baseado na concentração local (Modelo Wells-Riley).
        
        Args:
            virus_conc_at_pos (float): Concentração de quanta/m³ na célula atual do agente.
        """
        # Se já é infectado ou assintomático, não faz nada
        if self.status_saude != "Suscetivel":
            return

        # Taxa de respiração média ~0.5 m³/h. Ajuste simplificado:
        # Dose += Concentração * (Fator Respiração / TimeScale)
        # Fator 0.1 é um escalar empírico para ajustar a magnitude da dose no tempo simulado
        dose_increment = virus_conc_at_pos * 0.1 
        self.accumulated_dose += dose_increment
        
        # Probabilidade de Infecção P = 1 - exp(-Dose / k)
        infection_prob = 1.0 - np.exp(-self.accumulated_dose / IMMUNITY_FACTOR)
        
        # Sorteio estocástico (Monte Carlo)
        if np.random.random() < infection_prob:
            # Transição de Estado: S -> A (Assintomático/Incubando)
            # Escolhemos "Assintomático" para diferenciar visualmente quem começou doente (I)
            # e quem pegou na sala (A).
            self.status_saude = "Assintomatico"