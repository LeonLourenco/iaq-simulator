"""
IAQ Simulator - LBM Core Engine (High Performance)
==================================================

Este módulo contém as rotinas numéricas de baixo nível para a Dinâmica dos Fluidos Computacional (CFD).
Utiliza a biblioteca NUMBA (JIT) para compilar o código em velocidade de máquina (C/C++),
permitindo simulações em tempo real.

Autor: Leon Lourenço (UFRPE)
Licença: MIT
"""

import numpy as np
from numba import jit

# --- CONSTANTES LBM D2Q9 ---
NL = 9
cxs = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
cys = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
weights = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
opposite_idxs = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

@jit(nopython=True, cache=True)
def lbm_step_classroom(F, obstacle_mask, inlet_mask, outlet_mask, u_inlet_grid, v_inlet_grid, omega):
    """
    Passo LBM otimizado com Numba JIT.
    Executa Colisão, Propagação e Condições de Contorno em um único passo compilado.
    """
    ny, nx = F.shape[1], F.shape[2]

    # 1. Momentos Macroscópicos (Densidade e Velocidade)
    rho = np.sum(F, axis=0)
    rho_safe = np.maximum(rho, 1.0) # Evita divisão por zero

    ux = np.zeros((ny, nx))
    uy = np.zeros((ny, nx))
    for i in range(NL):
        ux += F[i] * cxs[i]
        uy += F[i] * cys[i]
    ux /= rho_safe
    uy /= rho_safe

    # --- TRAVA DE SEGURANÇA (Estabilidade Numérica) ---
    # Impede velocidades > 0.3 (limite Mach do LBM) que causariam explosão numérica
    limit = 0.3
    for y in range(ny):
        for x in range(nx):
            if ux[y, x] > limit: ux[y, x] = limit
            elif ux[y, x] < -limit: ux[y, x] = -limit

            if uy[y, x] > limit: uy[y, x] = limit
            elif uy[y, x] < -limit: uy[y, x] = -limit
    # --------------------------------------------------

    # 2. Colisão (BGK)
    # Relaxa a distribuição em direção ao equilíbrio
    for i in range(NL):
        cu = 3.0 * (cxs[i] * ux + cys[i] * uy)
        usqr = 1.5 * (ux**2 + uy**2)
        feq = rho * weights[i] * (1.0 + cu + 0.5 * (cu * cu) - usqr)
        F[i] = (1.0 - omega) * F[i] + omega * feq

    # 3. Bounce-Back (Paredes Sólidas)
    # Reflete partículas nas fronteiras de obstáculos
    F_old = F.copy()
    for y in range(ny):
        for x in range(nx):
            if obstacle_mask[y, x]:
                for i in range(1, NL):
                    F[opposite_idxs[i], y, x] = F_old[i, y, x]

    # 4. Inlets (Ar Condicionado / Entrada)
    # Força uma distribuição de equilíbrio baseada na velocidade de entrada
    for y in range(ny):
        for x in range(nx):
            if inlet_mask[y, x]:
                rho_in = 1.0
                u_in = u_inlet_grid[y, x]
                v_in = v_inlet_grid[y, x]
                u2 = u_in*u_in + v_in*v_in
                for i in range(NL):
                    cu = 3.0 * (cxs[i]*u_in + cys[i]*v_in)
                    F[i, y, x] = rho_in * weights[i] * (1.0 + cu + 4.5*(cu**2) - 1.5*u2)

    # 5. Outlets (Janelas / Saída)
    # Condição de Neumann (Gradiente Zero): Copia do vizinho interno
    for y in range(1, ny-1):
        for x in range(nx):
            if outlet_mask[y, x]:
                if x > nx // 2: # Saída na direita
                    for i in range(NL): F[i, y, x] = F[i, y, x-1]
                elif x < nx // 2: # Saída na esquerda
                    for i in range(NL): F[i, y, x] = F[i, y, x+1]
                elif y < ny // 2: # Saída embaixo
                    for i in range(NL): F[i, y, x] = F[i, y+1, x]
                else: # Saída em cima
                    for i in range(NL): F[i, y, x] = F[i, y-1, x]

    # 6. Streaming (Propagação)
    # Move as partículas para as células vizinhas
    for i in range(NL):
        cx, cy = cxs[i], cys[i]
        # Otimização: Uso de Slicing do Numpy compatível com Numba
        if cx == 1: F[i, :, 1:] = F[i, :, :-1]
        elif cx == -1: F[i, :, :-1] = F[i, :, 1:]
        
        if cy == 1: F[i, 1:, :] = F[i, :-1, :]
        elif cy == -1: F[i, :-1, :] = F[i, 1:, :]

    return F, ux, uy

@jit(nopython=True, cache=True)
def scalar_transport(C, ux, uy, source_mask, diffusion, decay):
    """
    Solver de Advecção-Difusão-Reação otimizado com Numba.
    Usa Diferenças Finitas explícitas para transportar contaminantes.
    """
    ny, nx = C.shape
    C_new = C.copy()
    
    # Loop explícito
    for y in range(1, ny-1):
        for x in range(1, nx-1):
            u, v = ux[y, x], uy[y, x]
            
            # Esquema Upwind para Advecção (Estabilidade)
            if u > 0: dCdx = C[y, x] - C[y, x-1]
            else:     dCdx = C[y, x+1] - C[y, x]
            
            if v > 0: dCdy = C[y, x] - C[y-1, x]
            else:     dCdy = C[y+1, x] - C[y, x]
            
            # Difusão (Diferenças Centrais)
            laplacian = C[y, x+1] + C[y, x-1] + C[y+1, x] + C[y-1, x] - 4*C[y, x]
            
            # Atualização Temporal
            C_new[y, x] = C[y, x] - (u * dCdx + v * dCdy) + diffusion * laplacian - decay * C[y, x]
            
    # Aplica Termos Fonte (Emissão dos Alunos)
    for y in range(ny):
        for x in range(nx):
            if source_mask[y, x] > 0: 
                C_new[y, x] += source_mask[y, x]
                
    return C_new