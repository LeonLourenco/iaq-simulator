"""
Módulo de Comportamentos (Behavior Strategy Pattern).
Define como os agentes tomam decisões baseadas em seu perfil (Arquétipo).
"""

import random
from abc import ABC, abstractmethod
from typing import Optional, Tuple

# Interface Base
class BehaviorStrategy(ABC):
    def __init__(self, agent):
        self.agent = agent

    @abstractmethod
    def decide_movement(self) -> Optional[Tuple[int, int]]:
        """Retorna a próxima posição (x, y) ou None se ficar parado."""
        pass

    @abstractmethod
    def get_emission_multiplier(self) -> float:
        """Retorna o multiplicador de emissão viral atual."""
        pass

# ============================================================================
# PERFIS DE ESCOLA
# ============================================================================

class StudentFocusedBehavior(BehaviorStrategy):
    """Aluno Focado: Fica na carteira, sai pouco."""
    def decide_movement(self):
        if random.random() < 0.95: return None
        return self.agent._random_move_in_radius(1) # Ajeitar na cadeira

    def get_emission_multiplier(self):
        return 1.0

class StudentSocialBehavior(BehaviorStrategy):
    """Aluno Social: Circula nos intervalos."""
    def decide_movement(self):
        if self.agent.model.is_break_time():
            return self.agent._move_towards_density()
        return None

    def get_emission_multiplier(self):
        return 2.0 if self.agent.model.is_break_time() else 1.0

class TeacherBehavior(BehaviorStrategy):
    """Professor: Patrulha a frente da sala."""
    def decide_movement(self):
        if self.agent.pos[1] > 3: return (self.agent.pos[0], 1)
        new_x = self.agent.pos[0] + random.choice([-1, 1])
        return (new_x, self.agent.pos[1])

    def get_emission_multiplier(self):
        return 5.0

# ============================================================================
# PERFIS DE ACADEMIA
# ============================================================================

class AthleteTreadmillBehavior(BehaviorStrategy):
    """Esteira: Estático, respiração pesada."""
    def decide_movement(self):
        return None 

    def get_emission_multiplier(self):
        return 10.0

class AthleteSocialBehavior(BehaviorStrategy):
    """Atleta Social: Treina e conversa."""
    def decide_movement(self):
        return self.agent._move_towards_density()

    def get_emission_multiplier(self):
        return 3.0

class AthleteIsolateBehavior(BehaviorStrategy):
    """Atleta Focado: Foge de gente."""
    def decide_movement(self):
        return self.agent._move_away_from_density()

    def get_emission_multiplier(self):
        return 4.0

# ============================================================================
# PERFIS DE ESCRITÓRIO
# ============================================================================

class WorkerFocusedBehavior(BehaviorStrategy):
    """
    Profissional Focado:
    - Fica sentado trabalhando a maior parte do tempo.
    - Ocasionalmente levanta para ir buscar café/água (Cozinha).
    """
    def decide_movement(self):
        # 96% do tempo trabalhando quieto
        if random.random() < 0.96:
            return None
            
        # 4% de chance de "ir à cozinha" (Simulado por andar longe)
        # Se tivéssemos a coordenada da cozinha no grid, usaríamos ela.
        # Como não temos, simulamos uma caminhada longa (raio 10)
        return self.agent._random_move_in_radius(10)

    def get_emission_multiplier(self):
        return 1.0 # Silencioso

class WorkerSocialBehavior(BehaviorStrategy):
    """
    Profissional Social:
    - Trabalha, mas busca a proximidade dos colegas.
    - Conversa enquanto trabalha (emissão média).
    """
    def decide_movement(self):
        # 10% de chance de ir até a mesa de alguém para conversar
        if random.random() < 0.10:
            return self.agent._move_towards_density()
        return None # 90% parado (mas conversando)

    def get_emission_multiplier(self):
        return 2.0 # Conversando moderadamente

# ============================================================================
# FACTORY
# ============================================================================

class BehaviorFactory:
    @staticmethod
    def create(profile_type: str, agent) -> BehaviorStrategy:
        mapping = {
            # Escola
            "student_focused": StudentFocusedBehavior,
            "student_social": StudentSocialBehavior,
            "teacher": TeacherBehavior,
            
            # Academia
            "athlete_treadmill": AthleteTreadmillBehavior,
            "athlete_social": AthleteSocialBehavior,
            "athlete_isolate": AthleteIsolateBehavior,
            
            # Escritório (AGORA CORRETO)
            "worker_focused": WorkerFocusedBehavior,
            "worker_social": WorkerSocialBehavior
        }
        # Fallback seguro
        return mapping.get(profile_type, StudentFocusedBehavior)(agent)