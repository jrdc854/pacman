# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import torch
import numpy as np
from net import PacmanNet
import os
from util import manhattanDistance
from game import Directions
import random, util
random.seed(42)  # For reproducibility
from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        score = successorGameState.getScore()

        #H1: Distancia a la comida más cercana 
        if newFood: 
            min_food_distance = min(manhattanDistance(newPos, f) for f in newFood)
            score += 1.0 / (min_food_distance + 1)

        #H2: Proximidad a fantasmas (peligrosos y asustados)
        for ghost_state in newGhostStates:
            ghost_pos = ghost_state.getPosition()
            ghost_distance = manhattanDistance(newPos, ghost_pos)

            if ghost_state.scaredTimer > 0:
                score += 50 / (ghost_distance + 1)
            else:
                if ghost_distance <= 2:
                    score -= 200
        
        #H3: Atracción hacia cápsulas de poder
        capsules = successorGameState.getCapsules()
        if capsules:
            min_capsule_distance = min(manhattanDistance(newPos, c) for c in capsules)
            score += 10.0 / (min_capsule_distance + 1)
            score -= 20.0 * len(capsules)

        #H4: Evitar accorralamiento por varios fantasmas activos
        active_ghost_near = sum(
            1 for g in newGhostStates
            if g.scaredTimer == 0 and manhattanDistance(newPos, g.getPosition()) <= 3
        ) 
        if active_ghost_near > 1:
            score -= 600
        

        #H5: Penalización por volumen de comida restante
        score -= 4.0 * len(newFood)

        #H6: Tener opciones de huida
        if len(successorGameState.getLegalActions(0)) >= 3:
            score += 15
        
        return score



def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        def value(agentIndex, depth, state):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            if agentIndex == 0:
                return maxValue(agentIndex, depth, state)
            return minValue(agentIndex, depth, state)
        
        def maxValue(agentIndex, depth, state):
            v = float('-inf')
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, value(1, depth, successor))
            return v
        
        def minValue(agentIndex, depth, state):
            v = float('inf')
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)
            nextAgent = agentIndex + 1
            nextDepth = depth
            if nextAgent == state.getNumAgents():
                nextAgent = 0
                nextDepth += 1
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                v = min(v, value(nextAgent, nextDepth, successor))
            return v
        
        bestAction = None
        bestScore = float('-inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = value(1, 0, successor)
            if score > bestScore or bestAction is None:
                bestScore = score
                bestAction = action
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def value(agentIndex, depth, state, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            if agentIndex == 0:
                return maxValue(agentIndex, depth, state, alpha, beta)
            return minValue(agentIndex, depth, state, alpha, beta)

        def maxValue(agentIndex, depth, state, alpha, beta):
            v = float('-inf')
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, value(1, depth, successor, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v
    
        def minValue(agentIndex, depth, state, alpha, beta):
                v = float('inf')
                legalActions = state.getLegalActions(agentIndex)
                if not legalActions:
                    return self.evaluationFunction(state)

                nextAgent = agentIndex + 1
                nextDepth = depth
                if nextAgent == state.getNumAgents():
                    nextAgent = 0
                    nextDepth += 1

                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    v = min(v, value(nextAgent, nextDepth, successor, alpha, beta))
                    if v < alpha:
                        return v
                    beta = min(beta, v)
                return v

        bestAction = None
        bestScore = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = value(1, 0, successor, alpha, beta)
            if score > bestScore or bestAction is None:
                bestScore = score
                bestAction = action
            alpha = max(alpha, bestScore)
        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def value(agentIndex, depth, state):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            if agentIndex == 0:
                return maxValue(agentIndex, depth, state)
            return expectValue(agentIndex, depth, state)

        def maxValue(agentIndex, depth, state):
            v = float('-inf')
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, value(1, depth, successor))
            return v
        
        def expectValue(agentIndex, depth, state):
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)

            nextAgent = agentIndex + 1
            nextDepth = depth
            if nextAgent == state.getNumAgents():
                nextAgent = 0
                nextDepth += 1

            total = 0.0
            probability = 1.0 / len(legalActions)
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                total += probability * value(nextAgent, nextDepth, successor)
            return total

        bestAction = None
        bestScore = float('-inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = value(1, 0, successor)
            if score > bestScore or bestAction is None:
                bestScore = score
                bestAction = action
        return bestAction


def betterEvaluationFunction(currentGameState: GameState):
    """
    Función de evaluación mejorada para Expectimax.

    Combina las mismas heurísticas del NeuralAgent:
    - Distancia a la comida más cercana
    - Proximidad a fantasmas (peligrosos y asustados)
    - Atracción hacia cápsulas de poder
    - Evitar acorralamiento por varios fantasmas
    - Penalización por volumen de comida restante
    - Tener opciones de huida
    """
    score = currentGameState.getScore()
    pacman_pos = currentGameState.getPacmanPosition()
    food_list = currentGameState.getFood().asList()
    ghost_states = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    # H1: Distancia a la comida más cercana
    if food_list:
        min_food_distance = min(manhattanDistance(pacman_pos, f) for f in food_list)
        score += 1.0 / (min_food_distance + 1)

    # H2: Proximidad a fantasmas (peligrosos y asustados)
    for ghost_state in ghost_states:
        ghost_pos = ghost_state.getPosition()
        ghost_distance = manhattanDistance(pacman_pos, ghost_pos)

        if ghost_state.scaredTimer > 0:
            score += 50 / (ghost_distance + 1)
        else:
            if ghost_distance <= 2:
                score -= 200

    # H3: Atracción hacia cápsulas de poder
    if capsules:
        min_capsule_distance = min(manhattanDistance(pacman_pos, c) for c in capsules)
        score += 10.0 / (min_capsule_distance + 1)
        score -= 20.0 * len(capsules)

    # H4: Evitar acorralamiento por varios fantasmas activos
    active_ghosts_near = sum(
        1 for g in ghost_states
        if g.scaredTimer == 0 and manhattanDistance(pacman_pos, g.getPosition()) <= 3
    )
    if active_ghosts_near > 1:
        score -= 600

    # H5: Penalización por volumen de comida restante
    score -= 4.0 * len(food_list)

    # H6: Tener opciones de huida
    if len(currentGameState.getLegalActions(0)) >= 3:
        score += 15

    return score

# Abbreviation
better = betterEvaluationFunction


###########################################################################
# Ahmed
###########################################################################

class NeuralAgent(Agent):
    """
    Un agente de Pacman que utiliza una red neuronal para tomar decisiones
    basado en la evaluación del estado del juego.
    """
    def __init__(self, model_path="models/pacman_model.pth", use_heuristics=True):
        super().__init__()
        self.model = None
        self.input_size = None
        # -a pasa strings: "False" como string es truthy en Python, hay que convertirlo
        if isinstance(use_heuristics, str):
            self.use_heuristics = use_heuristics.lower() not in ('false', '0', 'no')
        else:
            self.use_heuristics = bool(use_heuristics)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model(model_path)
        
        # Mapeo de índices a acciones
        self.idx_to_action = {
            0: Directions.STOP,
            1: Directions.NORTH,
            2: Directions.SOUTH,
            3: Directions.EAST,
            4: Directions.WEST
        }
        
        # Para evaluar alternativas
        self.action_to_idx = {v: k for k, v in self.idx_to_action.items()}
        
        # Contador de movimientos
        self.move_count = 0
        
        mode = "con heurísticas" if self.use_heuristics else "GREEDY PURO (solo red neuronal)"
        print(f"NeuralAgent inicializado [{mode}], usando dispositivo: {self.device}")

    def load_model(self, model_path):
        """Carga el modelo desde el archivo guardado"""
        try:
            if not os.path.exists(model_path):
                print(f"ERROR: No se encontró el modelo en {model_path}")
                return False
                
            # Cargar el modelo
            checkpoint = torch.load(model_path, map_location=self.device)
            self.input_size = checkpoint['input_size']
            
            # Crear y cargar el modelo
            self.model = PacmanNet(self.input_size, 128, 5).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()  # Modo evaluación
            
            print(f"Modelo cargado correctamente desde {model_path}")
            print(f"Tamaño de entrada: {self.input_size}")
            return True
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            return False

    def state_to_matrix(self, state):
        """Convierte el estado del juego en una matriz numérica normalizada"""
        # Obtener dimensiones del tablero
        walls = state.getWalls()
        width, height = walls.width, walls.height
        
        # Crear una matriz numérica
        # 0: pared, 1: espacio vacío, 2: comida, 3: cápsula, 4: fantasma, 5: Pacman
        numeric_map = np.zeros((width, height), dtype=np.float32)
        
        # Establecer espacios vacíos (todo lo que no es pared comienza como espacio vacío)
        for x in range(width):
            for y in range(height):
                if not walls[x][y]:
                    numeric_map[x][y] = 1
        
        # Agregar comida
        food = state.getFood()
        for x in range(width):
            for y in range(height):
                if food[x][y]:
                    numeric_map[x][y] = 2
        
        # Agregar cápsulas
        for x, y in state.getCapsules():
            numeric_map[x][y] = 3
        
        # Agregar fantasmas
        for ghost_state in state.getGhostStates():
            ghost_x, ghost_y = int(ghost_state.getPosition()[0]), int(ghost_state.getPosition()[1])
            # Si el fantasma está asustado, marcarlo diferente
            if ghost_state.scaredTimer > 0:
                numeric_map[ghost_x][ghost_y] = 6  # Fantasma asustado
            else:
                numeric_map[ghost_x][ghost_y] = 4  # Fantasma normal
        
        # Agregar Pacman
        pacman_x, pacman_y = state.getPacmanPosition()
        numeric_map[int(pacman_x)][int(pacman_y)] = 5
        
        # Normalizar
        numeric_map = numeric_map / 6.0
        
        return numeric_map

    def evaluationFunction(self, state):
        """
        Una función de evaluación basada en la red neuronal y en heurísticas adicionales.
        """
        if self.model is None:
            return 0  # Si no hay modelo, devolver 0
        
        # Convertir a matriz
        state_matrix = self.state_to_matrix(state)
        
        # Convertir a tensor
        state_tensor = torch.FloatTensor(state_matrix).unsqueeze(0).to(self.device)
        
        # Obtener predicciones
        with torch.no_grad():
            output = self.model(state_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
        
        # Obtener acciones legales
        legal_actions = state.getLegalActions()
        
        # Aplicar heurísticas adicionales, similar a betterEvaluationFunction
        score = state.getScore()
        
        # Mejorar la evaluación con conocimiento del dominio
        pacman_pos = state.getPacmanPosition()
        food = state.getFood().asList()
        ghost_states = state.getGhostStates()
        
        # Factor 1: Distancia a la comida más cercana
        if food:
            min_food_distance = min(manhattanDistance(pacman_pos, food_pos) for food_pos in food)
            score += 1.0 / (min_food_distance + 1)
        
        # Factor 2: Proximidad a fantasmas
        for ghost_state in ghost_states:
            ghost_pos = ghost_state.getPosition()
            ghost_distance = manhattanDistance(pacman_pos, ghost_pos)
            
            if ghost_state.scaredTimer > 0:
                # Si el fantasma está asustado, acercarse a él
                score += 50 / (ghost_distance + 1)
            else:
                # Si no está asustado, evitarlo
                if ghost_distance <= 2:
                    score -= 200  # Gran penalización por estar demasiado cerca
        
        # ==================================================
        # nuevas heurísticas añadidias (task 1)
        # ==================================================

        # heurística nueva 1: atracción hacia las cápsulas de poder
        # incentiva al pacman a buscar cápsulas si están cerca para activar el modo caza
        capsules = state.getCapsules()
        if capsules:
            min_capsule_distance = min(manhattanDistance(pacman_pos, cap_pos) for cap_pos in capsules)
            score += 10.0 / (min_capsule_distance + 1)
            score -= 20.0 * len(capsules) #penalizar por dejar  capsulas sin comer

        #heurística nueva 2: evitar caminos sin salida
        #alerta para que pacman no se meta voluntariamente en caminos sin salida
        for ghost_state in ghost_states:
            if ghost_state.scaredTimer == 0:
                ghost_pos = ghost_state.getPosition()
                ghost_distance = manhattanDistance(pacman_pos, ghost_pos)
                if ghost_distance == 3:
                    score -= 50
        
        #heuristica nueva 3: incentivo por muerte de un fantasma
        # si un fantasma asustado está a distancia de 1 paso, asegura que Pacman se lo coma
        for ghost_state in ghost_states:
            if ghost_state.scaredTimer > 0:
                ghost_pos = ghost_state.getPosition()
                ghost_distance = manhattanDistance(pacman_pos, ghost_pos)
                if ghost_distance == 1:
                    score += 500


        #heurística nueva 4: evitar acorralamiento
        #si hay más de un fantasma activo persiguiendo a pacman a corta distancia hay peligro
        active_ghosts_near = 0
        for ghost_state in ghost_states:
            if ghost_state.scaredTimer == 0:
                if manhattanDistance(pacman_pos, ghost_state.getPosition()) <= 3:
                    active_ghosts_near += 1
        if active_ghosts_near > 1:
            score -= 600

        #heurística nueva 5: gestión del tiempo del fantasma asustado
        for ghost_state in ghost_states:
            if 0 < ghost_state.scaredTimer <= 2:
                ghost_distance = manhattanDistance(pacman_pos, ghost_state.getPosition())
                if ghost_distance > ghost_state.scaredTimer:
                    score -= 100
        

        #heurística nueva 6: penalización por volumen de comida 
        #fuera al pacman a preferir estados donde el número total de comida sea menor
        if food:
            score -= 4.0 * len(food)


        #heurística nueva 7: tener opciones de huida
        # incentiva al pacman a mantener tres o más direcciones posibles para evitar ser arrinconado
        if len(legal_actions) >= 3:
            score += 15



        # Combinar la puntuación de la red con la heurística
        neural_score = 0
        for i, action in enumerate(self.idx_to_action.values()):
            if action in legal_actions:
                neural_score += probabilities[i] * 100
        
        return score + neural_score

    def getAction(self, state):
        """
        Devuelve la mejor acción basada en la evaluación de la red neuronal
        y heurísticas adicionales (si use_heuristics=True).
        Si use_heuristics=False, usa solo la predicción de la red neuronal (greedy puro).
        """
        self.move_count += 1
        
        # Si no hay modelo, hacer un movimiento aleatorio
        if self.model is None:
            print("ERROR: Modelo no cargado. Haciendo movimiento aleatorio.")
            exit()
            legal_actions = state.getLegalActions()
            return random.choice(legal_actions)
        
        # Obtener acciones legales
        legal_actions = state.getLegalActions()
        
        # Evaluación directa con la red neuronal
        state_matrix = self.state_to_matrix(state)
        state_tensor = torch.FloatTensor(state_matrix).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(state_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
        
        # Mapear índices del modelo a acciones del juego
        action_probs = []
        for idx, prob in enumerate(probabilities):
            action = self.idx_to_action[idx]
            if action in legal_actions:
                action_probs.append((action, prob))
        
        # Ordenar por probabilidad (mayor a menor)
        action_probs.sort(key=lambda x: x[1], reverse=True)
        
        # ===== MODO GREEDY PURO (sin heurísticas) =====
        if not self.use_heuristics:
            # Simplemente devolver la acción con mayor probabilidad de la red
            # Excluir STOP si hay otras opciones
            for action, prob in action_probs:
                if action != Directions.STOP or len(action_probs) == 1:
                    return action
            return action_probs[0][0]
        
        # ===== MODO CON HEURÍSTICAS (comportamiento original) =====
        # Exploración: con una probabilidad decreciente, elegir aleatoriamente
        exploration_rate = 0.2 * (0.99 ** self.move_count)  # Disminuye con el tiempo
        if random.random() < exploration_rate:
            # Excluir STOP si es posible
            if len(legal_actions) > 1 and Directions.STOP in legal_actions:
                legal_actions.remove(Directions.STOP)
            return random.choice(legal_actions)
        
        # Evaluación alternativa: generar sucesores y evaluar cada uno
        successors = []
        for action in legal_actions:
            successor = state.generateSuccessor(0, action)
            eval_score = self.evaluationFunction(successor)
            neural_score = 0
            for a, p in action_probs:
                if a == action:
                    neural_score = p * 100
                    break
            # Combinar evaluación heurística con la predicción de la red
            combined_score = eval_score + neural_score
            
            # Penalizar STOP a menos que sea la única opción
            if action == Directions.STOP and len(legal_actions) > 1:
                combined_score -= 50
                
            successors.append((action, combined_score))
        
        # Ordenar por puntuación combinada
        successors.sort(key=lambda x: x[1], reverse=True)
        
        # Devolver la mejor acción
        return successors[0][0]

# Definir una función para crear el agente
def createNeuralAgent(model_path="models/pacman_model.pth"):
    """
    Función de fábrica para crear un agente neuronal.
    Útil para integrarse con la estructura de pacman.py.
    """
    return NeuralAgent(model_path)


# NUEVO AGENTE AlphaBetaNeuralAgent:
"""
Hasta ahora, el agente que teníamos, NeuralAgent solo mira 1 paso hacia el futuro,
calcula la puntuación de las casillas adyacentes y se mueve a la mejor.
El nuevo agente, antes de mover dibuja un árbol de posibilidades evaluando varios turnos hacia el futuro.
Simula su movimiento y cómo responderían los fantasmas, luego su siguiente movimiento y así sucesivamente hasta la profundidad max
y ahí llama a la función de puntuación (heurísticas + red neuronal) para decidir.

Aquí entra la Poda Alpha-Beta
Pacman no puede calcular todas las posibilidades porque el árbol de decisiones crece exponencialmente,
para optimizar el cálculo se utiliza la poda alpha-beta.

- Alpha: La puntuación mínima que Pacman ya tiene "asegurada" en el mejor de los casos.
- Beta: La puntuación máxima que los Fantasmas están dispuestos a permitir a Pacman en el peor de los casos.
"""

class AlphaBetaNeuralAgent(MultiAgentSearchAgent):
    """
    Agente híbrido: Poda Alfa-Beta combinada con Red Neuronal y Heurísticas
    """
    def __init__(self, evalFn='scoreEvaluationFunction', depth='2', w_heuristic=1.0, w_neural=1.0, model_path="models/pacman_model.pth"):
        super().__init__(evalFn, depth)
        self.w_heuristic = float(w_heuristic)
        self.w_neural = float(w_neural)
        
        self.neural_helper = NeuralAgent(model_path)
        
        # Intentamos cargar el modelo (similar a como lo hace NeuralAgent)

        self.model = self.neural_helper.model
        if self.model is not None: 
            print(f"AlphaBetaNeuralAgent: Enlazado al modelo de NeuralAgent con éxito.")
        else:
            print(f"AlphaBetaNeuralAgent: ADVERTENCIA - No se pudo enlazar el modelo (es None).")


    def getAction(self, gameState: GameState):
        """
        Función principal que devuelve la mejor acción usando el árbol Alfa-Beta
        """
        # Función recursiva general
        def alphabeta(agentIndex, depth, state, alpha, beta):
            # Casos base: si ganamos, perdemos o llegamos al límite de profundidad mental
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluation_combined(state)

            if agentIndex == 0:  # Turno de Pacman (Maximizar)
                return maxValue(agentIndex, depth, state, alpha, beta)
            else:  # Turno de los Fantasmas (Minimizar)
                return minValue(agentIndex, depth, state, alpha, beta)

        # Turno de Pacman: Busca la puntuación más ALTA
        def maxValue(agentIndex, depth, state, alpha, beta):
            v = float('-inf')
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions: return self.evaluation_combined(state)
                
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                # Siguiente turno: primer fantasma (agentIndex 1), misma profundidad
                v = max(v, alphabeta(1, depth, successor, alpha, beta))
                if v > beta: 
                    return v # Poda! (El fantasma nunca permitirá llegar a este caso)
                alpha = max(alpha, v)
            return v

        # Turno de Fantasmas: Buscan la puntuación más BAJA para fastidiar a Pacman
        def minValue(agentIndex, depth, state, alpha, beta):
            v = float('inf')
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions: return self.evaluation_combined(state)

            nextAgent = agentIndex + 1
            nextDepth = depth
            # Si ya movieron todos los fantasmas, volvemos a Pacman (agent 0) y bajamos un nivel de profundidad
            if nextAgent == state.getNumAgents():
                nextAgent = 0
                nextDepth += 1

            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                v = min(v, alphabeta(nextAgent, nextDepth, successor, alpha, beta))
                if v < alpha: 
                    return v # Poda! (Pacman nunca elegirá la rama que llevó a este caso)
                beta = min(beta, v)
            return v

        # AQUÍ EMPIEZA LA LÓGICA DEL PRIMER MOVIMIENTO DE PACMAN
        bestAction = None
        bestScore = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        legalActions = gameState.getLegalActions(0)
        # Por defecto, quitamos STOP si hay más opciones para que no se quede quieto
        if len(legalActions) > 1 and Directions.STOP in legalActions:
            legalActions.remove(Directions.STOP)

        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            # Iniciamos la recursión con el fantasma 1
            score = alphabeta(1, 0, successor, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, bestScore)

        return bestAction

    def evaluation_combined(self, state): # mejoramos esta funcion 
        """
        Calcula la puntuación del estado final ajustando los pesos DINÁMICAMENTE
        según la situación del juego (Punto Opcional).
        """
        trad_score = self.traditional_evaluation(state)
        neural_score = self.neural_evaluation(state)
        
        # Pesos por defecto
        dynamic_w_heuristic = self.w_heuristic
        dynamic_w_neural = self.w_neural
        
        # Comprobamos si hay algún fantasma asustado en este estado
        ghost_states = state.getGhostStates()
        hay_fantasmas_asustados = any(ghost.scaredTimer > 0 for ghost in ghost_states)
        
        if hay_fantasmas_asustados:
            # Las heurísticas saben exactamente cómo cazar (Heurística 3). 
            # Le damos 90% de importancia a la heurística y 10% a la red.
            dynamic_w_heuristic = 0.9
            dynamic_w_neural = 0.1
        else:
            # fantasmas peligrosos
            # La red neuronal entrenada es mejor esquivando a largo plazo.
            # Le damos 70% de importancia a la intuición de la red y 30% a las reglas básicas.
            dynamic_w_heuristic = 0.3
            dynamic_w_neural = 0.7
            
        return (dynamic_w_heuristic * trad_score) + (dynamic_w_neural * neural_score)


    def traditional_evaluation(self, state):
        """
        Reutilizamos la lógica y heurísticas de tu compañero
        """
        score = state.getScore()
        pacman_pos = state.getPacmanPosition()
        food = state.getFood().asList()
        ghost_states = state.getGhostStates()
        capsules = state.getCapsules()

        # 1. Distancia a comida
        if food:
            min_food_distance = min(manhattanDistance(pacman_pos, food_pos) for food_pos in food)
            score += 1.0 / (min_food_distance + 1)
            score -= 4.0 * len(food) # Heurística 6 (Volumen de comida)

        # 2. Heurística 1 (Atracción a cápsulas)
        if capsules:
            min_cap_dist = min(manhattanDistance(pacman_pos, cap_pos) for cap_pos in capsules)
            score += 10.0 / (min_cap_dist + 1)
            score -= 20.0 * len(capsules)

        # 3. Análisis de Fantasmas
        active_ghosts_near = 0
        for ghost in ghost_states:
            ghost_pos = ghost.getPosition()
            ghost_distance = manhattanDistance(pacman_pos, ghost_pos)
            
            if ghost.scaredTimer > 0:
                # Si está asustado y cerca, ¡a por él! (Heurística 3)
                if ghost_distance == 1: score += 500
                else: score += 50 / (ghost_distance + 1)
                
                # Heurística 5 (Gestión de tiempo asustado)
                if ghost.scaredTimer <= 2 and ghost_distance > ghost.scaredTimer:
                    score -= 100
            else:
                # Si es peligroso
                if ghost_distance <= 2: score -= 200
                if ghost_distance == 3: score -= 50 # Heurística 2 (Evitar callejones)
                if ghost_distance <= 3: active_ghosts_near += 1

        # Heurística 4 (Evitar acorralamiento)
        if active_ghosts_near > 1: score -= 600
        
        # Heurística 7 (Opciones de huida)
        if len(state.getLegalActions(0)) >= 3: score += 15

        return score

    def neural_evaluation(self, state):
        """
        Usa la red neuronal para darle nota a este estado. 
        Calcula la probabilidad de que la red decidiera moverse a él.
        """
        if self.model is None: return 0

        state_matrix = self.neural_helper.state_to_matrix(state)
        state_tensor = torch.FloatTensor(state_matrix).unsqueeze(0).to(self.neural_helper.device)
        
        with torch.no_grad():
            output = self.model(state_tensor)
            # Usamos el máximo valor de predicción como la puntuación "intuitiva" de la red para este tablero
            score = torch.max(output).item()
            
        return score
