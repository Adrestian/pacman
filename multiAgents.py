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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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
        print(f"action: {legalMoves[chosenIndex]}")

        print(f"{scores=}")
        print(f"{legalMoves=}")

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        newPacmanPos = successorGameState.getPacmanPosition()
        #print(f"{newPos=}") # Tuple(x,y), pacman's position
        newFood = successorGameState.getFood()
        foodList = newFood.asList()

        #print(f"{newFood=}") # game.Grid object
        newGhostStates = successorGameState.getGhostStates()
        #print(f"{newGhostStates=}") # game.AgentState object
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        #print(f"{newScaredTimes=}") # list of int

        "*** YOUR CODE HERE ***"
        ghostDistanceWt = 0.1
        foodDistanceWt = 1

        closestFoodDistance = float('inf')
        for i, fd in enumerate(foodList):
            FoodDistance = manhattanDistance(fd, newPacmanPos)
            if FoodDistance < closestFoodDistance:
                closestFoodDistance = FoodDistance
        # closer to food is good

        totalGhostDistance = 0
        scaredReward = 0
        for i, ghostState in enumerate(newGhostStates):
            ghostPosition = ghostState.getPosition()
            ghostDirection = ghostState.getDirection()
            if ghostState.scaredTimer > 0:
                # ghost is scared
                scaredReward = 200
                continue
            else:
                ghostDistance = manhattanDistance(newPacmanPos, ghostPosition)
                if ghostDistance > 3:
                    totalGhostDistance += 500
                    continue
                totalGhostDistance += ghostDistance
        # away from ghost is good
        
        return  successorGameState.getScore() + (foodDistanceWt/closestFoodDistance+1) + ghostDistanceWt * totalGhostDistance + scaredReward

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        "*** YOUR CODE HERE ***"

        evalFunction = betterEvaluationFunction # custom eval function
        # evalFunction = self.evaluationFunction # default eval function
        targetSearchDepth = self.depth
        numAgents = gameState.getNumAgents()
        
        def minState(gameState, agentIndex, currentSearchDepth):
            v = float('inf')
            
            # sanity check, should never happen, guard removed
            # if agentIndex == 0: raise RuntimeError("min state but agent index is 0")

            legalActions = gameState.getLegalActions(agentIndex)

            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                # for each successorState of the state
                # v = max(v, dispatchFunction(successorState))
                nextAgentIndex = (agentIndex + 1) % numAgents
                nextSearchDepth = currentSearchDepth+1 if nextAgentIndex == 0 else currentSearchDepth
                v = min(v, dispatchFunction(successorState, nextAgentIndex, numAgents, nextSearchDepth, targetSearchDepth))   
            return v
        
        def maxState(gameState, agentIndex, currentSearchDepth):
            v = float('-inf')
            
            # sanity check, guard removed 
            # if agentIndex != 0: raise RuntimeError("max state but not pacman")
            
            legalActions = gameState.getLegalActions(agentIndex)
            
            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                # for each successorState of the state
                # v = max(v, dispatchFunction(successorState))
                v = max(v, dispatchFunction(successorState, (agentIndex+1)%numAgents, numAgents, currentSearchDepth, targetSearchDepth))   

            return v

        def dispatchFunction(gameState, agentIndex, nAgents, currentSearchDepth, targetSearchDepth):
            if gameState.isWin() or gameState.isLose():
                return gameState.getScore()
            if currentSearchDepth == targetSearchDepth:
                return evalFunction(gameState)
            
            if agentIndex < 0 or agentIndex >= nAgents: # sanity check
                raise RuntimeError("In dispatchFunction: agentIndex out of bound")

            # if maxstate, return max(state)
            if agentIndex == 0: # pacman play max state
                return maxState(gameState, agentIndex, currentSearchDepth)

            # if min state, return min(state)
            if agentIndex != 0: # ghost play min state
                return minState(gameState, agentIndex, currentSearchDepth)
           
            
        currentAgentIndex = 0 # pacman move first
        legalMoves = gameState.getLegalActions(currentAgentIndex) # List[action]
        scores = [0]*len(legalMoves)

        # print(f"{legalMoves=}")
        for i, action in enumerate(legalMoves):
            nextAgentIndex = currentAgentIndex + 1
            successorGameState = gameState.generateSuccessor(currentAgentIndex, action)
            scores[i] = dispatchFunction(successorGameState, nextAgentIndex, numAgents, currentSearchDepth=0, targetSearchDepth=targetSearchDepth )

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)

        print(f"{bestScore=} for depth {self.depth} search")
        return legalMoves[chosenIndex]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        evalFunction = betterEvaluationFunction
        # evalFunction = self.evaluationFunction # default eval function

        targetSearchDepth = self.depth
        numAgents = gameState.getNumAgents()
        
        def minState(gameState, agentIndex, currentSearchDepth, alpha, beta):
            v = float('inf')
            
            # sanity check, should never happen, guard removed
            # if agentIndex == 0: raise RuntimeError("min state but agent index is 0")

            legalActions = gameState.getLegalActions(agentIndex)

            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                # for each successorState of the state
                # v = max(v, dispatchFunction(successorState))
                nextAgentIndex = (agentIndex + 1) % numAgents
                nextSearchDepth = currentSearchDepth+1 if nextAgentIndex == 0 else currentSearchDepth
                v = min(v, dispatchFunction(successorState, nextAgentIndex, numAgents, nextSearchDepth, targetSearchDepth, alpha, beta))   
                if v < alpha: return v
                beta = min(beta, v)
            return v
        
        def maxState(gameState, agentIndex, currentSearchDepth, alpha, beta):
            v = float('-inf')
            
            # sanity check, guard removed 
            # if agentIndex != 0: raise RuntimeError("max state but not pacman")
            
            legalActions = gameState.getLegalActions(agentIndex)
            
            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                # for each successorState of the state
                # v = max(v, dispatchFunction(successorState))
                v = max(v, dispatchFunction(successorState, (agentIndex+1)%numAgents, numAgents, currentSearchDepth, targetSearchDepth, alpha, beta))   
                if v > beta: return v
                alpha = max(alpha, v)
            return v

        def dispatchFunction(gameState, agentIndex, nAgents, currentSearchDepth, targetSearchDepth, alpha, beta):
            if gameState.isWin() or gameState.isLose():
                return gameState.getScore()
            if currentSearchDepth == targetSearchDepth:
                return evalFunction(gameState)
            
            if agentIndex < 0 or agentIndex >= nAgents: # sanity check
                raise RuntimeError("In dispatchFunction: agentIndex out of bound")

            # if maxstate, return max(state)
            if agentIndex == 0: # pacman play max state
                return maxState(gameState, agentIndex, currentSearchDepth, alpha, beta)

            # if min state, return min(state)
            if agentIndex != 0: # ghost play min state
                return minState(gameState, agentIndex, currentSearchDepth, alpha, beta)
           
            
        currentAgentIndex = 0 # pacman move first
        legalMoves = gameState.getLegalActions(currentAgentIndex) # List[action]
        scores = [0]*len(legalMoves)

        # print(f"{legalMoves=}")
        for i, action in enumerate(legalMoves):
            nextAgentIndex = currentAgentIndex + 1
            successorGameState = gameState.generateSuccessor(currentAgentIndex, action)
            scores[i] = dispatchFunction(successorGameState, nextAgentIndex, numAgents, currentSearchDepth=0, targetSearchDepth=targetSearchDepth, alpha=float('-inf') , beta=float('inf'))

        bestScore = max(scores)

        chosenIndex = len(scores)
        for i in range(len(scores)):
            if scores[i] == bestScore:
                chosenIndex = i
                break

        print(f"{bestScore=} for depth {self.depth} search")
        return legalMoves[chosenIndex]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """

        evalFunction = betterEvaluationFunction
        # evalFunction = self.evaluationFunction # default eval function

        targetSearchDepth = self.depth
        numAgents = gameState.getNumAgents()
        
        def expState(gameState, agentIndex, currentSearchDepth):
            v = 0 # init v = 0
            
            # sanity check, should never happen, guard removed
            # if agentIndex == 0: raise RuntimeError("min state but agent index is 0")

            legalActions = gameState.getLegalActions(agentIndex)

            prob = 1/len(legalActions) # assume uniform distribution
            
            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                # for each successorState of the state
                # v = max(v, dispatchFunction(successorState))
                nextAgentIndex = (agentIndex + 1) % numAgents
                nextSearchDepth = currentSearchDepth+1 if nextAgentIndex == 0 else currentSearchDepth
                # v = min(v, dispatchFunction(successorState, nextAgentIndex, numAgents, nextSearchDepth, targetSearchDepth))   
                v += prob * dispatchFunction(successorState, nextAgentIndex, numAgents, nextSearchDepth, targetSearchDepth)
            return v
        
        def maxState(gameState, agentIndex, currentSearchDepth):
            v = float('-inf')
            
            # sanity check, guard removed 
            # if agentIndex != 0: raise RuntimeError("max state but not pacman")
            
            legalActions = gameState.getLegalActions(agentIndex)
            
            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                # for each successorState of the state
                # v = max(v, dispatchFunction(successorState))
                v = max(v, dispatchFunction(successorState, (agentIndex+1)%numAgents, numAgents, currentSearchDepth, targetSearchDepth))   

            return v

        def dispatchFunction(gameState, agentIndex, nAgents, currentSearchDepth, targetSearchDepth):
            if gameState.isWin() or gameState.isLose():
                return gameState.getScore()
            if currentSearchDepth == targetSearchDepth:
                return evalFunction(gameState)
            
            if agentIndex < 0 or agentIndex >= nAgents: # sanity check
                raise RuntimeError("In dispatchFunction: agentIndex out of bound")

            # if maxstate, return max(state)
            if agentIndex == 0: # pacman play max state
                return maxState(gameState, agentIndex, currentSearchDepth)

            # if min state, return min(state)
            if agentIndex != 0: # ghost play min state
                return expState(gameState, agentIndex, currentSearchDepth)
           
            
        currentAgentIndex = 0 # pacman move first
        legalMoves = gameState.getLegalActions(currentAgentIndex) # List[action]
        scores = [0]*len(legalMoves)

        # print(f"{legalMoves=}")
        for i, action in enumerate(legalMoves):
            nextAgentIndex = currentAgentIndex + 1
            successorGameState = gameState.generateSuccessor(currentAgentIndex, action)
            scores[i] = dispatchFunction(successorGameState, nextAgentIndex, numAgents, currentSearchDepth=0, targetSearchDepth=targetSearchDepth )

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)

        print(f"{bestScore=} for depth {self.depth} search")
        return legalMoves[chosenIndex]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
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
        
    pacmanPos = currentGameState.getPacmanPosition()
    #print(f"{newPos=}") # Tuple(x,y), pacman's position
    food = currentGameState.getFood()
    foodList = food.asList()

    #print(f"{newFood=}") # game.Grid object
    ghostStates = currentGameState.getGhostStates()
    #print(f"{newGhostStates=}") # game.AgentState object
    newScaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    #print(f"{newScaredTimes=}") # list of int

    ghostDistanceWt = 0.1
    foodDistanceWt = 1

    closestFoodDistance = float('inf')
    for i, fd in enumerate(foodList):
        FoodDistance = manhattanDistance(fd, pacmanPos)
        if FoodDistance < closestFoodDistance:
            closestFoodDistance = FoodDistance
    # closer to food is good

    totalGhostDistance = 0
    scaredReward = 0
    for i, ghostState in enumerate(ghostStates):
        ghostPosition = ghostState.getPosition()
        ghostDirection = ghostState.getDirection()
        if ghostState.scaredTimer > 0:
            # ghost is scared
            scaredReward = 200
            continue
        else:
            ghostDistance = manhattanDistance(pacmanPos, ghostPosition)
            if ghostDistance > 3:
                totalGhostDistance += 500
                continue
            totalGhostDistance += ghostDistance
    # away from ghost is good
    
    return  currentGameState.getScore() + (foodDistanceWt/closestFoodDistance+1) + ghostDistanceWt * totalGhostDistance + scaredReward


# Abbreviation
better = betterEvaluationFunction
