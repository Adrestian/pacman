# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

DEBUG = True

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()
        self.tmpValues = util.Counter() # A Counter is a dict with default 0, both values and q-values are cached from here
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        ####################################
        ## Bootstrapping and init         ##
        ####################################
        for state in states:
            self.tmpValues[(state, 0)] = 0 # V_0 = 0
            print(f"{state=}, possible actions={self.mdp.getPossibleActions(state)}")
        

        for iteration in range(1, self.iterations+1):
            for state in states:
                actions = self.mdp.getPossibleActions(state)
                
                if self.mdp.isTerminal(state) or len(actions) == 0: # no actions available, possibly a terminal state
                    self.tmpValues[(state, iteration)] = 0
                    continue
                
                qValues = [0] * len(actions)
                for i, action in enumerate(actions):
                    thisActionReward = 0
                    transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
                    
                    for nextStateProb in transitionStatesAndProbs:
                        nextState, prob = nextStateProb
                        instReward = self.mdp.getReward(state, action, nextState)
                        discountedFutureReward = self.discount * self.tmpValues[(nextState, iteration-1)]
                        thisActionReward += prob * (instReward + discountedFutureReward)
                    

                    qValues[i] = thisActionReward
                maxQValue = max(qValues)
                self.tmpValues[(state, iteration)] = maxQValue


        # copy the final answer from tmpValues to values
        i = self.iterations
        for state in states:
            self.values[state] = self.tmpValues[(state, i)]


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        if self.mdp.isTerminal(state):
            print("terminal state in computeQValueFromValues")
            return 0

        actionReward = 0
        transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        for nextStateProb in transitionStatesAndProbs:
            nextState, prob = nextStateProb
            instReward = self.mdp.getReward(state, action, nextState)
            discountedFutureRewards = self.discount * self.getValue(nextState)
            actionReward += prob * (instReward + discountedFutureRewards)
        return actionReward


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        if self.mdp.isTerminal(state) or len(actions) == 0:
            return None
        qValues = [0] * len(actions)
        for i, action in enumerate(actions):
            qval = self.computeQValueFromValues(state, action)
            qValues[i] = qval

        maxQval = max(qValues)

        bestActions = []
        for i, action in enumerate(actions):
            if qValues[i] == maxQval:
                bestActions.append(action)
        
        # if DEBUG:
        #     print(f"{bestActions=}")
        from random import choice
        return choice(bestActions)


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        self.version = dict()
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        nStates = len(states)

        for state in states: # bootstrapping and init
            self.version[state] = 0
            self.tmpValues[(state, 0)] = 0
        
        for it in range(self.iterations):
            state = states[it%nStates] # get the states that this iteration will works on
            actions = self.mdp.getPossibleActions(state)
            
            currentVersion = self.version[state] + 1
            if self.mdp.isTerminal(state) or len(actions) == 0:
                self.tmpValues[(state, currentVersion)] = self.tmpValues[(state, currentVersion-1)]
                continue # should be zero 

            qValues = [0] * len(actions)
            for i, action in enumerate(actions):
                thisActionReward = 0
                transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)

                for nextStateProb in transitionStatesAndProbs:
                    nextState, prob = nextStateProb
                    instReward = self.mdp.getReward(state, action, nextState)
                    discountedFutureReward = self.discount * self.tmpValues[(nextState, currentVersion-1)]
                    thisActionReward += prob * (instReward + discountedFutureReward)
                qValues[i] = thisActionReward
            maxQValue = max(qValues)
            self.tmpValues[(state, currentVersion)] = maxQValue
            self.version[state] = currentVersion # update the version tracker as well
            
        # copy to final destination
        for state in states:
            ver = self.version[state]
            self.values[state] = self.tmpValues[(state, ver)]

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        from collections import defaultdict
        self.pred = defaultdict(set) # hold mapping from state to its predecessors states
        # Pred States: (all states that have non-zero probability of reaching s by taking some action a) 
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Compute the predecessors of all states
        self.computePredStates()
        states = self.mdp.getStates()
        for state in states: # init
            self.values[state] = 0

        pq = util.PriorityQueue()

        # for each non-terminal state s, do:
        for state in states:
            if self.mdp.isTerminal(state): continue # ignore terminal states
            # Find the absolute value of the difference between the current value of s 
            # in self.values and the highest Q-value across all possible actions from s 
            # (this represents what the value should be); 
            # call this number diff. Do NOT update self.values[s] in this step.
            currentValue = self.values[state]
            # find the highest Q-Value
            maxQvalue = self.getMaxQValue(state)
            diff = abs(currentValue-maxQvalue) 
            # Push s into the priority queue with priority -diff (note that this is negative). 
            # We use a negative because the priority queue is a min heap, 
            # but we want to prioritize updating states that have a higher error
            pq.push(state, -diff)
        
        for _ in range(self.iterations):
            if pq.isEmpty(): # If the priority queue is empty, then terminate.
                return
            
            state = pq.pop()
            if not self.mdp.isTerminal(state):  # if it is not a terminal state, update value
                newValue = self.updateState(state)
                self.values[state] = newValue
            # for each predecessor p of s do:
            predecessors = self.getPredecessorStates(state)
            for p in predecessors:
                currentPValue = self.values[p]
                pMaxQValue = self.getMaxQValue(p)
                pDiff = abs(currentPValue-pMaxQValue) 
                if pDiff > self.theta:
                    pq.update(p, -pDiff)

    # Apply one step look ahead(single iteration)
    def updateState(self, state):
        actions = self.mdp.getPossibleActions(state)
        qValues = [0] * len(actions)
        for i, action in enumerate(actions):
            thisActionReward = 0
            transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
            for nextStateProb in transitionStatesAndProbs:
                nextState, prob = nextStateProb
                instReward = self.mdp.getReward(state, action, nextState)
                discountedFutureReward = self.discount * self.values[nextState]
                thisActionReward += prob * (instReward + discountedFutureReward)
            qValues[i] = thisActionReward
        return max(qValues)

    def computePredStates(self):
        states = self.mdp.getStates()
        for state in states:
            for predstate in states:
                actions = self.mdp.getPossibleActions(predstate)
                for action in actions:
                    nextStateProbPairs = self.mdp.getTransitionStatesAndProbs(predstate, action)
                    for nextStateProbPair in nextStateProbPairs:
                        nextState, prob = nextStateProbPair
                        if prob == 0: # ignore
                            continue
                        if nextState == state:
                            self.pred[state].add(predstate)
    
    def getPredecessorStates(self, state):
        states = self.mdp.getStates()
        ret = set()
        for predstate in states:
            actions = self.mdp.getPossibleActions(predstate)
            for action in actions:
                nextStateProbPairs = self.mdp.getTransitionStatesAndProbs(predstate, action)
                for nextStateProbPair in nextStateProbPairs:
                    nextState, prob = nextStateProbPair
                    if prob == 0:
                        continue
                    if nextState == state:
                        ret.add(predstate)
        return ret
    
    def getMaxQValue(self, state):
        actions = self.mdp.getPossibleActions(state)
        qValues = [0] * len(actions)
        for i, action in enumerate(actions):
            qValues[i] = self.getQValue(state, action)
        maxQValue = max(qValues)
        return maxQValue


