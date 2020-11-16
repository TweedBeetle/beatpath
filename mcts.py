# https://github.com/pbsinclair42/MCTS
from __future__ import division

import time
import math
import random

from tqdm import tqdm


def randomPolicy(state):
    while not state.isTerminal():
        try:
            action = random.choice(state.getPossibleActions())
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.takeAction(action)
    return state.getReward()


class treeNode():
    def __init__(self, state, parent):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}

    def value(self):
        return self.totalReward / self.numVisits


class mcts():
    def __init__(self, timeLimit=None, iterationLimit=None, explorationConstant=1 / math.sqrt(2),
                 rolloutPolicy=randomPolicy):
        if timeLimit is not None:
            if iterationLimit is not None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = 'time'
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'
        self.explorationConstant = explorationConstant
        self.rollout = rolloutPolicy

        self.best_node = None
        self.best_reward = float("-inf")

        self.percentage_time_remaining = 1

    def search(self, initialState):
        self.root = treeNode(initialState, None)

        bar = tqdm()

        if self.limitType == 'time':
            end_time = time.time() + self.timeLimit

            while (time_remaining := end_time - time.time()) > 0:
                self.percentage_time_remaining = time_remaining / self.timeLimit

                self.executeRound()
                bar.update()
                bar.set_description(f'best reward: {self.best_reward}')
        else:
            for i in range(self.searchLimit):
                self.executeRound()
                bar.update()

    def best_action(self):
        bestChild = self.getBestChild(self.root, 0)
        return self.getAction(self.root, bestChild)

    def best_state(self):
        child = self.getBestChild(self.root, 0)

        nodes = [child]

        while len(child.children) != 0:
            # nodes.add(child)
            nodes.append(child)
            child = max(child.children.values(), key=lambda child: child.value())

        # nodes.add(child)
        nodes.append(child)

        # return max(nodes, key=lambda child: child.totalReward / child.numVisits).state
        return child.state

    def executeRound(self):
        node = self.selectNode(self.root)
        reward = self.rollout(node.state)
        self.backpropogate(node, reward)

    def selectNode(self, node):
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node)

        if node.value() > self.best_reward:
            self.best_node = node
            self.best_reward = node.value()

        return node

    def expand(self, node):
        actions = node.state.getPossibleActions()
        for action in actions:
            if action not in node.children.keys():
                newNode = treeNode(node.state.takeAction(action), node)
                node.children[action] = newNode
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                return newNode

        raise Exception("Should never reach here")

    def backpropogate(self, node, reward):
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def getBestChild(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            nodeValue = child.totalReward / child.numVisits + explorationValue * (math.sqrt(
                2 * math.log(node.numVisits) / child.numVisits))
            # nodeValue = child.totalReward / child.numVisits + explorationValue * (math.sqrt(
            #     2 * math.log(node.numVisits) / child.numVisits)) * min([10, (1 / (1 - self.percentage_time_remaining))])
            # nodeValue = child.totalReward / child.numVisits + min([1000, (1 / (1 - self.percentage_time_remaining))]) * (math.sqrt(
            #     2 * math.log(node.numVisits) / child.numVisits))
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)

    def getAction(self, root, bestChild):
        for action, node in root.children.items():
            if node is bestChild:
                return action
