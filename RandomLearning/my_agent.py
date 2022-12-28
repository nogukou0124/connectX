from random import choice
from kaggle_environments import evaluate

def my_agent(state, configuration):
    return choice([c for c in range(configuration.columns) if state.board[c] == 0])

print(evaluate("connectx", [my_agent, "random"], num_episodes=3))


