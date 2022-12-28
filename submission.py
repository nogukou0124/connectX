from random import choice

def my_agent(state, configuration):
    return choice([c for c in range(configuration.columns) if state.board[c] == 0])
