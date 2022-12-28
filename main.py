from kaggle_environments import make, utils

env = make("connectx", debug=True)
env.render()
print(env.configuration)

trainer = env.train([None, "random"])
state = trainer.reset()
print(f"board: {state.board}\n"\
      f"mark: {state.mark}")

while not env.done:
    state,reward,done,info = trainer.step(0)
    print(f"reward: {reward}, done: {done}, info: {info}")
    board = state.board

env.render(mode="ipython", width=350, height=300)