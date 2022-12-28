import sys
from kaggle_environments import make,utils,agent

out = sys.stdout
submission = utils.read_file("./submission.py")
a = agent.get_last_callable(submission)
sys.stdout = out

env = make("connectx", debug=True)
env.run([agent, agent])
print("Success" if env.state[0].status == env.state[1].status == "DONE" else "Failed")
