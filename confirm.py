import sys
from kaggle_environments import make,utils,agent

out = sys.stdout
submission = agent.read_file("./submission.py")
a = agent.get_last_callable(submission, path="./")
sys.stdout = out

env = make("connectx", debug=True)
env.run([a, a])
print("Success" if env.state[0].status == env.state[1].status == "DONE" else "Failed")
