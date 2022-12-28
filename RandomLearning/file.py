import inspect
import os
import my_agent

# 関数をファイルに書き出す
def write_agent_to_file(function, file):
    with open(file, "a" if os.path.exists(file) else "w") as f:
        f.write(inspect.getsource(function))

# エージェントファイルをファイルに書き出す
write_agent_to_file(my_agent.my_agent, "../submission.py")