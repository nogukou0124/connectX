from kaggle_environments import make
import QLearningAgent as QA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

env = make("connectx", debug=True)
trainer = env.train([None, "random"])

# 学習
qa = QA.QLearningAgent(env)
qa.learn(trainer)

# ゲーム終了時に得られた報酬の移動平均
import seaborn as sns
sns.set(style='darkgrid')
pd.DataFrame({'Average Reward': qa.reward_log}).rolling(500).mean().plot(figsize=(10,5))
plt.show()

tmp_dict_q_table = qa.q_table.Q.copy()
dict_q_table = dict()

# 学習したQテーブルで、一番Q値の大きいActionに置き換える
for k in tmp_dict_q_table:
    if np.count_nonzero(tmp_dict_q_table[k]) > 0:
        dict_q_table[k] = int(np.argmax(tmp_dict_q_table[k]))

my_agent = '''def my_agent(observation, configuration):
    from random import choice
    # 作成したテーブルを文字列に変換して、Pythonファイル上でdictとして扱えるようにする
    q_table = ''' \
    + str(dict_q_table).replace(' ', '') \
    + '''
    board = observation.board[:]
    board.append(observation.mark)
    state_key = list(map(str, board))
    state_key = hex(int(''.join(state_key), 3))[2:]
    # Qテーブルに存在しない状態の場合
    if state_key not in q_table.keys():
        return choice([c for c in range(configuration.columns) if observation.board[c] == 0])
    # Qテーブルから最大のQ値をとるActionを選択
    action = q_table[state_key]
    # 選んだActionが、ゲーム上選べない場合
    if observation.board[action] != 0:
        return choice([c for c in range(configuration.columns) if observation.board[c] == 0])
    return action
    '''

with open('../submission.py', 'w') as f:
    f.write(my_agent)