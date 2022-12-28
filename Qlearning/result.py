from kaggle_environments import make
import QLearningAgent
import pandas as pd
import matplotlib.pyplot as plt

env = make("connectx", debug=True)
trainer = env.train([None, "random"])

# 学習
qa = QLearningAgent.QLearningAgent(env)
qa.learn(trainer)

# ゲーム終了時に得られた報酬の移動平均
import seaborn as sns
sns.set(style='darkgrid')
pd.DataFrame({'Average Reward': qa.reward_log}).rolling(500).mean().plot(figsize=(10,5))
plt.show()