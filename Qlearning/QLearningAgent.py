from kaggle_environments import make
import QTable as QT
import numpy as np
from random import choice
from tqdm.notebook import tqdm

env = make("connectx", debug=True)
trainer = env.train([None, "random"])

class QLearningAgent():
    def __init__(self, env, epsilon=0.99):
        self.env = env
        self.actions = list(range(self.env.configuration.columns))
        self.q_table = QT.QTable(self.actions)
        self.epsilon = epsilon
        self.reward_log = []
    
    # QTableをもとに、ある状態におけるQ値が最大になるactionを選択する
    def policy(self, state):
        if np.random.random() < self.epsilon:
            # epsilonの割合で、ランダムにactionを選択する
            return choice([c for c in range(len(self.actions)) if state.board[c] == 0])
        else:
            # ゲーム上選択可能で、Q値が最大なactionを選択する
            q_values = self.q_table.get_q_values(state)
            selected_items = [q if state.board[idx] == 0 else -1e7 for idx,q in enumerate(q_values)]
            return int(np.argmax(selected_items))
    
    # QTableの作成がうまくいくように、報酬関数をカスタマイズ
    def custom_reward(self, reward, done):
        if done:
            if reward == 1: # 勝ち
                return 20
            elif reward == 0: # 負け
                return -20
            else: # 引き分け
                return 10
        else:
            return -0.05 # 勝敗がついていない
    
    # エピソード毎にQTableを更新して学習する
    def learn(self, trainer, episode_cnt= 10000, gamma=0.6,
              learn_rate=0.3, epsilon_decay_rate=0.9999, min_epsilon=0.1):
        for episode in tqdm(range(episode_cnt)):
            # ゲーム環境をリセット
            state = trainer.reset()
            # epsilonを徐々に小さくする
            self.epsilon = max(min_epsilon, self.epsilon * epsilon_decay_rate)
            while not env.done:
                # どの列にドロップするかを決めて実行する
                action = self.policy(state)
                next_state, reward, done, info = trainer.step(action)
                reward = self.custom_reward(reward, done)
                # 誤差を計算してQTableを更新する
                gain = reward + gamma * max(self.q_table.get_q_values(next_state))
                estimate = self.q_table.get_q_values(next_state)[action]
                self.q_table.update(state, action, learn_rate * (gain - estimate))
                state = next_state
            
            self.reward_log.append(reward)