import numpy as np

# Q値を格納しておくためのテーブル
class QTable():
    # コンストラクタ
    def __init__(self, actions):
        # 辞書型で格納
        # key:状態 value:全actionのQ値を配列で格納
        self.Q = {}
        self.actions = actions
    
    # 16進数で状態のkeyを生成
    def get_state_key(self, state):
        board = state.board[:]
        board.append(state.mark)
        state_key = np.array(board).astype(str)
        return hex(int(''.join(state_key),3))[2:]
    
    # 状態に対する、全actionのQ値の配列を出力
    def get_q_values(self, state):
        state_key = self.get_state_key(state)
        if state_key not in self.Q.keys():
            self.Q[state_key] = [0] * len(self.actions)
        return self.Q[state_key]
    
    # Q値の更新
    def update(self, state, action, add_q):
        state_key = self.get_state_key(state)
        self.Q[state_key] = [q + add_q if idx == action else q for idx, q in enumerate(self.Q[state_key])]