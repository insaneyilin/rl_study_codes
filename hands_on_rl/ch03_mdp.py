import numpy as np

np.random.seed(0)

# ==================== 马尔可夫奖励过程 (MRP) ====================
# 定义状态转移矩阵 P：6个状态 (s0-s5) 之间的转移概率
P = np.array([
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],  # s0 -> s0:0.9, s0 -> s1:0.1
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],  # s1 -> s0:0.5, s1 -> s2:0.5
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],  # s2 -> s3:0.6, s2 -> s5:0.4
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],  # s3 -> s4:0.3, s3 -> s5:0.7
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],  # s4 -> s1:0.2, s4 -> s2:0.3, s4 -> s3:0.5
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # s5 -> s5:1.0 (终止状态)
])

rewards = np.array([-1, -2, -2, 10, 1, 0])  # 每个状态的即时奖励
gamma = 0.5  # 折扣因子


# -------------------- 回报计算 --------------------
def compute_return(start_index, chain, gamma):
    """计算从给定状态序列的某个起始点开始的总折扣回报"""
    G = 0
    for i in reversed(range(start_index, len(chain))):  # 逆序计算
        G = rewards[chain[i]] + gamma * G  # G = r_t + γ * G_{t+1}
    return G


# 示例：计算状态序列 [s0, s1, s2, s5] 的回报
chain = [0, 1, 2, 5]
print(f"状态序列: {chain}")
print(
    f"从 s{chain[0]} 开始的回报: {compute_return(0, chain, gamma):.2f}")  # 输出 -2.50
print(
    f"从 s{chain[1]} 开始的回报: {compute_return(1, chain, gamma):.2f}")  # 输出 -3.00


# -------------------- 状态价值计算（解析解）--------------------
def compute_value(P, rewards, gamma, states_num):
    """通过贝尔曼方程的矩阵形式直接计算状态价值"""
    rewards = rewards.reshape((-1, 1))  # 转为列向量
    value = np.linalg.inv(np.eye(states_num) - gamma * P) @ rewards
    return value.flatten()


V = compute_value(P, rewards, gamma, 6)
print("\nMRP各状态价值:\n", V)

# ==================== 马尔可夫决策过程 (MDP) ====================
# 定义MDP五元组 (S, A, P, R, γ)
S = ["s1", "s2", "s3", "s4", "s5"]  # 状态集合
A = ["保持s1", "前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往"]  # 动作集合

# 状态转移函数 P(s'|s,a)
P = {
    "s1-保持s1-s1": 1.0,  # 在s1选择"保持s1"，100%留在s1
    "s1-前往s2-s2": 1.0,  # 在s1选择"前往s2"，100%转移到s2
    "s2-前往s1-s1": 1.0,
    "s2-前往s3-s3": 1.0,
    "s3-前往s4-s4": 1.0,
    "s3-前往s5-s5": 1.0,
    "s4-前往s5-s5": 1.0,
    "s4-概率前往-s2": 0.2,  # 在s4选择"概率前往"，20%到s2
    "s4-概率前往-s3": 0.4,
    "s4-概率前往-s4": 0.4,
}

# 奖励函数 R(s,a)
R = {
    "s1-保持s1": -1,  # 在s1选择"保持s1"获得-1奖励
    "s1-前往s2": 0,
    "s2-前往s1": -1,
    "s2-前往s3": -2,
    "s3-前往s4": -2,
    "s3-前往s5": 0,
    "s4-前往s5": 10,
    "s4-概率前往": 1,
}

gamma = 0.5
MDP = (S, A, P, R, gamma)

# -------------------- 策略定义 --------------------
# 策略1：随机策略（各动作概率均匀分布）
Pi_1 = {
    "s1-保持s1": 0.5,
    "s1-前往s2": 0.5,
    "s2-前往s1": 0.5,
    "s2-前往s3": 0.5,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.5,
    "s4-概率前往": 0.5,
}

# 策略2：偏向特定动作的策略
Pi_2 = {
    "s1-保持s1": 0.6,
    "s1-前往s2": 0.4,
    "s2-前往s1": 0.3,
    "s2-前往s3": 0.7,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.1,
    "s4-概率前往": 0.9,
}


# -------------------- 蒙特卡洛采样 --------------------
def MC_sample(MDP, Pi, timestep_max, num_episodes):
    """用给定策略Pi采样多条轨迹"""
    S, A, P, R, gamma = MDP
    episodes = []

    for _ in range(num_episodes):
        episode = []
        s = np.random.choice(S[:-1])  # 随机选择非终止状态作为起点
        timestep = 0

        while s != "s5" and timestep < timestep_max:
            timestep += 1
            # 根据策略选择动作
            rand = np.random.rand()
            cumulative_prob = 0
            for a in A:
                cumulative_prob += Pi.get(f"{s}-{a}", 0)
                if rand <= cumulative_prob:
                    break

            # 根据转移概率得到下一个状态
            rand = np.random.rand()
            cumulative_prob = 0
            for s_next in S:
                key = f"{s}-{a}-{s_next}"
                cumulative_prob += P.get(key, 0)
                if rand <= cumulative_prob:
                    break

            r = R.get(f"{s}-{a}", 0)
            episode.append((s, a, r, s_next))
            s = s_next

        episodes.append(episode)
    return episodes


# 采样示例
print("\n=== 蒙特卡洛采样示例 ===")
episodes = MC_sample(MDP, Pi_1, 20, 3)
for i, ep in enumerate(episodes, 1):
    print(f"轨迹{i}: [ s(t), a(t), r(t), s(t+1) ] \n {ep[:5]}...")  # 只显示前5步


# -------------------- 占用度量计算 --------------------
def occupancy(episodes, target_s, target_a, timestep_max, gamma):
    """计算状态-动作对的占用度量（策略评估）"""
    total_visits = np.zeros(timestep_max)
    target_visits = np.zeros(timestep_max)

    for episode in episodes:
        for t, (s, a, _, _) in enumerate(episode):
            if t >= timestep_max:
                break
            total_visits[t] += 1
            if s == target_s and a == target_a:
                target_visits[t] += 1

    rho = 0
    for t in reversed(range(timestep_max)):
        if total_visits[t] > 0:
            rho += (gamma**t) * (target_visits[t] / total_visits[t])

    return (1 - gamma) * rho


# 比较两种策略在(s4, "概率前往")的占用度量
rho_1 = occupancy(MC_sample(MDP, Pi_1, 1000, 1000), "s4", "概率前往", 1000, gamma)
rho_2 = occupancy(MC_sample(MDP, Pi_2, 1000, 1000), "s4", "概率前往", 1000, gamma)
print(f"\n策略1在(s4, '概率前往')的占用度量: {rho_1:.4f}")
print(f"策略2在(s4, '概率前往')的占用度量: {rho_2:.4f}")

rho_1 = occupancy(MC_sample(MDP, Pi_1, 1000, 1000), "s4", "前往s5", 1000, gamma)
rho_2 = occupancy(MC_sample(MDP, Pi_2, 1000, 1000), "s4", "前往s5", 1000, gamma)
print(f"\n策略1在(s4, '前往s5')的占用度量: {rho_1:.4f}")
print(f"策略2在(s4, '前往s5')的占用度量: {rho_2:.4f}")
