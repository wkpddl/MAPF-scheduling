import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import MessagePassing

class BipartiteGNNLayer(MessagePassing):
    def __init__(self, robot_dim, task_dim, edge_dim, hidden_dim):
        super(BipartiteGNNLayer, self).__init__(aggr='add')
        self.f_theta_e = torch.nn.Sequential(
            torch.nn.Linear(robot_dim + task_dim + edge_dim, hidden_dim),
            torch.nn.LeakyReLU(negative_slope=0.1),  # 使用LeakyReLU
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.f_theta_v = torch.nn.Sequential(
            torch.nn.Linear(robot_dim + hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(negative_slope=0.1),  # 使用LeakyReLU
            torch.nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x_robot, x_task, edge_index, edge_attr):
        x = torch.cat([x_robot, x_task], dim=0)
        aggregated_messages = self.get_aggregated_messages(x_robot, x_task, edge_index, edge_attr)
        updated_robots = self.update_robots(x_robot, aggregated_messages)
        return updated_robots, x_task

    def get_aggregated_messages(self, x_robot, x_task, edge_index, edge_attr):
        x = torch.cat([x_robot, x_task], dim=0)
        robot_indices = edge_index[0]
        task_indices = edge_index[1] + x_robot.size(0)
        adjusted_edge_index = torch.stack([robot_indices, task_indices], dim=0)
        return self.propagate(adjusted_edge_index, x=x, edge_attr=edge_attr, size=(x.size(0), x.size(0)))

    def update_robots(self, x_robot, aggregated_messages):
        robot_messages = aggregated_messages[:x_robot.size(0)]
        update_input = torch.cat([robot_messages, x_robot], dim=1)
        return self.f_theta_v(update_input)

    def message(self, x_i, x_j, edge_attr):
        msg = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.f_theta_e(msg)

class TaskSelectionMLP(torch.nn.Module):
    def __init__(self, robot_hidden_dim, task_dim, num_tasks, mlp_hidden=128):
        super(TaskSelectionMLP, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(robot_hidden_dim + task_dim, mlp_hidden),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(mlp_hidden, 1)  # 输出每个机器人-任务对的一个分数
        )

    def forward(self, robot_features, task_features):
        N, H = robot_features.shape  # N = 机器人数量
        M, D = task_features.shape   # M = 任务数量
        
        # 扩展维度并拼接特征
        robot_expanded = robot_features.unsqueeze(1).expand(-1, M, -1)  # [N, M, H]
        task_expanded = task_features.unsqueeze(0).expand(N, -1, -1)    # [N, M, D]
        combined = torch.cat([robot_expanded, task_expanded], dim=2)   # [N, M, H+D]
        
        # 通过MLP计算每个机器人-任务对的分数
        scores = self.mlp(combined.reshape(-1, H + D))  # [N*M, 1]
        
        # 重塑为 [N, M] 并应用softmax
        scores = scores.reshape(N, M)  # [N, M]
        task_probs = F.softmax(scores, dim=1)  # 对每个机器人的任务分数进行归一化
        
        return task_probs
    
class RobotTaskPolicy(torch.nn.Module):
    def __init__(self, robot_dim, task_dim, edge_dim, gnn_hidden, mlp_hidden, num_tasks):
        """
        机器人任务分配策略模型
        
        参数:
        - robot_dim: 机器人特征维度
        - task_dim: 任务特征维度
        - edge_dim: 边特征维度
        - gnn_hidden: GNN隐藏层维度
        - mlp_hidden: MLP隐藏层维度
        - num_tasks: 任务数量
        """
        super(RobotTaskPolicy, self).__init__()
        self.gnn = BipartiteGNNLayer(
            robot_dim=robot_dim,
            task_dim=task_dim,
            edge_dim=edge_dim,
            hidden_dim=gnn_hidden
        )
        self.task_selector = TaskSelectionMLP(
            robot_hidden_dim=gnn_hidden,
            task_dim=task_dim,
            num_tasks=num_tasks,
            mlp_hidden=mlp_hidden
        )

    def forward(self, x_robot, x_task, edge_index, edge_attr):
        """
        前向传播
        
        参数:
        - x_robot: 机器人特征 [num_robots, robot_dim]
        - x_task: 任务特征 [num_tasks, task_dim]
        - edge_index: 边索引 [2, num_edges]
        - edge_attr: 边特征 [num_edges, edge_dim]
        
        返回:
        - task_probs: 任务选择概率 [num_robots, num_tasks]
        """
        robot_hidden, _ = self.gnn(x_robot, x_task, edge_index, edge_attr)
        return self.task_selector(robot_hidden, x_task)
# 使用示例
def test_robot_task_selection():
    num_robots = 4  # 4个机器人
    num_tasks = 6  # 6个任务
    robot_dim = 12  # 机器人特征维度
    task_dim = 12   # 任务特征维度
    edge_dim = 8    # 边特征维度（如距离、优先级等）
    gnn_hidden = 64 # GNN隐藏层维度
    mlp_hidden = 32 # MLP隐藏层维度

    # 随机初始化输入数据
    x_robot = torch.randn(num_robots, robot_dim)   # 机器人特征
    x_task = torch.randn(num_tasks, task_dim)     # 任务特征
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 2, 3, 3],  # 机器人索引（0-3）
        [0, 1, 2, 3, 4, 5, 0, 5]   # 任务索引（0-5）
    ], dtype=torch.long)
    edge_attr = torch.randn(edge_index.size(1), edge_dim)  # 边特征

    # 初始化模型
    model = RobotTaskPolicy(
        robot_dim=robot_dim,
        task_dim=task_dim,
        edge_dim=edge_dim,
        gnn_hidden=gnn_hidden,
        mlp_hidden=mlp_hidden,
        num_tasks=num_tasks
    )

    # 前向传播获取任务选择概率
    task_probs = model(x_robot, x_task, edge_index, edge_attr)
    print("机器人-任务选择概率矩阵：")
    print(task_probs.shape)  # 输出形状: [num_robots, num_tasks]
    print(task_probs)
    print_assignments(task_probs, x_robot, x_task)


def assign_tasks_based_on_probability(task_probs, robots, tasks):
    """
    基于概率矩阵进行一对一任务分配
    
    参数:
    - task_probs: 概率矩阵 [num_robots, num_tasks]
    - robots: 机器人位置列表
    - tasks: 任务位置列表
    
    返回:
    - assignments: 分配结果列表 [(robot_idx, task_idx, robot_pos, task_pos, prob), ...]
    """
    # 转为numpy数组方便处理
    probs = task_probs.detach().numpy()
    num_robots, num_tasks = probs.shape
    
    # 初始化未分配的机器人和任务集合
    unassigned_robots = set(range(num_robots))
    unassigned_tasks = set(range(min(num_robots, num_tasks)))  # 确保任务数不超过机器人数
    
    assignments = []
    
    # 一对一分配过程
    while unassigned_robots and unassigned_tasks:
        # 构建当前未分配的子概率矩阵
        current_robot_indices = list(unassigned_robots)
        current_task_indices = list(unassigned_tasks)
        
        # 从未分配的项中构建子概率矩阵
        sub_probs = np.zeros((len(current_robot_indices), len(current_task_indices)))
        for i, r_idx in enumerate(current_robot_indices):
            for j, t_idx in enumerate(current_task_indices):
                sub_probs[i, j] = probs[r_idx, t_idx]
        
        # 归一化概率
        total_prob = sub_probs.sum()
        if total_prob > 0:
            sub_probs = sub_probs / total_prob
        else:
            # 处理所有概率为0的特殊情况
            sub_probs = np.ones_like(sub_probs) / sub_probs.size
        
        # 将二维概率矩阵展平为一维
        flat_probs = sub_probs.flatten()
        
        # 使用np.random.choice按概率权重选择一个索引
        if len(flat_probs) > 0:
            flat_index = np.random.choice(len(flat_probs), p=flat_probs)
            # 转回二维索引
            row, col = divmod(flat_index, len(current_task_indices))
            
            # 获取实际的机器人和任务索引
            robot_idx = current_robot_indices[row]
            task_idx = current_task_indices[col]
            
            # 记录分配结果
            assignments.append((
                robot_idx, 
                task_idx, 
                robots[robot_idx], 
                tasks[task_idx], 
                probs[robot_idx, task_idx]
            ))
            
            # 从未分配集合中移除已分配的项
            unassigned_robots.remove(robot_idx)
            unassigned_tasks.remove(task_idx)
    
    return assignments

# 使用代码示例
def print_assignments(task_probs, robots, tasks):
    assignments = assign_tasks_based_on_probability(task_probs, robots, tasks)
    
    print("\n基于概率的任务分配结果:")
    for robot_idx, task_idx, robot_pos, task_pos, prob in assignments:
        print(f"机器人 {robot_idx} 位置:{robot_pos} -> 任务 {task_idx} 位置:{task_pos} (概率: {prob:.4f})")
    
    # 转换为(y,x),(y,x)格式的起点-终点对
    start_goal_pairs = []
    for _, _, robot_pos, task_pos, _ in assignments:
        # 确保坐标为(y,x)格式
        if isinstance(robot_pos, torch.Tensor):
            # 如果是张量，取前两个元素作为y,x坐标
            start = (float(robot_pos[0]), float(robot_pos[1]))
        elif hasattr(robot_pos, '__iter__') and len(robot_pos) >= 2:
            # 如果是可迭代对象(列表、元组等)，取前两个元素
            start = (robot_pos[0], robot_pos[1])
        else:
            raise ValueError(f"无法从{robot_pos}提取(y,x)坐标")
            
        if isinstance(task_pos, torch.Tensor):
            goal = (float(task_pos[0]), float(task_pos[1]))
        elif hasattr(task_pos, '__iter__') and len(task_pos) >= 2:
            goal = (task_pos[0], task_pos[1])
        else:
            raise ValueError(f"无法从{task_pos}提取(y,x)坐标")
            
        start_goal_pairs.append((start, goal))
    
    print("\nEECBS输入格式 (y,x)坐标对:")
    for i, (start, goal) in enumerate(start_goal_pairs):
        print(f"机器人 {i}: ({start[0]:.1f}, {start[1]:.1f}) -> ({goal[0]:.1f}, {goal[1]:.1f})")
    
    return assignments, start_goal_pairs
if __name__ == "__main__":
    test_robot_task_selection()
