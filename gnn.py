import torch
import subprocess
import heapq
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from math import sqrt
from scipy.optimize import linear_sum_assignment

def read_paths_from_file(paths_file):
    """
    Reads paths from the paths.txt file.

    Args:
        paths_file (str): Path to the paths.txt file.

    Returns:
        list[list[tuple[int, int]]]: List of paths for each agent.
    """
    paths = []
    with open(paths_file, 'r') as file:
        for line in file:
            if line.startswith("Agent"):  # Only process lines starting with "Agent"
                path = []
                # Extract the path part after the colon
                path_data = line.split(":")[1].strip()
                # Split the path into individual points
                points = path_data.split("->")
                for point in points:
                    point = point.strip("()").strip()
                    if point:  # Skip empty points
                        y, x = map(int, point.split(","))
                        path.append((y, x))
                paths.append(path)
    return paths

def calculate_total_cost(paths):
    """
    Calculates the total cost of all paths.

    Args:
        paths (list[list[tuple[int, int]]]): List of paths for each agent.

    Returns:
        float: The total cost of all paths.
    """
    total_cost = 0
    for path in paths:
        cost = 0
        for i in range(1, len(path)):
            prev = path[i - 1]
            curr = path[i]
            # Calculate the cost based on Manhattan distance or Euclidean distance
            if abs(prev[0] - curr[0]) + abs(prev[1] - curr[1]) == 2:  # Diagonal move
                cost += np.sqrt(2)
            else:  # Horizontal or vertical move
                cost += 1
        total_cost += cost
    return total_cost

def astar_shortest_path_four(matrix, start, goal):
    """
    Finds the shortest path between start and goal using A* algorithm.

    Args:
        matrix (list[list[int]]): The 2D matrix representing the map.
        start (tuple[int, int]): The starting point (row, col).
        goal (tuple[int, int]): The goal point (row, col).

    Returns:
        tuple[list[tuple[int, int]], float]: The shortest path as a list of coordinates and its total cost,
                                             or ([], -1) if no path exists.
    """
    rows, cols = len(matrix), len(matrix[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1) # Up, Down, Left, Right
                  ]  
    open_set = []
    heapq.heappush(open_set, (0, start))  # (priority, current_position)
    g_cost = {start: 0}
    f_cost = {start: heuristic(start, goal)}
    came_from = {}
    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)

        # If we reach the goal, reconstruct and return the path and cost
        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, g_cost[current]

        visited.add(current)

        for dr, dc in directions:
            new_row, new_col = current[0] + dr, current[1] + dc
            neighbor = (new_row, new_col)

            # Check if the neighbor is within bounds
            if not (0 <= new_row < rows and 0 <= new_col < cols):
                continue

            # Check if the neighbor is passable
            if matrix[new_row][new_col] != 1:
                continue

            # Prevent cutting corners (diagonal moves)
            if abs(dr) + abs(dc) == 2:  # Diagonal move
                if matrix[current[0] + dr][current[1]] == 0 or matrix[current[0]][current[1] + dc] == 0:
                    continue

            # Calculate the tentative g_cost
            move_cost = sqrt(2) if abs(dr) + abs(dc) == 2 else 1
            tentative_g_cost = g_cost[current] + move_cost

            if neighbor not in g_cost or tentative_g_cost < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g_cost
                f_cost[neighbor] = tentative_g_cost + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_cost[neighbor], neighbor))
                came_from[neighbor] = current

    # If no path exists, return an empty list and -1
    return [], -1

def read_matrix_from_file(file_path):
    """
    Reads a 2D matrix from a .txt file.

    Args:
        file_path (str): Path to the .txt file.

    Returns:
        list[list[int]]: 2D matrix representation of the data.
    """
    matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            # Convert each line into a list of integers
            matrix.append([int(num) for num in line.strip().split()])
    return matrix

def find_free_points(matrix):
    """
    Finds all free points (value 1) in the matrix.

    Args:
        matrix (list[list[int]]): The 2D matrix.

    Returns:
        list[tuple[int, int]]: List of coordinates of free points.
    """
    free_points = []
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if value == 1:  # Free point
                free_points.append((i, j))
    return free_points

def assign_start_and_goal_points(free_points, num_pairs):
    """
    Randomly assigns start and goal points from the list of free points.

    Args:
        free_points (list[tuple[int, int]]): List of free point coordinates.
        num_pairs (int): Number of start-goal pairs to assign.

    Returns:
        list[tuple[tuple[int, int], tuple[int, int]]]: List of start-goal pairs.
    """
    if len(free_points) < num_pairs * 2:
        raise ValueError("Not enough free points to assign start and goal pairs.")

    # Randomly shuffle the free points
    random.shuffle(free_points)

    # Assign start and goal points
    pairs = []
    for i in range(num_pairs):
        start = free_points.pop()
        goal = free_points.pop()
        pairs.append((start, goal))
    return pairs

def astar_shortest_path(matrix, start, goal):
    """
    Finds the shortest path between start and goal using A* algorithm.

    Args:
        matrix (list[list[int]]): The 2D matrix representing the map.
        start (tuple[int, int]): The starting point (row, col).
        goal (tuple[int, int]): The goal point (row, col).

    Returns:
        tuple[list[tuple[int, int]], float]: The shortest path as a list of coordinates and its total cost,
                                             or ([], -1) if no path exists.
    """
    rows, cols = len(matrix), len(matrix[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Up, Down, Left, Right
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonal moves
    open_set = []
    heapq.heappush(open_set, (0, start))  # (priority, current_position)
    g_cost = {start: 0}
    f_cost = {start: heuristic(start, goal)}
    came_from = {}
    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)

        # If we reach the goal, reconstruct and return the path and cost
        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, g_cost[current]

        visited.add(current)

        for dr, dc in directions:
            new_row, new_col = current[0] + dr, current[1] + dc
            neighbor = (new_row, new_col)

            # Check if the neighbor is within bounds
            if not (0 <= new_row < rows and 0 <= new_col < cols):
                continue

            # Check if the neighbor is passable
            if matrix[new_row][new_col] != 1:
                continue

            # Prevent cutting corners (diagonal moves)
            if abs(dr) + abs(dc) == 2:  # Diagonal move
                if matrix[current[0] + dr][current[1]] == 0 or matrix[current[0]][current[1] + dc] == 0:
                    continue

            # Calculate the tentative g_cost
            move_cost = sqrt(2) if abs(dr) + abs(dc) == 2 else 1
            tentative_g_cost = g_cost[current] + move_cost

            if neighbor not in g_cost or tentative_g_cost < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g_cost
                f_cost[neighbor] = tentative_g_cost + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_cost[neighbor], neighbor))
                came_from[neighbor] = current

    # If no path exists, return an empty list and -1
    return [], -1

def heuristic(point, goal):
    """
    Heuristic function for A* (Euclidean distance).

    Args:
        point (tuple[int, int]): Current point (row, col).
        goal (tuple[int, int]): Goal point (row, col).

    Returns:
        float: Estimated cost from point to goal.
    """
    return sqrt((point[0] - goal[0]) ** 2 + (point[1] - goal[1]) ** 2)

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def plot_map_with_paths(matrix, start_goal_pairs, paths):
    plt.figure(figsize=(12, 12))
    matrix = np.array(matrix)
    plt.imshow(matrix, cmap="Greys", origin="upper")

    for i, ((start, goal), path) in enumerate(zip(start_goal_pairs, paths)):
        path = np.array(path)
        plt.plot(path[:, 1], path[:, 0], label=f"Path {i + 1}")
        plt.scatter(start[1], start[0], color="blue", label="Start" if i == 0 else "")
        plt.scatter(goal[1], goal[0], color="red", label="Goal" if i == 0 else "")

    plt.legend()
    plt.title("Map with Paths")
    plt.show()

def save_pairs_to_scen_file(pairs, costs, map_name, output_file):
    """
    Save start-goal pairs to a scenario file in MAPF benchmark format.

    Args:
        pairs (list[tuple[tuple[int, int], tuple[int, int]]]): List of start-goal pairs.
        costs (list[float]): List of path costs for each pair.
        map_name (str): Name of the map file.
        output_file (str): Path to the output scenario file.
    """
    version = 1  # Default version for the scenario file
    with open(output_file, 'w') as file:
        file.write(f"version {version}\n")  # Write version header
        for i, ((start, goal), cost) in enumerate(zip(pairs, costs)):
            start_col, start_row = start[1], start[0]
            goal_col, goal_row = goal[1], goal[0]
            file.write(f"{i}\t{map_name}\t{128}\t{128}\t{start_col}\t{start_row}\t{goal_col}\t{goal_row} \t{cost:.6f}\n")

def run_eecbs(map_file, scen_file, output_csv, output_paths, num_agents, time_limit, suboptimality,verbose):
    """
    Run the eecbs executable with the given parameters.

    Args:
        map_file (str): Path to the map file.
        scen_file (str): Path to the scenario file.
        output_csv (str): Path to the output CSV file.
        output_paths (str): Path to the output paths file.
        num_agents (int): Number of agents.
        time_limit (int): Time limit in seconds.
        suboptimality (float): Suboptimality factor.

    Returns:
        str: The standard output from the eecbs command.
    """
    # Construct the command
    command = [
        "./eecbs",
        "-m", map_file,
        "-a", scen_file,
        "-o", output_csv,
        "--outputPaths", output_paths,
        "-k", str(num_agents),
        "-t", str(time_limit),
        "--suboptimality", str(suboptimality)
    ]

    try:
        # Run the command and capture the output
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        if verbose:
            print("EECBS executed successfully.")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Error occurred while running EECBS:")
        print(e.stderr)
        return None


class BipartiteGNNLayer(MessagePassing):
    def __init__(self, robot_dim, task_dim, edge_dim, hidden_dim, leaky_relu_neg_slope=0.1):
        """
        二分图GNN层（使用LeakyReLU激活函数）
        
        参数:
        - robot_dim: 机器人节点特征维度
        - task_dim: 任务节点特征维度
        - edge_dim: 边特征维度
        - hidden_dim: 隐藏层维度
        - leaky_relu_neg_slope: LeakyReLU的负斜率参数
        """
        super(BipartiteGNNLayer, self).__init__(aggr='add')
        
        # 定义消息函数 f_θe（使用LeakyReLU）
        self.f_theta_e = torch.nn.Sequential(
            torch.nn.Linear(robot_dim + task_dim + edge_dim, hidden_dim),
            torch.nn.LeakyReLU(negative_slope=leaky_relu_neg_slope),  # 替换为LeakyReLU
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 定义节点更新函数 f_θv（使用LeakyReLU）
        self.f_theta_v = torch.nn.Sequential(
            torch.nn.Linear(robot_dim + hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(negative_slope=leaky_relu_neg_slope),  # 替换为LeakyReLU
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=leaky_relu_neg_slope)  # 可复用的激活函数
        
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
        return self.f_theta_v(update_input)  # 自动应用LeakyReLU
    
    def message(self, x_i, x_j, edge_attr):
        msg = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.f_theta_e(msg)  # 自动应用LeakyReLU
    
    # 保留update方法（尽管未使用）
    def update(self, aggr_out, x):
        return aggr_out

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

def build_bipartite_graph(robots, tasks, matrix):
    """
    构建二分图，将机器人和任务转化为图的两类顶点。

    Args:
        robots (list[tuple[int, int]]): 机器人坐标列表 [(y1, x1), (y2, x2), ...]。
        tasks (list[tuple[int, int]]): 任务目标坐标列表 [(y1, x1), (y2, x2), ...]。
        matrix (list[list[int]]): 地图矩阵。

    Returns:
        Data: PyTorch Geometric 的图数据对象。
    """
    num_robots = len(robots)
    num_tasks = len(tasks)

    # 顶点特征：机器人和任务的坐标
    robot_features = torch.tensor(robots, dtype=torch.float)  # 机器人节点特征
    task_features = torch.tensor(tasks, dtype=torch.float)    # 任务节点特征

    # 边的起点和终点
    edge_index = []
    edge_features = []

    for i, robot in enumerate(robots):
        for j, task in enumerate(tasks):
            # 计算边特征
            manhattan_distance = abs(robot[0] - task[0]) + abs(robot[1] - task[1])
            path, actual_distance = astar_shortest_path_four(matrix, robot, task)
            if actual_distance == -1:  # 如果路径不可达，跳过
                continue
            priority = 1  # 假设任务优先级为 1，可根据实际情况调整
            difficulty = (actual_distance - manhattan_distance) / priority

            # 添加边
            edge_index.append([i, j])  # 从机器人到任务
            edge_features.append([manhattan_distance,actual_distance,difficulty])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # 转置为 [2, num_edges]
    edge_features = torch.tensor(edge_features, dtype=torch.float)

    # 合并节点特征
    x = torch.cat([robot_features, task_features], dim=0)

    # 构建 PyTorch Geometric 的图数据对象
    
    return robot_features, task_features, edge_index, edge_features

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

def print_assignments(task_probs, robots, tasks, verbose=False):
    assignments = assign_tasks_based_on_probability(task_probs, robots, tasks)
    
    if verbose:
        print("\n基于概率的任务分配结果:")
        for robot_idx, task_idx, robot_pos, task_pos, prob in assignments:
            print(f"机器人 {robot_idx} 位置:{robot_pos} -> 任务 {task_idx} 位置:{task_pos} (概率: {prob:.4f})")
    
    # 转换为(y,x),(y,x)格式的起点-终点对
    start_goal_pairs = []
    for _, _, robot_pos, task_pos, _ in assignments:
        # 确保坐标为(y,x)格式的整数
        if isinstance(robot_pos, torch.Tensor):
            start = (int(robot_pos[0]), int(robot_pos[1]))
        elif hasattr(robot_pos, '__iter__') and len(robot_pos) >= 2:
            start = (int(robot_pos[0]), int(robot_pos[1]))
        else:
            raise ValueError(f"无法从{robot_pos}提取(y,x)坐标")
            
        if isinstance(task_pos, torch.Tensor):
            goal = (int(task_pos[0]), int(task_pos[1]))
        elif hasattr(task_pos, '__iter__') and len(task_pos) >= 2:
            goal = (int(task_pos[0]), int(task_pos[1]))
        else:
            raise ValueError(f"无法从{task_pos}提取(y,x)坐标")
            
        start_goal_pairs.append((start, goal))
    
    if verbose:
        print("\nEECBS输入格式 (y,x)坐标对:")
        for i, (start, goal) in enumerate(start_goal_pairs):
            print(f"机器人 {i}: ({start[0]}, {start[1]}) -> ({goal[0]}, {goal[1]})")
    
    return assignments, start_goal_pairs

def calculate_total_cost_from_assignment(start_goal_pairs, matrix, verbose=False):
    paths=[]
    costs = []
    # 计算每对的路径和成本
    for i, (start, goal) in enumerate(start_goal_pairs):
        path, cost = astar_shortest_path(matrix, start, goal)
        paths.append(path)
        costs.append(cost)
    
    if verbose:
        # 打印详细信息
        for i, ((start, goal), cost) in enumerate(zip(start_goal_pairs, costs)):
            print(f"Pair {i + 1}: Start {start}, Goal {goal}, Cost: {cost:.2f}")
    
    # 保存并运行EECBS
    save_pairs_to_scen_file(start_goal_pairs, costs, "map.map", "/home/uio/EECBS/task.scen")
    output = run_eecbs("map.map", "task.scen", "test.csv", "paths.txt", 20, 60, 1.2,verbose=False)
    
    if verbose and output:
        print("EECBS Output:")
        print(output)

    # 读取并计算总成本
    agent_paths = read_paths_from_file("/home/uio/EECBS/paths.txt")
    total_cost = calculate_total_cost(agent_paths)
    return total_cost

def train_with_policy_gradient(num_episodes=1000, learning_rate=0.001, save_interval=100, 
                               batch_size=5, entropy_coef=0.01, checkpoint_path=None,seed=42):
    """
    使用策略梯度训练任务分配模型
    
    参数:
    - num_episodes: 训练回合数
    - learning_rate: 学习率
    - save_interval: 保存模型的间隔
    - batch_size: 批处理大小，每隔多少个样本更新一次
    - entropy_coef: 熵正则化系数
    - checkpoint_path: 检查点路径，用于继续训练
    
    返回:
    - model: 训练好的模型
    - history: 训练历史记录
    """
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 初始化模型和优化器
    robot_dim=2
    task_dim=2
    edge_dim=3
    hidden_dim=32
    gnn_hidden=32
    mlp_hidden=128
    num_tasks=20
    
    model = RobotTaskPolicy(
        robot_dim=robot_dim,
        task_dim=task_dim,
        edge_dim=edge_dim,
        gnn_hidden=gnn_hidden,
        mlp_hidden=mlp_hidden,
        num_tasks=num_tasks
    )
    
    # 1. 从检查点恢复训练
    start_episode = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_episode = checkpoint['episode'] + 1
        history = checkpoint['history']
        best_advantage = checkpoint.get('best_advantage', float('-inf'))
        print(f"从检查点恢复训练，从第 {start_episode} 回合开始")
    else:
        history = {"rewards": [], "costs": [], "advantages": [], "entropies": []}
        best_advantage = float('-inf')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 3. 动态学习率
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    
    # 如果从检查点恢复，也恢复优化器状态
    if checkpoint_path and os.path.exists(checkpoint_path):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    # 批处理变量
    batch_policy_losses = []
    batch_entropies = []
    
    # 训练循环
    for episode in range(start_episode, num_episodes):
        # 重置环境
        robots = []
        tasks = []
        matrix = read_matrix_from_file("/home/uio/EECBS/map_matrix.txt")
        free_points = find_free_points(matrix)
        start_goal_pairs = assign_start_and_goal_points(free_points, 20)
        for start, goal in start_goal_pairs:
            robots.append(start)
            tasks.append(goal)
            
        # 计算匈牙利算法的基准成本
        cost_matrix = np.zeros((len(robots), len(tasks)))
        for i, robot in enumerate(robots):
            for j, task in enumerate(tasks):
                _, cost = astar_shortest_path(matrix, robot, task)
                cost_matrix[i][j] = cost if cost > 0 else 1e9

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        hungarian_pairs = [(robots[i], tasks[j]) for i, j in zip(row_ind, col_ind)]
        hungarian_cost = calculate_total_cost_from_assignment(hungarian_pairs, matrix, verbose=False)

        # 构建图
        robot_features, task_features, edge_index, edge_features = build_bipartite_graph(robots, tasks, matrix)
        
        # 前向传播获取任务选择概率
        task_probs = model(robot_features, task_features, edge_index, edge_features)
        
        # 2. 计算策略熵
        entropy = -(task_probs * torch.log(task_probs + 1e-10)).sum(dim=1).mean()
        batch_entropies.append(entropy)
        
        # 保存动作的对数概率，用于计算梯度
        assignments = assign_tasks_based_on_probability(task_probs, robot_features, task_features)
        log_probs = []
        for robot_idx, task_idx, _, _, _ in assignments:
            log_probs.append(torch.log(task_probs[robot_idx, task_idx]))
        

        _, start_goal_pairs = print_assignments(task_probs, robot_features, task_features, verbose=False)
        current_cost = calculate_total_cost_from_assignment(start_goal_pairs, matrix, verbose=False)
        advantage_ratio = min(0.5 + episode / num_episodes * 0.5, 1.0)
        advantage = hungarian_cost - current_cost  # 正值表示我们的方案优于匈牙利算法
        
        # 计算策略损失
        episode_policy_loss = []
        for log_prob in log_probs:
            episode_policy_loss.append(-log_prob * advantage * advantage_ratio)
        episode_policy_loss = torch.stack(episode_policy_loss).sum()
        
        # 加入熵正则化
        loss = episode_policy_loss - entropy_coef * entropy
        batch_policy_losses.append(loss)
        
        # 1. 批量处理：累积指定批量后更新
        if len(batch_policy_losses) >= batch_size or episode == num_episodes - 1:
            optimizer.zero_grad()
            total_loss = torch.stack(batch_policy_losses).sum()
            total_loss.backward()
            optimizer.step()
            # 3. 更新学习率
            scheduler.step()
            batch_policy_losses = []
            batch_entropies = []
            

        
        # 记录训练数据
        history["rewards"].append(-current_cost)
        history["costs"].append(current_cost)
        history["advantages"].append(advantage)
        history["entropies"].append(entropy.item())
        
        # 保存表现最好的模型
        if advantage > best_advantage:
            best_advantage = advantage
            checkpoint = {
                'episode': episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_advantage': best_advantage,
                'history': history,
                'scheduler_state_dict': scheduler.state_dict()
            }
            torch.save(checkpoint, "best_model_checkpoint.pth")
            print(f"Episode {episode+1}: 保存最佳模型，优势: {advantage:.2f}")
            
        # 定期保存检查点
        if (episode + 1) % save_interval == 0:
            checkpoint = {
                'episode': episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_advantage': best_advantage,
                'history': history,
                'scheduler_state_dict': scheduler.state_dict()
            }
            torch.save(checkpoint, f"checkpoint_episode_{episode+1}.pth")
            
        # 打印训练进度
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes}, Cost: {current_cost}, " 
                  f"Hungarian: {hungarian_cost}, Advantage: {advantage:.2f}, "
                  f"Entropy: {entropy.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    return model, history

def continue_training_from_checkpoint(checkpoint_path, additional_episodes=500):
    """
    从检查点继续训练
    
    参数:
    - checkpoint_path: 检查点文件路径
    - additional_episodes: 额外训练的回合数
    
    返回:
    - model: 训练好的模型
    - history: 训练历史记录
    """
    checkpoint = torch.load(checkpoint_path)
    episode = checkpoint['episode']
    total_episodes = episode + 1 + additional_episodes
    
    return train_with_policy_gradient(
        num_episodes=total_episodes, 
        checkpoint_path=checkpoint_path,
        save_interval=100
    )

def plot_training_history(history):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(history["rewards"])
    plt.title("奖励变化")
    
    plt.subplot(2, 2, 2)
    plt.plot(history["costs"])
    plt.title("成本变化")
    
    plt.subplot(2, 2, 3)
    plt.plot(history["advantages"])
    plt.title("优势值变化")
    
    plt.subplot(2, 2, 4)
    plt.plot(history["entropies"])
    plt.title("策略熵变化")
    
    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()
    
if __name__ == "__main__":
    # 从零开始训练
    # train_with_policy_gradient(num_episodes=1000, batch_size=5, entropy_coef=0.01)
    
    # 或者从检查点继续训练
    # continue_training_from_checkpoint("checkpoint_episode_500.pth", additional_episodes=500)
    
    # 默认训练
    model, history = train_with_policy_gradient()
    # 然后单独保存历史记录
    torch.save(history, "training_history.pth")
    # 保存模型参数（推荐用于部署）
    torch.save(model.state_dict(), "final_model_params.pth")
    print("模型参数已保存至 final_model_params.pth")
    # 保存完整模型（包括架构）
    torch.save(model, "full_model.pth")
    print("完整模型已保存至 full_model.pth")
        



