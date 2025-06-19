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
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'Droid Sans Fallback', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


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
    def __init__(self, robot_dim, task_dim, edge_dim, hidden_dim):
        """
        二分图GNN层，用于处理机器人和任务之间的消息传递
        
        参数:
        - robot_dim: 机器人节点特征维度
        - task_dim: 任务节点特征维度
        - edge_dim: 边特征维度
        - hidden_dim: 隐藏层维度
        """
        super(BipartiteGNNLayer, self).__init__(aggr='add')
        
        # 消息函数
        self.f_phi_e = torch.nn.Sequential(
            torch.nn.Linear(robot_dim + task_dim + edge_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 节点更新函数
        self.f_phi_v = torch.nn.Sequential(
            torch.nn.Linear(robot_dim + hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
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
        - updated_robots: 更新后的机器人特征 [num_robots, hidden_dim]
        - x_task: 任务特征保持不变 [num_tasks, task_dim]
        """
        x = torch.cat([x_robot, x_task], dim=0)
        # 获取聚合消息
        robot_indices = edge_index[0]
        task_indices = edge_index[1] + x_robot.size(0)
        edge_index_adj = torch.stack([robot_indices, task_indices], dim=0)
        
        out = self.propagate(edge_index_adj, x=x, edge_attr=edge_attr, size=(x.size(0), x.size(0)))
        
        # 仅更新机器人节点
        robot_messages = out[:x_robot.size(0)]
        robot_update_input = torch.cat([robot_messages, x_robot], dim=1)
        updated_robots = self.f_phi_v(robot_update_input)
        
        return updated_robots, x_task
    
    def message(self, x_i, x_j, edge_attr):
        """
        定义消息函数
        
        参数:
        - x_i: 源节点特征
        - x_j: 目标节点特征
        - edge_attr: 边特征
        
        返回:
        - msg: 消息 [num_edges, hidden_dim]
        """
        # 组合源节点、目标节点和边特征
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.f_phi_e(msg_input)

class EnhancedBipartiteGNNLayer(MessagePassing):
    def __init__(self, robot_dim, task_dim, edge_dim, hidden_dim, dropout=0.0):
        super(EnhancedBipartiteGNNLayer, self).__init__(aggr='add')
        self.f_theta_e = torch.nn.Sequential(
            torch.nn.Linear(robot_dim + task_dim + edge_dim, hidden_dim),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.f_theta_v = torch.nn.Sequential(
            torch.nn.Linear(robot_dim + hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(negative_slope=0.1),
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
    def __init__(self, robot_hidden_dim, task_dim, num_tasks, mlp_hidden=128, temperature=1.0, num_episodes=1000):
        super(TaskSelectionMLP, self).__init__()
        self.temperature = temperature
        self.num_episodes = num_episodes
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(robot_hidden_dim + task_dim, mlp_hidden),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Linear(mlp_hidden, 1)
        )
    def forward(self, robot_features, task_features, c_episode=None):
        N, H = robot_features.shape
        M, D = task_features.shape
        robot_expanded = robot_features.unsqueeze(1).expand(-1, M, -1)
        task_expanded = task_features.unsqueeze(0).expand(N, -1, -1)
        combined = torch.cat([robot_expanded, task_expanded], dim=2)
        scores = self.mlp(combined.reshape(-1, H + D))
        scores = scores.reshape(N, M)
        scores = torch.clamp(scores, -10.0, 10.0)
        scores = scores - scores.max(dim=1, keepdim=True)[0]
        if c_episode is not None:
            current_temp = max(self.temperature * (1.0 - c_episode / self.num_episodes), 0.1)
        else:
            current_temp = self.temperature
        task_probs = F.softmax(scores / current_temp, dim=1)
        task_probs = torch.clamp(task_probs, 1e-6, 1.0 - 1e-6)
        return task_probs

class RobotTaskPolicy(torch.nn.Module):
    def __init__(self, robot_dim, task_dim, edge_dim, gnn_hidden, mlp_hidden, num_tasks, num_gnn_layers=3):
        super(RobotTaskPolicy, self).__init__()
        self.robot_proj = torch.nn.Linear(robot_dim, gnn_hidden)
        self.task_proj = torch.nn.Linear(task_dim, gnn_hidden)
        self.gnn_layers = torch.nn.ModuleList([
            EnhancedBipartiteGNNLayer(
                robot_dim=gnn_hidden,
                task_dim=gnn_hidden,
                edge_dim=edge_dim,
                hidden_dim=gnn_hidden,
                dropout=0.0
            ) for _ in range(num_gnn_layers)
        ])
        self.skip_connections = torch.nn.ModuleList([
            torch.nn.Linear(gnn_hidden, gnn_hidden)
            for _ in range(max(0, num_gnn_layers - 1))
        ])
        self.task_selector = TaskSelectionMLP(
            robot_hidden_dim=gnn_hidden,
            task_dim=gnn_hidden,
            num_tasks=num_tasks,
            mlp_hidden=mlp_hidden,
            num_episodes=1000
        )
        self.num_gnn_layers = num_gnn_layers
    def forward(self, x_robot, x_task, edge_index, edge_attr, episode):
        robot_hidden = self.robot_proj(x_robot)
        task_hidden = self.task_proj(x_task)
        layer_outputs = [robot_hidden]
        for i in range(self.num_gnn_layers):
            new_robot_hidden, _ = self.gnn_layers[i](robot_hidden, task_hidden, edge_index, edge_attr)
            if i > 0 and i < len(self.skip_connections) + 1:
                skip_input = layer_outputs[max(0, i - 2)]
                new_robot_hidden = new_robot_hidden + self.skip_connections[i - 1](skip_input)
            robot_hidden = robot_hidden + new_robot_hidden
            layer_outputs.append(robot_hidden)
        return self.task_selector(robot_hidden, task_hidden, c_episode=episode)



def build_bipartite_graph(robots, tasks, matrix):
    H, W = len(matrix), len(matrix[0])
    max_dist = sqrt(H*H + W*W)

    robot_features = torch.tensor(robots, dtype=torch.float32)
    task_features  = torch.tensor(tasks,  dtype=torch.float32)
    # 归一化坐标 -> [0,1]
    robot_features /= torch.tensor([H, W], dtype=torch.float32)
    task_features  /= torch.tensor([H, W], dtype=torch.float32)

    edge_index = []
    edge_features = []
    for i,(ry,rx) in enumerate(robots):
        for j,(ty,tx) in enumerate(tasks):
            man = abs(ry-ty)+abs(rx-tx)
            _,act = astar_shortest_path_four(matrix,(ry,rx),(ty,tx))
            if act<0: continue
            # 归一化距离
            edge_features.append([man/max_dist, act/max_dist, ((act-man)/max_dist)])
            edge_index.append([i,j])

    edge_index   = torch.tensor(edge_index,dtype=torch.long).t().contiguous()
    edge_features= torch.tensor(edge_features,dtype=torch.float32)
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

def calculate_normalized_advantage(batch_advantages, advantage):
    """
    增强版优势归一化计算，提高训练稳定性
    
    参数:
    - batch_advantages: 批次内所有优势值列表
    - advantage: 当前需要归一化的优势值
    
    返回:
    - normalized_advantage: 归一化后的优势值
    """
    # 使用非零最小标准差确保稳定性
    if len(batch_advantages) >= 5:  # 增加最小样本要求
        adv_mean = np.mean(batch_advantages)
        adv_std = max(np.std(batch_advantages) + 1e-8, 0.5)  # 增加最小标准差
        normalized_advantage = (advantage - adv_mean) / adv_std
    else:
        # 样本不足时使用原始值除以常数进行归一化
        normalized_advantage = advantage / 1000.0
    
    # 使用更严格的裁剪限制
    return torch.clamp(torch.tensor(normalized_advantage, dtype=torch.float32), -1.0, 1.0)

def adaptive_entropy_coefficient(episode, base_coef=0.01, min_coef=0.001, decay_factor=0.995):
    """
    自适应熵系数调整函数
    
    参数:
    - episode: 当前训练回合
    - base_coef: 基础熵系数
    - min_coef: 最小熵系数
    - decay_factor: 衰减因子
    
    返回:
    - current_entropy_coef: 当前适用的熵系数
    """
    # 训练早期使用更大的熵系数鼓励探索
    if episode < 1000:
        return max(base_coef * (decay_factor ** (episode / 10)), min_coef)
    # 训练中期适度探索
    elif episode < 10000:
        return max(base_coef * (decay_factor ** (episode / 50)), min_coef)
    # 训练后期减小探索
    else:
        return min_coef

def calculate_optimal_learning_rate(episode, warmup_episodes, min_lr, max_lr):
    """
    计算优化的学习率，使用平滑的预热曲线
    
    参数:
    - episode: 当前训练回合
    - warmup_episodes: 预热期回合数
    - min_lr: 最小学习率
    - max_lr: 最大学习率
    
    返回:
    - current_lr: 当前应用的学习率
    """
    if episode < warmup_episodes:
        # 使用更平缓的预热曲线(余弦预热)
        progress = episode / warmup_episodes
        curr_lr = min_lr + (max_lr - min_lr) * (np.sin((progress - 0.5) * np.pi) / 2 + 0.5)
    else:
        # 预热期结束后使用最大学习率
        curr_lr = max_lr
    return curr_lr

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
    
    return train_with_consistent_batch_tasks(
        num_episodes=total_episodes, 
        checkpoint_path=checkpoint_path,
        save_interval=100
    )

def calculate_normalized_advantage(raw_advantage, baseline, batch_advantages, baseline_alpha=0.9):
    """优化的优势计算函数 - 保持梯度流版本"""
    # 保存真实优势值用于显示和分析
    debug_adv = raw_advantage
    
    # 将原始优势值转为tensor并裁剪
    raw_advantage_tensor = torch.tensor(raw_advantage, dtype=torch.float32)
    
    # 更新基线值 - 在计算图外更新基线（基线通常不需要梯度）
    if baseline is None:
        baseline = raw_advantage
    else:
        baseline = baseline_alpha * baseline + (1 - baseline_alpha) * raw_advantage
    
    # 计算优势(原始优势 - 基线)，保持tensor形式
    advantage_value = raw_advantage - baseline
    
    # 记录到批次优势列表(使用实际值)
    batch_advantages.append(advantage_value)
    
    # 归一化优势计算
    if len(batch_advantages) >= 5:
        adv_mean = np.mean(batch_advantages)
        # 增加最小标准差，避免归一化到接近0
        adv_std = max(np.std(batch_advantages), 10.0)
        normalized_advantage = (advantage_value - adv_mean) / adv_std
    else:
        normalized_advantage = advantage_value / 50.0
    
    # 将归一化优势转为tensor并适当裁剪
    normalized_advantage_tensor = torch.clamp(
        torch.tensor(normalized_advantage, dtype=torch.float32), 
        -2.0, 2.0
    )
    
    return normalized_advantage_tensor, debug_adv, advantage_value, baseline

def sample_tasks_differentiable(task_probs, robots, tasks):
    """使用PyTorch的可微分采样进行任务分配，保持计算图完整"""
    num_robots, num_tasks = task_probs.shape
    
    # 初始化结果列表和对数概率列表
    assignments = []
    log_probs = []
    
    # 未分配的机器人和任务集合
    unassigned_robots = set(range(num_robots))
    unassigned_tasks = set(range(min(num_robots, num_tasks)))
    
    # 一对一分配过程
    while unassigned_robots and unassigned_tasks:
        # 构建当前未分配的子概率矩阵
        current_robot_indices = list(unassigned_robots)
        current_task_indices = list(unassigned_tasks)
        
        # 创建掩码后的概率矩阵
        masked_probs = torch.zeros_like(task_probs)
        for r_idx in current_robot_indices:
            for t_idx in current_task_indices:
                masked_probs[r_idx, t_idx] = task_probs[r_idx, t_idx]
        
        # 归一化
        sum_probs = masked_probs.sum() + 1e-10
        robot_task_probs = masked_probs / sum_probs
        
        # 展平概率
        flat_probs = robot_task_probs.reshape(-1)
        
        # 使用PyTorch的Categorical分布进行可微分采样
        distribution = torch.distributions.Categorical(probs=flat_probs)
        flat_action = distribution.sample()  # 保持计算图连接的采样
        
        # 计算对数概率（保持梯度流）
        action_log_prob = distribution.log_prob(flat_action)
        
        # 分离动作索引（这里需要分离，但保存了计算图中的log_prob）
        flat_action_idx = flat_action.item()  # 只有在这里使用item()，之前已保存了梯度信息
        robot_idx, task_idx = divmod(flat_action_idx, num_tasks)
        
        # 添加对数概率到列表，这里保持了梯度流
        log_probs.append(action_log_prob)
        
        # 只处理有效分配（概率不为零）
        if masked_probs[robot_idx, task_idx] > 0:
            assignments.append((
                robot_idx, 
                task_idx, 
                robots[robot_idx], 
                tasks[task_idx]
            ))
            
            # 从未分配集合中移除
            if robot_idx in unassigned_robots:
                unassigned_robots.remove(robot_idx)
            if task_idx in unassigned_tasks:
                unassigned_tasks.remove(task_idx)
    
    return assignments, log_probs

def calculate_entropy(probs):
    """计算策略熵"""
    # 更严格的裁剪，确保概率值在安全范围内
    safe_probs = torch.clamp(probs, 1e-7, 1.0 - 1e-7)
    log_probs = torch.log(safe_probs)
    # 计算熵并处理可能的NaN值
    entropy = -(safe_probs * log_probs).sum(dim=1).mean()
    if torch.isnan(entropy):
        print(f"警告: 检测到熵为NaN，使用默认值0.0")
        return torch.tensor(0.0)
    return entropy

def convert_assignments_to_pairs(assignments):
    """将分配结果转换为起点终点对"""
    start_goal_pairs = []
    for robot_idx, task_idx, robot_pos, task_pos in assignments:
        if isinstance(robot_pos, torch.Tensor):
            start = (int(robot_pos[0].item()), int(robot_pos[1].item()))
        else:
            start = (int(robot_pos[0]), int(robot_pos[1]))
            
        if isinstance(task_pos, torch.Tensor):
            goal = (int(task_pos[0].item()), int(task_pos[1].item()))
        else:
            goal = (int(task_pos[0]), int(task_pos[1]))
            
        start_goal_pairs.append((start, goal))
    
    return assignments, start_goal_pairs

def gumbel_softmax_sample(logits, tau=1.0, hard=False):
    """Gumbel-Softmax采样，返回[N, M]软分配矩阵"""
    gumbels = -torch.empty_like(logits).exponential_().log()
    y = (logits + gumbels) / tau
    y_soft = F.softmax(y, dim=-1)
    if hard:
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        return (y_hard - y_soft).detach() + y_soft
    else:
        return y_soft

def soft_assignment_cost(assign_matrix, cost_matrix):
    """根据软分配矩阵和代价矩阵计算soft cost"""
    # assign_matrix: [N, M], cost_matrix: [N, M]
    return (assign_matrix * cost_matrix).sum()

def train_with_consistent_batch_tasks(
    num_episodes=1000, learning_rate=0.0001, save_interval=100,
    batch_size=50, entropy_coef=0.05, checkpoint_path=None,
    weight_decay=5e-4, seed=42, grad_accum_steps=8
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    matrix = read_matrix_from_file("/home/uio/EECBS/map_matrix.txt")
    robot_dim = 2
    task_dim = 2
    edge_dim = 3
    num_tasks = 20
    model = RobotTaskPolicy(
        robot_dim=robot_dim,
        task_dim=task_dim,
        edge_dim=edge_dim,
        gnn_hidden=64,
        mlp_hidden=256,
        num_tasks=num_tasks,
        num_gnn_layers=3
    )
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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=max(2, num_episodes // batch_size * 2),
        T_mult=2,
        eta_min=1e-5
    )
    if checkpoint_path and os.path.exists(checkpoint_path):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    baseline = 0
    baseline_alpha = 0.95
    warmup_episodes = 100
    min_lr = learning_rate * 0.2
    max_lr = learning_rate
    for batch_start in range(start_episode, num_episodes, batch_size):
        robots = []
        tasks = []
        free_points = find_free_points(matrix)
        start_goal_pairs = assign_start_and_goal_points(free_points, 20)
        for start, goal in start_goal_pairs:
            robots.append(start)
            tasks.append(goal)
        cost_matrix = np.zeros((len(robots), len(tasks)))
        for i, robot in enumerate(robots):
            for j, task in enumerate(tasks):
                _, cost = astar_shortest_path(matrix, robot, task)
                cost_matrix[i][j] = cost if cost > 0 else 1e9
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        hungarian_pairs = [(robots[i], tasks[j]) for i, j in zip(row_ind, col_ind)]
        hungarian_cost = calculate_total_cost_from_assignment(hungarian_pairs, matrix, verbose=False)
        robot_features, task_features, edge_index, edge_features = build_bipartite_graph(robots, tasks, matrix)
        batch_policy_losses = []
        batch_entropies = []
        batch_advantages = []
        batch_assignments = []
        optimizer.zero_grad()
        batch_end = min(batch_start + batch_size, num_episodes)
        for episode in range(batch_start, batch_end):
            # 学习率warmup
            if episode < warmup_episodes:
                progress = episode / warmup_episodes
                curr_lr = min_lr + (max_lr - min_lr) * (np.sin((progress - 0.5) * np.pi) / 2 + 0.5)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = curr_lr
            # 增强探索：训练初期加噪声
            task_probs = model(robot_features, task_features, edge_index, edge_features, episode)  # [N, M]
            if episode < 5000:
                task_probs = task_probs + torch.randn_like(task_probs) * 0.05
                task_probs = torch.clamp(task_probs, 1e-6, 1.0 - 1e-6)
                task_probs = task_probs / task_probs.sum(dim=1, keepdim=True)
            # 自适应熵系数
            entropy_coef = max(entropy_coef * (0.999 ** (episode // 100)), 0.01)
            # --- 严格RL动作采样 ---
            m = torch.distributions.Categorical(probs=task_probs)
            actions = m.sample()  # [N]
            log_probs = m.log_prob(actions)  # [N]
            entropy = m.entropy().mean()
            # 构造 one-hot 分配矩阵
            assign_matrix = torch.zeros_like(task_probs)
            assign_matrix[torch.arange(len(actions)), actions] = 1.0
            torch_cost_matrix = torch.tensor(cost_matrix, dtype=torch.float32)
            current_cost = soft_assignment_cost(assign_matrix, torch_cost_matrix)
            # 优化reward
            reward = -(current_cost - hungarian_cost) / (hungarian_cost + 1e-6)
            # 基线更新
            if baseline is None:
                baseline = reward.item()
            else:
                baseline = baseline_alpha * baseline + (1 - baseline_alpha) * reward.item()
            advantage = reward - baseline
            batch_advantages.append(advantage.item())
            # 归一化优势
            if len(batch_advantages) >= 5:
                adv_mean = np.mean(batch_advantages)
                adv_std = max(np.std(batch_advantages), 0.5)
                normalized_advantage = (advantage.item() - adv_mean) / adv_std
            else:
                normalized_advantage = advantage.item() / 10.0
            normalized_advantage = torch.clamp(torch.tensor(normalized_advantage, dtype=torch.float32), -2.0, 2.0)
            # policy loss（严格RL）
            policy_loss = -normalized_advantage * log_probs.sum()
            # imitation loss 可选
            if episode % 5 == 0:
                probs = task_probs.detach().cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(-probs)
                hungarian_label = np.zeros_like(probs)
                hungarian_label[row_ind, col_ind] = 1
                hungarian_label = torch.tensor(hungarian_label, dtype=task_probs.dtype, device=task_probs.device)
                imitation_loss = F.binary_cross_entropy(task_probs, hungarian_label)
                loss = policy_loss - entropy_coef * entropy + 5.0 * imitation_loss
            else:
                loss = policy_loss - entropy_coef * entropy
            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()
            # 梯度异常检查
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"警告: {name} 梯度出现NaN")
            should_update = ((episode - batch_start + 1) % grad_accum_steps == 0) or (episode == batch_end - 1)
            if should_update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            history["rewards"].append(reward.item())
            history["costs"].append(current_cost.item())
            history["advantages"].append(advantage.item())
            history["entropies"].append(entropy.item())
            batch_entropies.append(entropy.item())
            if advantage.item() > best_advantage:
                best_advantage = advantage.item()
                checkpoint = {
                    'episode': episode,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_advantage': best_advantage,
                    'history': history,
                    'scheduler_state_dict': scheduler.state_dict()
                }
                torch.save(checkpoint, "best_model_checkpoint.pth")
                print(f"Episode {episode+1}: 保存最佳模型，优势: {advantage.item():.2f}")
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
            if (episode + 1) % 5 == 0:
                print(
                    f"回合 {episode+1}/{num_episodes} | "
                    f"匈牙利算法成本: {hungarian_cost:.2f} | "
                    f"GNN模型成本: {current_cost.item():.2f} | "
                    f"奖励: {reward.item():.4f} | "
                    f"归一化优势: {normalized_advantage.item():.4f} | "
                    f"熵: {entropy.item():.4f} | "
                    f"熵系数: {entropy_coef:.6f} | "
                    f"学习率: {optimizer.param_groups[0]['lr']:.6f} | "
                    f"基线: {baseline:.2f}"
                )
        if len(batch_advantages) > 0:
            best_batch_advantage = max(batch_advantages)
            avg_batch_advantage = sum(batch_advantages) / len(batch_advantages)
            avg_entropy = sum(batch_entropies) / len(batch_entropies) if len(batch_entropies) > 0 else 0.0
            print(
                f"批次{batch_start//batch_size + 1} | "
                f"最佳优势值: {best_batch_advantage:.2f} | "
                f"平均优势值: {avg_batch_advantage:.2f} | "
                f"平均熵值: {avg_entropy:.4f} | "
                f"累计进度: {(batch_end/num_episodes)*100:.2f}%"
            )
        scheduler.step()
        batch_advantages.clear()
        batch_entropies.clear()
    return model, history


if __name__ == "__main__":
    # 修改训练参数
    model, history = train_with_consistent_batch_tasks(
        num_episodes=10000, 
        learning_rate=0.0005,     # 提高基础学习率
        save_interval=100, 
        batch_size=1280,           # 更合理的批大小
        entropy_coef=0.05,        # 增加初始探索系数
        weight_decay=5e-5,        # 减少权重衰减
        seed=42,
        grad_accum_steps=2        # 减少累积步数提高更新频率
    )
    
    # 保存训练结果
    torch.save(history, "training_history.pth")
    torch.save(model.state_dict(), "final_model_params.pth")
    print("模型参数已保存至 final_model_params.pth")
    torch.save(model, "full_model.pth")
    print("完整模型已保存至 full_model.pth")
    
    # 绘制训练曲线
    plt.figure(figsize=(16, 8))
    plt.subplot(2, 2, 1)
    plt.plot(history["rewards"])
    plt.title("奖励曲线")
    plt.xlabel("回合")
    plt.ylabel("奖励值")
    
    plt.subplot(2, 2, 2)
    plt.plot(history["advantages"])
    plt.title("优势值曲线")
    plt.xlabel("回合")
    plt.ylabel("优势值")
    
    plt.subplot(2, 2, 3)
    plt.plot(history["entropies"])
    plt.title("熵值曲线")
    plt.xlabel("回合")
    plt.ylabel("熵")
    
    plt.subplot(2, 2, 4)
    plt.plot(history["costs"])
    plt.title("成本曲线")
    plt.xlabel("回合")
    plt.ylabel("路径成本")
    
    plt.tight_layout()
    plt.savefig("training_curves.png")
    print("训练曲线已保存至 training_curves.png")
