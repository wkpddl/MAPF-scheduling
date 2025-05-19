import torch
import numpy as np
import heapq
from math import sqrt
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from torch_geometric.data import Data
from paths import calculate_total_cost, read_paths_from_file
from rm import read_matrix_from_file, run_eecbs, save_pairs_to_scen_file,heuristic,reconstruct_path
from rm import astar_shortest_path,find_free_points ,assign_start_and_goal_points



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
            edge_index.append([i, num_robots+j])  # 从机器人到任务
            edge_features.append([manhattan_distance,actual_distance,difficulty])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # 转置为 [2, num_edges]
    edge_features = torch.tensor(edge_features, dtype=torch.float)

    # 合并节点特征
    x = torch.cat([robot_features, task_features], dim=0)

    # 构建 PyTorch Geometric 的图数据对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_features)
    return robot_features, task_features, edge_index, edge_features

class BipartiteGNNLayer(MessagePassing):
    def __init__(self, robot_dim, task_dim, edge_dim, hidden_dim,out_dim=2):
        """
        二分图GNN层，仅更新机器人节点
        
        参数:
        - robot_dim: 机器人节点特征维度
        - task_dim: 任务节点特征维度
        - edge_dim: 边特征维度
        - hidden_dim: 隐藏层维度
        """
        # 消息传递方向为从任务(j)到机器人(i)
        super(BipartiteGNNLayer, self).__init__(aggr='add')
        
        # 定义消息函数 f_θe
        self.f_theta_e = torch.nn.Sequential(
            torch.nn.Linear(robot_dim + task_dim + edge_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 定义节点更新函数 f_θv
        self.f_theta_v = torch.nn.Sequential(
            torch.nn.Linear(robot_dim + hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x_robot, x_task, edge_index, edge_attr):
        """
        前向传播
        
        参数:
        - x_robot: 机器人节点特征 [num_robots, robot_dim]
        - x_task: 任务节点特征 [num_tasks, task_dim]
        - edge_index: 边索引 [2, num_edges]
        - edge_attr: 边特征 [num_edges, edge_dim]
        """
        # 拼接两种节点的特征
        x = torch.cat([x_robot, x_task], dim=0)
        
        # 消息传递
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, 
                            size=(x.size(0), x.size(0)))
        
        # 只更新机器人节点，任务节点保持不变
        return out[:x_robot.size(0)], x_task
    
    def message(self, x_i, x_j, edge_attr):
        """
        生成消息 - 从任务(j)到机器人(i)
        
        参数:
        - x_i: 目标节点(机器人)特征 [num_edges, robot_dim]
        - x_j: 源节点(任务)特征 [num_edges, task_dim]
        - edge_attr: 边特征 [num_edges, edge_dim]
        """
        # 拼接机器人节点、任务节点和边的特征
        msg = torch.cat([x_i, x_j, edge_attr], dim=1)
        
        # 应用消息函数
        return self.f_theta_e(msg)
    
    def update(self, aggr_out, x):
        """
        更新节点 - 仅更新机器人节点
        
        参数:
        - aggr_out: 聚合后的消息 [num_robots, hidden_dim]
        - x: 原始节点特征 [num_robots, robot_dim]
        """
        # 只取机器人节点的原始特征
        x_robot = x[:aggr_out.size(0)]
        
        # 拼接聚合后的消息和原始机器人特征
        update_input = torch.cat([aggr_out, x_robot], dim=1)
        
        # 应用节点更新函数
        return self.f_theta_v(update_input)


class PolicyNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, robot_embeddings, task_embeddings):
        scores = []
        for robot in robot_embeddings:
            for task in task_embeddings:
                combined = torch.cat([robot, task], dim=0)
                score = self.mlp(combined)
                scores.append(score)
        return torch.stack(scores).view(len(robot_embeddings), len(task_embeddings))


def train_gnn_policy(gnn, policy_net, data, optimizer, robots, tasks, matrix, num_epochs=100):
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # GNN 前向传播
        embeddings = gnn(data)
        robot_embeddings = embeddings[:len(robots)]
        task_embeddings = embeddings[len(robots):]

        # 策略网络前向传播
        scores = policy_net(robot_embeddings, task_embeddings)

        # 生成任务分配
        assignment = torch.argmax(scores, dim=1)
              
        start_goal_pairs = []
        for robot_idx, task_idx in enumerate(assignment):
            start = robots[robot_idx]
            goal = tasks[task_idx.item()]
            start_goal_pairs.append((start, goal))


        # 调用路径规划器（如 EECBS）计算总代价
        total_cost = calculate_total_cost_from_assignment(start_goal_pairs,  matrix)

        # 反向传播优化
        loss = total_cost
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.2f}")


def calculate_total_cost_from_assignment(start_goal_pairs,  matrix):
    paths=[]
    costs = []
    # Print the start-goal pairs
    for i, (start, goal) in enumerate(start_goal_pairs):
        path,cost= astar_shortest_path(matrix, start, goal)
        paths.append(path)
        costs.append(cost)
    # Print the start-goal pairs and their costs
    for i, ((start, goal), cost) in enumerate(zip(start_goal_pairs, costs)):
        print(f"Pair {i + 1}: Start {start}, Goal {goal}, Cost: {cost:.2f}")
    save_pairs_to_scen_file(start_goal_pairs, costs, "map.map", "/home/uio/EECBS/task.scen")
    output = run_eecbs("map.map", "task.scen", "test.csv", "paths.txt", 20, 60, 1.2)
    # Print the output from EECBS
    if output:
        print("EECBS Output:")
        print(output)

    agent_paths = read_paths_from_file("/home/uio/EECBS/paths.txt")
    # Calculate the total cost
    total_cost = calculate_total_cost(agent_paths)
    return total_cost


if __name__ == "__main__":

    robots = []
    tasks = []
    matrix = read_matrix_from_file("/home/uio/EECBS/map_matrix.txt")
    free_points = find_free_points(matrix)
    # Assign 20 start-goal pairs
    start_goal_pairs=assign_start_and_goal_points(free_points,20)
    for i, (start, goal) in enumerate(start_goal_pairs):
        robots.append(start)
        tasks.append(goal)


    robot_features, task_features, edge_index, edge_features=build_bipartite_graph(robots, tasks, matrix)
   
    # 初始化二分图GNN层

    gnn_layer_1= BipartiteGNNLayer(
        robot_dim=2,
        task_dim=2,
        edge_dim=3,
        hidden_dim=32
    )
    updated_robots, tasks = gnn_layer_1(robot_features, task_features, edge_index, edge_features)
    
    gnn_layer_2= BipartiteGNNLayer(
        robot_dim=2,
        task_dim=2,
        edge_dim=3,
        hidden_dim=32
    )


   #train_gnn_policy(gnn, policy_net, data, optimizer, robots, tasks, matrix)