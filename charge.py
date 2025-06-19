import numpy as np
import torch
import heapq
import random
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from gnn import astar_shortest_path_four, build_bipartite_graph, run_eecbs, save_pairs_to_scen_file, read_paths_from_file
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Serif CN']
plt.rcParams['axes.unicode_minus'] = False
class Robot:
    """机器人类，包含位置和电量信息"""
    def __init__(self, id, position, max_battery=100, current_battery=None):
        self.id = id
        self.position = position  # (y, x)
        self.max_battery = max_battery
        self.current_battery = current_battery if current_battery is not None else max_battery
        self.charging = False
        self.assigned_task = None
        self.path = []
        self.status = "idle"  # idle, working, charging, waiting_for_charge
    
    def update_battery(self, consumption):
        """更新电量"""
        if self.charging:
            # 充电状态，电量增加
            self.current_battery = min(self.current_battery + 5, self.max_battery)
            return True
        else:
            # 工作状态，电量减少
            self.current_battery -= consumption
            return self.current_battery > 0
    
    def needs_charging(self, threshold):
        """判断是否需要充电"""
        return self.current_battery < threshold and not self.charging

class ChargingStation:
    """充电桩类"""
    def __init__(self, id, position, capacity=1):
        self.id = id
        self.position = position  # (y, x)
        self.capacity = capacity
        self.occupied = 0
        self.queue = []  # 等待队列
    
    def is_available(self):
        """判断是否有空位"""
        return self.occupied < self.capacity
    
    def add_robot(self, robot):
        """添加机器人到充电桩"""
        if self.is_available():
            self.occupied += 1
            robot.charging = True
            return True
        else:
            self.queue.append(robot)
            robot.status = "waiting_for_charge"
            return False

    def remove_robot(self, robot):
        """移除充电中的机器人"""
        if robot.charging:
            self.occupied -= 1
            robot.charging = False
            
            # 检查队列中是否有等待的机器人
            if self.queue:
                next_robot = self.queue.pop(0)
                self.occupied += 1
                next_robot.charging = True
                next_robot.status = "charging"
            return True
        return False

class BatteryAwareTaskScheduler:
    """基于电量预筛选的任务调度器"""
    def __init__(self, matrix, battery_threshold=30, charging_stations=None):
        self.matrix = matrix
        self.battery_threshold = battery_threshold
        self.robots = []
        self.tasks = []
        self.charging_stations = charging_stations if charging_stations else []
        self.model = None  # GNN模型
    
    def add_robot(self, robot):
        """添加机器人"""
        self.robots.append(robot)
    
    def add_task(self, task_position):
        """添加分拣任务"""
        self.tasks.append(task_position)
    
    def add_charging_station(self, position, capacity=1):
        """添加充电桩"""
        station = ChargingStation(len(self.charging_stations), position, capacity)
        self.charging_stations.append(station)
        return station
    
    def find_nearest_charging_station(self, robot_position):
        """寻找最近的可用充电桩"""
        min_distance = float('inf')
        nearest_station = None
        
        for station in self.charging_stations:
            if station.is_available():
                _, cost = astar_shortest_path_four(self.matrix, robot_position, station.position)
                if cost > 0 and cost < min_distance:
                    min_distance = cost
                    nearest_station = station
        
        return nearest_station, min_distance
    
    def prescreen_robots_by_battery(self):
        """根据电量预筛选机器人"""
        normal_robots = []
        low_battery_robots = []
        
        for robot in self.robots:
            if robot.needs_charging(self.battery_threshold):
                low_battery_robots.append(robot)
            else:
                normal_robots.append(robot)
        
        return normal_robots, low_battery_robots
    
    def generate_charging_tasks(self, low_battery_robots):
        """为低电量机器人生成充电任务"""
        charging_tasks = []
        
        for robot in low_battery_robots:
            nearest_station, distance = self.find_nearest_charging_station(robot.position)
            if nearest_station and (distance/15) < robot.current_battery:  # 确保电量足够到达充电桩
                charging_tasks.append((robot, nearest_station))
                robot.status = "waiting_for_charge"
            else:
                print(f"警告: 机器人 {robot.id} 电量不足以到达最近的充电桩! 电量: {robot.current_battery}, 距离: {distance}")
        
        return charging_tasks
    
    def schedule(self, model=None):
        """执行调度"""
        # 1. 电量预筛选
        normal_robots, low_battery_robots = self.prescreen_robots_by_battery()
        
        # 2. 生成充电任务
        charging_tasks = self.generate_charging_tasks(low_battery_robots)
        
        # 3. 对正常电量机器人进行分拣任务分配(使用GNN或匈牙利算法)
        if not normal_robots or not self.tasks:
            print("没有可用机器人或任务")
            normal_assignments = []
        else:
            normal_assignments = self.assign_tasks_to_normal_robots(normal_robots, model)
        
        # 4. 合并两种任务并返回
        all_assignments = normal_assignments + charging_tasks
        return all_assignments
    
    def assign_tasks_to_normal_robots(self, normal_robots, model=None):
        """为正常电量的机器人分配分拣任务"""
        robot_positions = [robot.position for robot in normal_robots]
        
        # 如果任务数量少于机器人数量，则只使用部分机器人
        num_tasks = min(len(self.tasks), len(normal_robots))
        robot_positions = robot_positions[:num_tasks]
        tasks = self.tasks[:num_tasks]
        
        if not robot_positions or not tasks:
            return []
        
        if model:  # 使用GNN模型进行分配
            robot_features, task_features, edge_index, edge_features = build_bipartite_graph(
                robot_positions, tasks, self.matrix
            )
            
            model.eval()
            with torch.no_grad():
                task_probs = model(robot_features, task_features, edge_index, edge_features, episode=0)
            
            # 概率转换为分配
            probs = task_probs.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(-probs)  # 最大化概率
            
            assignments = []
            for i, j in zip(row_ind, col_ind):
                if i < len(normal_robots) and j < len(tasks):
                    robot = normal_robots[i]
                    task = tasks[j]
                    path, _ = astar_shortest_path_four(self.matrix, robot.position, task)
                    robot.path = path
                    robot.assigned_task = task
                    robot.status = "working"
                    assignments.append((robot, task))
            
            return assignments
            
        else:  # 使用匈牙利算法
            cost_matrix = np.zeros((len(robot_positions), len(tasks)))
            for i, robot_pos in enumerate(robot_positions):
                for j, task_pos in enumerate(tasks):
                    _, cost = astar_shortest_path_four(self.matrix, robot_pos, task_pos)
                    cost_matrix[i, j] = cost if cost > 0 else float('inf')
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            assignments = []
            for i, j in zip(row_ind, col_ind):
                if i < len(normal_robots) and j < len(tasks):
                    robot = normal_robots[i]
                    task = tasks[j]
                    path, _ = astar_shortest_path_four(self.matrix, robot.position, task)
                    robot.path = path
                    robot.assigned_task = task
                    robot.status = "working"
                    assignments.append((robot, task))
            
            return assignments
    
    def update_system(self):
        """更新系统状态（电量、充电完成等）"""
        # 更新充电中的机器人
        for robot in self.robots:
            if robot.charging:
                robot.update_battery(0)  # 充电
                if robot.current_battery >= robot.max_battery * 0.9:
                    # 充电完成，寻找充电桩并释放
                    for station in self.charging_stations:
                        if station.remove_robot(robot):
                            print(f"机器人 {robot.id} 充电完成，电量 {robot.current_battery}/{robot.max_battery}")
                            robot.status = "idle"
                            break
            elif robot.status == "working":
                # 工作中的机器人消耗电量
                if not robot.update_battery(1/15):
                    print(f"警告: 机器人 {robot.id} 电量耗尽!")
                    robot.status = "idle"  # 应该有更好的处理方式

# 从gnn.py引入A*寻路算法和其他辅助函数
def astar_shortest_path_four(matrix, start, goal):
    """从gnn.py引入的A*算法寻找最短路径 (只有四个方向的移动)"""
    rows, cols = len(matrix), len(matrix[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右四个方向
    
    open_set = []
    heapq.heappush(open_set, (0, start))  # (优先级, 位置)
    g_cost = {start: 0}
    f_cost = {start: heuristic(start, goal)}
    came_from = {}
    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, g_cost[current]

        visited.add(current)

        for dr, dc in directions:
            new_row, new_col = current[0] + dr, current[1] + dc
            neighbor = (new_row, new_col)

            # 检查边界
            if not (0 <= new_row < rows and 0 <= new_col < cols):
                continue

            # 检查障碍物 (0表示障碍物)
            if matrix[new_row][new_col] != 1:
                continue

            # 计算当前g值
            tentative_g_cost = g_cost[current] + 1

            if neighbor not in g_cost or tentative_g_cost < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g_cost
                f_cost[neighbor] = tentative_g_cost + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_cost[neighbor], neighbor))
                came_from[neighbor] = current

    return [], -1  # 没找到路径

def heuristic(a, b):
    """曼哈顿距离启发函数"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def reconstruct_path(came_from, current):
    """重建路径"""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def read_matrix_from_file(file_path):
    """从文件读取地图矩阵"""
    try:
        with open(file_path, 'r') as file:
            matrix = []
            for line in file:
                # 将每行转换为整数列表
                row = [int(num) for num in line.strip().split()]
                matrix.append(row)
        return matrix
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在，创建默认地图。")
        return None

def find_free_points(matrix):
    """找出所有可通行点"""
    free_points = []
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1:
                free_points.append((i, j))
    return free_points

def build_bipartite_graph(robots, tasks, matrix):
    """构建二分图，参考gnn.py"""
    H, W = len(matrix), len(matrix[0])
    max_dist = np.sqrt(H*H + W*W)

    robot_features = torch.tensor(robots, dtype=torch.float32)
    task_features = torch.tensor(tasks, dtype=torch.float32)
    
    # 归一化坐标
    robot_features = robot_features / torch.tensor([H, W], dtype=torch.float32)
    task_features = task_features / torch.tensor([H, W], dtype=torch.float32)

    edge_index = []
    edge_features = []
    
    for i, (ry, rx) in enumerate(robots):
        for j, (ty, tx) in enumerate(tasks):
            man_dist = abs(ry-ty) + abs(rx-tx)
            _, act_cost = astar_shortest_path_four(matrix, (ry, rx), (ty, tx))
            if act_cost < 0:
                continue  # 无路径
                
            # 归一化距离特征
            edge_features.append([
                man_dist/max_dist, 
                act_cost/max_dist, 
                (act_cost-man_dist)/max_dist
            ])
            edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_features = torch.tensor(edge_features, dtype=torch.float32)
    
    return robot_features, task_features, edge_index, edge_features


def visualize_system(matrix, robots, tasks, charging_stations, assignments=None):
    """可视化系统状态：黑色墙体白底，并将机器人路径显示出来（去掉绿色虚线）"""
    plt.figure(figsize=(12, 10))
    # 使用 "gray" cmap，确保矩阵中 0 为黑（墙体），1 为白（空地）
    plt.imshow(matrix, cmap='gray', origin='upper')
    
    # 绘制机器人、任务和充电桩
    for robot in robots:
        if robot.status == "idle":
            plt.plot(robot.position[1], robot.position[0], 'bo', markersize=10, label='空闲机器人' if robot.id == 0 else "")
        elif robot.status == "working":
            plt.plot(robot.position[1], robot.position[0], 'go', markersize=10, label='工作中机器人' if robot.id == 0 else "")
        elif robot.status == "charging":
            plt.plot(robot.position[1], robot.position[0], 'yo', markersize=10, label='充电中机器人' if robot.id == 0 else "")
        else:  # waiting_for_charge
            plt.plot(robot.position[1], robot.position[0], 'ro', markersize=10, label='等待充电机器人' if robot.id == 0 else "")
            
        # 绘制电量百分比
        plt.text(robot.position[1], robot.position[0]-0.5, f"{robot.current_battery}%", 
                 color='red' if robot.needs_charging(30) else 'black', 
                 fontsize=9, ha='center')
    
    # 绘制任务标记
    for task in tasks:
        plt.plot(task[1], task[0], 'r*', markersize=12, 
                 label='分拣任务' if tasks.index(task) == 0 else "")
    
    # 绘制充电桩标记
    for station in charging_stations:
        plt.plot(station.position[1], station.position[0], 'ks', markersize=12, 
                 label='充电桩' if station.id == 0 else "")
        
    # 绘制任务分配路径 -- 仅绘制充电任务路径（黄色虚线）
    if assignments:
        for robot, target in assignments:
            if isinstance(target, ChargingStation):  # 充电任务路径
                plt.plot([robot.position[1], target.position[1]], 
                         [robot.position[0], target.position[0]], 
                         'y--', linewidth=2, label='充电路径' if robot.id == 0 else "")
    
    # 绘制机器人规划的详细路径（蓝色线条显示）
    for robot in robots:
        if robot.path:
            path = np.array(robot.path)
            plt.plot(path[:, 1], path[:, 0], 'b-', linewidth=2, alpha=0.8,
                     label='机器人规划路径' if robot.id == 0 else "")
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.title('基于电量预筛选的充电任务处理系统', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('battery_aware_task_scheduling.png', dpi=300)
    plt.show()
    """可视化系统状态：黑色墙体白底，并将机器人路径显示出来"""
    plt.figure(figsize=(12, 10))
    # 使用 "gray" cmap，确保矩阵中 0 为黑（墙体），1 为白（空地）
    plt.imshow(matrix, cmap='gray', origin='upper')
    
    # 绘制机器人、任务和充电桩
    for robot in robots:
        if robot.status == "idle":
            plt.plot(robot.position[1], robot.position[0], 'bo', markersize=10, label='空闲机器人' if robot.id == 0 else "")
        elif robot.status == "working":
            plt.plot(robot.position[1], robot.position[0], 'go', markersize=10, label='工作中机器人' if robot.id == 0 else "")
        elif robot.status == "charging":
            plt.plot(robot.position[1], robot.position[0], 'yo', markersize=10, label='充电中机器人' if robot.id == 0 else "")
        else:  # waiting_for_charge
            plt.plot(robot.position[1], robot.position[0], 'ro', markersize=10, label='等待充电机器人' if robot.id == 0 else "")
            
        # 绘制电量条
        plt.text(robot.position[1], robot.position[0]-0.5, f"{robot.current_battery}%", 
                 color='red' if robot.needs_charging(30) else 'black', 
                 fontsize=9, ha='center')
    
    # 绘制任务标记
    for task in tasks:
        plt.plot(task[1], task[0], 'r*', markersize=12, label='分拣任务' if tasks.index(task) == 0 else "")
    
    # 绘制充电桩标记
    for station in charging_stations:
        plt.plot(station.position[1], station.position[0], 'ks', markersize=12, label='充电桩' if station.id == 0 else "")
        
    # 绘制任务分配路径
    if assignments:
        for robot, target in assignments:
            if isinstance(target, ChargingStation):  # 充电任务路径
                plt.plot([robot.position[1], target.position[1]], 
                         [robot.position[0], target.position[0]], 
                         'y--', linewidth=2, label='充电路径' if robot.id == 0 else "")
            else:  # 分拣任务路径
                plt.plot([robot.position[1], target[1]], 
                         [robot.position[0], target[0]], 
                         'g--', linewidth=2, label='分拣任务路径' if robot.id == 0 else "")
    
    # 绘制机器人规划的详细路径（蓝色线条显示）
    for robot in robots:
        if robot.path:
            path = np.array(robot.path)
            plt.plot(path[:, 1], path[:, 0], 'b-', linewidth=2, alpha=0.8,
                     label='机器人规划路径' if robot.id == 0 else "")
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.title('基于电量预筛选的充电任务处理系统', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('battery_aware_task_scheduling.png', dpi=300)
    plt.show()
def run_eecbs_global_planning(scheduler, matrix):
    """
    使用 EECBS 求解器对所有机器人进行全局路径规划（消除冲突）。
    这里参考 gnn.py 中的调用方式，生成 start-goal 对后调用外部 EECBS 命令，
    最后将 EECBS 求解得到的全局规划路径更新到每个机器人的 robot.path 中。
    
    注意：这里的 run_eecbs 是占位函数，实际使用时请确保可执行文件和场景文件正确生成。
    """
    # 提取所有分配任务的机器人的起点和目标
    start_goal_pairs = []
    for robot in scheduler.robots:
        if robot.assigned_task is not None:
            if isinstance(robot.assigned_task, ChargingStation):
                target = robot.assigned_task.position
            else:
                target = robot.assigned_task
            start_goal_pairs.append( (robot.position, target) )
    
    # 如果没有分配任务则直接返回
    if not start_goal_pairs:
        return scheduler
    
    # 保存 start-goal 对文件（场景文件），参考 gnn.py 的 save_pairs_to_scen_file
    # 此处假设地图保存为 "map.map"，场景文件为 "task.scen" ，路径输出 "paths.txt"
    costs = []
    for start, goal in start_goal_pairs:
        path, cost = astar_shortest_path_four(matrix, start, goal)
        costs.append(cost)
    save_pairs_to_scen_file(start_goal_pairs, costs, "map.map", "/home/uio/EECBS/task.scen")
    
    # 调用 EECBS 求解器（通过 run_eecbs 调用外部命令）
    output = run_eecbs("map.map", "/home/uio/EECBS/task.scen", "result.csv", "/home/uio/EECBS/paths.txt", 
                       num_agents=len(start_goal_pairs), time_limit=60, suboptimality=1.2, verbose=True)
    
    # 读取 EECBS 输出路径（参考 gnn.py 的 read_paths_from_file）
    global_paths = read_paths_from_file("/home/uio/EECBS/paths.txt")
    
    # 将求解到的全局规划路径更新到各机器人（假设顺序与 start_goal_pairs一致）
    for robot, path in zip([r for r in scheduler.robots if r.assigned_task is not None], global_paths):
        robot.path = path
    
    return scheduler

def demonstrate_charging_system_with_eecbs():
    """
    演示整个系统运行流程：
    1. 读取地图矩阵（黑底白墙）
    2. 利用电量预筛选和 GNN 任务分配获得初步分配
    3. 使用 EECBS 对所有机器人进行全局无冲突路径规划
    4. 可视化最终规划结果（显示机器人、任务、充电桩及蓝色规划路径）
    """
    print("开始演示基于电量预筛选 + GNN + EECBS 的全局路径规划系统...")
    
    matrix = read_matrix_from_file("/home/uio/EECBS/map_matrix.txt")
    if matrix is None:
        rows, cols = 30, 30
        matrix = np.ones((rows, cols))
        # 生成仓库风格地图（0为墙体，1为空地）
        for i in range(3, rows-3, 6):
            for j in range(3, cols-3, 4):
                for dx in range(3):
                    for dy in range(4):
                        if i+dy < rows and j+dx < cols:
                            matrix[i+dy, j+dx] = 0

    free_points = find_free_points(matrix)
    scheduler = BatteryAwareTaskScheduler(matrix, battery_threshold=25)
    
    # 在四角区域布置充电桩
    rows, cols = len(matrix), len(matrix[0])
    corner_regions = [
        (0, 0, rows//5, cols//5),
        (0, cols - cols//5, rows//5, cols),
        (rows - rows//5, 0, rows, cols//5),
        (rows - rows//5, cols - cols//5, rows, cols)
    ]
    charging_positions = []
    for min_y, min_x, max_y, max_x in corner_regions:
        region_free = [(y, x) for y in range(min_y, max_y) for x in range(min_x, max_x)]
        if region_free:
            center = ((min_y + max_y)//2, (min_x + max_x)//2)
            best = min(region_free, key=lambda pos: abs(pos[0]-center[0]) + abs(pos[1]-center[1]))
            charging_positions.append(best)
            if best in free_points:
                free_points.remove(best)
    
    # 如果充电桩数量不足，补充一些额外的充电桩
    desired_charging_count = 8  # 可根据需要调整
    while len(charging_positions) < desired_charging_count and free_points:
        pos = random.choice(free_points)
        free_points.remove(pos)
        charging_positions.append(pos)
    
    for pos in charging_positions:
        scheduler.add_charging_station(pos, capacity=2)
    
    # 以下部分保持原样：添加机器人、任务，执行调度及全局路径规划
    robot_count = 15
    robot_positions = []
    for i in range(robot_count):
        if not free_points:
            break
        pos = random.choice(free_points)
        free_points.remove(pos)
        robot_positions.append(pos)
    for i, pos in enumerate(robot_positions):
        if i < 3:
            battery = random.randint(15, 25)
        elif i < 7:
            battery = random.randint(26, 35)
        elif i < 12:
            battery = random.randint(36, 70)
        else:
            battery = random.randint(71, 100)
        robot = Robot(i, pos, max_battery=100, current_battery=battery)
        scheduler.add_robot(robot)
    
    task_count = 12
    task_positions = []
    for i in range(task_count):
        if not free_points:
            break
        pos = random.choice(free_points)
        free_points.remove(pos)
        task_positions.append(pos)
        scheduler.add_task(pos)
    
    # 初始状态可视化（机器人、任务、充电桩显示在黑底白墙地图上）
    plt.figure(figsize=(12, 10))
    plt.imshow(matrix, cmap='gray', origin='upper')
    for robot in scheduler.robots:
        color = 'ro' if robot.needs_charging(scheduler.battery_threshold) else 'bo'
        plt.plot(robot.position[1], robot.position[0], color, markersize=10)
        plt.text(robot.position[1], robot.position[0]-0.5, f"{robot.current_battery}%", 
                 color='red' if robot.needs_charging(scheduler.battery_threshold) else 'black', fontsize=9, ha='center')
    for pos in task_positions:
        plt.plot(pos[1], pos[0], 'r*', markersize=12)
    for station in scheduler.charging_stations:
        plt.plot(station.position[1], station.position[0], 'ks', markersize=12)
    plt.title('系统初始状态 - 未进行电量预筛选', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('initial_state.png', dpi=300)
    plt.show()
    
    # 执行任务调度（此处 GNN 分配和电量预筛选已内置）
    assignments = scheduler.schedule()
    print(f"调度后分配：充电任务 {len([a for a in assignments if isinstance(a[1], ChargingStation)])}，分拣任务 {len([a for a in assignments if not isinstance(a[1], ChargingStation)])}")
    
    # 调用 EECBS 对所有已分配机器人的路径进行全局规划
    scheduler = run_eecbs_global_planning(scheduler, matrix)
    
    # 最终结果可视化：黑色墙体白底，机器人规划路径以蓝色显示
    visualize_system(matrix, scheduler.robots, scheduler.tasks, scheduler.charging_stations, assignments)
    
    print("演示结束！") 
if __name__ == "__main__":
    demonstrate_charging_system_with_eecbs()

