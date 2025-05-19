import random
import heapq
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import subprocess

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


if __name__ == "__main__":
    # Replace with the path to your .txt file
    file_path = "/home/uio/EECBS/map_matrix.txt"

    # Read the matrix from the file
    matrix = read_matrix_from_file(file_path)

    # Find all free points
    free_points = find_free_points(matrix)

    # Assign 20 start-goal pairs
    num_pairs = 20
    start_goal_pairs = assign_start_and_goal_points(free_points, num_pairs)

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

    #plot_map_with_paths(matrix, start_goal_pairs, paths)

        # Save the pairs to a scenario file
    map_name = "map.map"  # Replace with your map name
    output_scen_file = "/home/uio/EECBS/task.scen"
    save_pairs_to_scen_file(start_goal_pairs, costs, map_name, output_scen_file)

    print(f"Scenario file saved to {output_scen_file}")

    # Run EECBS
    output = run_eecbs("map.map", "task.scen", "test.csv", "paths.txt", num_pairs, 60, 1.2)

    # Print the output from EECBS
    if output:
        print("EECBS Output:")
        print(output)