import matplotlib.pyplot as plt
import numpy as np

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

def plot_map_with_agent_paths(matrix, paths):
    """
    Plots the map and the paths of agents.

    Args:
        matrix (list[list[int]]): The 2D map matrix.
        paths (list[list[tuple[int, int]]]): List of paths for each agent.
    """
    plt.figure(figsize=(12, 12))
    matrix = np.array(matrix)
    plt.imshow(matrix, cmap="Greys", origin="upper")

    colors = plt.cm.get_cmap("tab10", len(paths))  # Use a colormap for different agents

    for i, path in enumerate(paths):
        path = np.array(path)
        plt.plot(path[:, 1], path[:, 0], label=f"Agent {i + 1}", color=colors(i))
        plt.scatter(path[0, 1], path[0, 0], color=colors(i), marker='o', label=f"Start {i + 1}")
        plt.scatter(path[-1, 1], path[-1, 0], color=colors(i), marker='x', label=f"Goal {i + 1}")

    plt.legend()
    plt.title("Agent Paths on Map")
    plt.show()


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


if __name__ == "__main__":
    # Replace with the path to your map file
    file_path = "/home/uio/EECBS/map_matrix.txt"
    paths_file = "/home/uio/EECBS/paths.txt"

    # Read the map matrix
    matrix = read_matrix_from_file(file_path)


    # Read the paths from the paths.txt file
    agent_paths = read_paths_from_file(paths_file)

    
    # Calculate the total cost
    total_cost = calculate_total_cost(agent_paths)
    print(f"Total cost of all paths: {total_cost:.5f}")

    # Plot the map with agent paths
    plot_map_with_agent_paths(matrix, agent_paths)