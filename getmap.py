def parse_map_file(file_path):
    """
    Parses a .map file and converts the map section into a 2D matrix.

    Args:
        file_path (str): Path to the .map file.

    Returns:
        list[list[int]]: 2D matrix representation of the map with 1 for passable and 0 for impassable.
    """
    matrix = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        map_section = False
        for line in lines:
            # Start processing after the "map" keyword
            if line.strip() == "map":
                map_section = True
                continue
            if map_section:
                # Convert each character to 1 (passable) or 0 (impassable)
                matrix.append([1 if char == '.' else 0 for char in line.strip()])
    return matrix


def save_matrix_to_file(matrix, output_file):
    """
    Saves a 2D matrix to a file.

    Args:
        matrix (list[list[int]]): The 2D matrix to save.
        output_file (str): Path to the output file.
    """
    with open(output_file, 'w') as file:
        for row in matrix:
            file.write(" ".join(map(str, row)) + "\n")


if __name__ == "__main__":
    # Replace with the path to your .map file
    file_path = "/home/uio/EECBS/map.map"
    output_file = "/home/uio/EECBS/map_matrix.txt"

    # Parse the map file and save the matrix
    matrix = parse_map_file(file_path)
    save_matrix_to_file(matrix, output_file)

    print(f"Matrix saved to {output_file}")