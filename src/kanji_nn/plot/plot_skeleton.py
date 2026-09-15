import numpy as np
import matplotlib.pyplot as plt

# TODO: deprecated
def plot_skeleton(glyph):
    """
    Plots the raw, fragmented paths extracted by Skan.
    Each distinct topological branch is colored differently.
    """
    # Extract structural components from your glyph dictionary
    skeleton = glyph["skeleton"]
    size = glyph["size"]
    coordinates = skeleton.coordinates

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, size[0])
    ax.set_ylim(size[1], 0)

    # Optional: Plot the original binary skeleton in the background for reference
    # ax.imshow(skan_obj.skeleton, cmap='gray_r', alpha=0.15)

    # Loop through every isolated path fragment found by Skan
    for i in range(skeleton.n_paths):
        # Skan provides the node IDs belonging to path 'i'
        path_nodes = skeleton.path(i)

        # Pull the actual Y, X pixel coordinates for these nodes
        # Note: Skan indices coordinates as (row, col) i.e., (Y, X)
        path_coords = coordinates[path_nodes]

        y_coords = path_coords[:, 0]
        x_coords = path_coords[:, 1]

        # Plot the line fragment
        ax.plot(x_coords, y_coords, linewidth=3, label=f"Path {i}")

        # Draw small markers on the individual pixels
        ax.scatter(x_coords, y_coords, s=15, zorder=3)

    # Highlight the junctions/endpoints using degrees
    degrees = np.array(skeleton.degrees)
    junctions = coordinates[degrees > 2]
    endpoints = coordinates[degrees == 1]

    if len(junctions) > 0:
        ax.scatter(junctions[:, 1], junctions[:, 0], color='red', marker='X', s=100, zorder=4, label='Junction')
    if len(endpoints) > 0:
        ax.scatter(endpoints[:, 1], endpoints[:, 0], color='blue', marker='o', s=60, zorder=4, label='Endpoint')

    # Formatting for traditional image coordinate space (Y-axis inverted)
    ax.set_aspect('equal')
    ax.set_title("Fragmented Skeleton Paths & Critical Nodes")
    ax.grid(True, linestyle='--', alpha=0.5)
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Uncomment if path count is low

    plt.show()
    return glyph