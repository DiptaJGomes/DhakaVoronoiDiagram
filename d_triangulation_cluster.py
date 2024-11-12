import numpy as np
from scipy.spatial import Delaunay

# Coordinates
x_values = [23.78197886, 23.76956855, 23.78576776, 23.79935451, 23.86083892, 23.79967071, 23.79613673, 23.82047893,
            23.84539702, 23.70710976, 23.74198834, 23.73833144, 23.73341013, 23.72324368, 23.73389908, 23.73243831,
            23.71056218, 23.78602155, 23.75916729, 23.76842393, 23.85477676, 23.8364697, 23.47528336, 23.84790958,
            23.9894958]
y_values = [90.41684932, 90.422085, 90.34467405, 90.35308546, 90.3653244, 90.42393032, 90.41397396, 90.45268123,
            90.40249605, 90.40977218, 90.38320531, 90.39534173, 90.38366809, 90.41258346, 90.40354883, 90.42639376,
            90.43492837, 90.32978285, 90.38699582, 90.38298348, 90.50546445, 90.54570568, 90.27074127, 90.25773549,
            90.38144177]

# Convert to numpy array
points = np.column_stack((x_values, y_values))

# Construct Delaunay triangulation
tri = Delaunay(points)

# Query point
query_point = np.array([23.8, 90.4])  # Example query point

# Find simplex containing the query point
simplex_index = tri.find_simplex(query_point)

# Get the vertices of the simplex containing the query point
simplex_vertices = tri.simplices[simplex_index]

# Search nearby triangles
nearby_triangles = set()
for vertex in simplex_vertices:
    for neighbor in tri.neighbors[vertex]:
        if neighbor != -1:
            nearby_triangles.add(neighbor)

# Compute distances to points in nearby triangles
nearest_neighbor = None
min_distance = float('inf')
for triangle_index in nearby_triangles:
    for vertex_index in tri.simplices[triangle_index]:
        distance = np.linalg.norm(points[vertex_index] - query_point)
        if distance < min_distance:
            min_distance = distance
            nearest_neighbor = points[vertex_index]

print("Query Point:", query_point)
print("Nearest Neighbor:", nearest_neighbor)
print("Distance:", min_distance)
