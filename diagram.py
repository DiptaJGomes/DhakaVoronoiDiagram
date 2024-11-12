import networkx as nx
from scipy.spatial import Voronoi
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

# Values of x and y
x_values = [23.78197886, 23.76956855, 23.78576776, 23.79935451, 23.86083892, 23.79967071, 23.79613673, 23.82047893,
            23.84539702, 23.70710976, 23.74198834, 23.73833144, 23.73341013, 23.72324368, 23.73389908, 23.73243831,
            23.71056218, 23.78602155, 23.75916729, 23.76842393, 23.85477676, 23.8364697, 23.47528336, 23.84790958,
            23.9894958]
y_values = [90.41684932, 90.422085, 90.34467405, 90.35308546, 90.3653244, 90.42393032, 90.41397396, 90.45268123,
            90.40249605, 90.40977218, 90.38320531, 90.39534173, 90.38366809, 90.41258346, 90.40354883, 90.42639376,
            90.43492837, 90.32978285, 90.38699582, 90.38298348, 90.50546445, 90.54570568, 90.27074127, 90.25773549,
            90.38144177]

# Plotting the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(x_values, y_values, color='blue', marker='o')
plt.title('Scatter Plot of Latitude and Longitude')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.grid(True)
plt.show()

# Points
points = list(zip(x_values, y_values))

# Compute Voronoi diagram
vor = Voronoi(points)

# Plot Voronoi diagram
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, 'bo')  # Plot points
plt.plot(vor.vertices[:, 0], vor.vertices[:, 1], 'r.')  # Plot vertices
plt.title('Voronoi Diagram with Vertices')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.grid(True)
plt.show()


# Function to calculate distance between two points
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Points
points = list(zip(x_values, y_values))

# Compute Voronoi diagram
vor = Voronoi(points)

# Plot Voronoi diagram
fig, ax = plt.subplots(figsize=(8, 6))
voronoi_plot_2d(vor, ax=ax, show_vertices=False)

# Generate short path destination points in each zone and plot them
for region_index in vor.regions:
    if -1 not in region_index and len(region_index) > 0:
        region_vertices = [vor.vertices[i] for i in region_index]
        # Choose the point closest to the centroid as the destination
        centroid = np.mean(region_vertices, axis=0)
        nearest_point = min(region_vertices, key=lambda p: distance(p, centroid))
        plt.plot(nearest_point[0], nearest_point[1], 'ro')  # Plot nearest point as red dot

plt.title('Voronoi Diagram with Short Path Destination Points')
plt.xlabel('x Values')
plt.ylabel('y Values')
plt.grid(True)
plt.show()

print("Voronoi Vertices:")
for vertex in vor.vertices:
    print(vertex)

# Print Voronoi edges with weights
print("Voronoi Edges with Weights:")
for region_index in vor.regions:
    if region_index:  # Non-empty region
        for i in range(len(region_index)):
            if i != len(region_index) - 1:
                # Calculate distance between vertices
                v1 = vor.vertices[region_index[i]]
                v2 = vor.vertices[region_index[i + 1]]
                edge_weight = distance(v1, v2)
                print(f"Edge: ({v1[0]}, {v1[1]}) - ({v2[0]}, {v2[1]}), Weight: {edge_weight}")
        # Connect the last and first vertices to close the polygon
        v1 = vor.vertices[region_index[-1]]
        v2 = vor.vertices[region_index[0]]
        edge_weight = distance(v1, v2)
        print(f"Edge: ({v1[0]}, {v1[1]}) - ({v2[0]}, {v2[1]}), Weight: {edge_weight}")

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# Convert points to numpy array
points_array = np.array(points)

# Compute Delaunay triangulation
tri = Delaunay(points_array)

# Plot Delaunay triangulation
plt.figure(figsize=(8, 6))
plt.triplot(points_array[:, 0], points_array[:, 1], tri.simplices)
plt.plot(points_array[:, 0], points_array[:, 1], 'o')
plt.title('Delaunay Triangulation')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.grid(True)
plt.show()

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

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

# Values of x and y
x_values = [23.78197886, 23.76956855, 23.78576776, 23.79935451, 23.86083892, 23.79967071, 23.79613673, 23.82047893,
            23.84539702, 23.70710976, 23.74198834, 23.73833144, 23.73341013, 23.72324368, 23.73389908, 23.73243831,
            23.71056218, 23.78602155, 23.75916729, 23.76842393, 23.85477676, 23.8364697, 23.47528336, 23.84790958,
            23.9894958]
y_values = [90.41684932, 90.422085, 90.34467405, 90.35308546, 90.3653244, 90.42393032, 90.41397396, 90.45268123,
            90.40249605, 90.40977218, 90.38320531, 90.39534173, 90.38366809, 90.41258346, 90.40354883, 90.42639376,
            90.43492837, 90.32978285, 90.38699582, 90.38298348, 90.50546445, 90.54570568, 90.27074127, 90.25773549,
            90.38144177]

# Ensure the points are in the correct format
points = np.column_stack((x_values, y_values))

# Assigning weights to each point
weights = np.random.rand(len(points)) * 5  # Random weights between 0 and 5

# Compute Voronoi diagram
vor = Voronoi(points)

# Plot Voronoi diagram with random colors for each region
plt.figure(figsize=(8, 6))
voronoi_plot_2d(vor, show_vertices=False, line_colors='orange', line_width=2)

# Assign random colors to each region
for region_index in vor.regions:
    if -1 not in region_index and len(region_index) > 0:
        region_vertices = [vor.vertices[i] for i in region_index]
        plt.fill(*zip(*region_vertices), alpha=0.4, color=np.random.rand(3, ))

plt.title('Weighted Voronai Diagram')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.grid(True)
plt.show()

import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

# Values of x and y
x_values = [23.78197886, 23.76956855, 23.78576776, 23.79935451, 23.86083892, 23.79967071, 23.79613673, 23.82047893,
            23.84539702, 23.70710976, 23.74198834, 23.73833144, 23.73341013, 23.72324368, 23.73389908, 23.73243831,
            23.71056218, 23.78602155, 23.75916729, 23.76842393, 23.85477676, 23.8364697, 23.47528336, 23.84790958,
            23.9894958]
y_values = [90.41684932, 90.422085, 90.34467405, 90.35308546, 90.3653244, 90.42393032, 90.41397396, 90.45268123,
            90.40249605, 90.40977218, 90.38320531, 90.39534173, 90.38366809, 90.41258346, 90.40354883, 90.42639376,
            90.43492837, 90.32978285, 90.38699582, 90.38298348, 90.50546445, 90.54570568, 90.27074127, 90.25773549,
            90.38144177]

# Points
points = np.column_stack((x_values, y_values))

# Compute Voronoi diagram
vor = Voronoi(points)

# Plot Voronoi diagram with random colors for each region
plt.figure(figsize=(8, 6))
voronoi_plot_2d(vor, show_vertices=False, line_colors='orange', line_width=2)

# Assign random colors to each region
for region_index in vor.regions:
    if -1 not in region_index and len(region_index) > 0:
        region_vertices = [vor.vertices[i] for i in region_index]
        plt.fill(*zip(*region_vertices), alpha=0.4, color=np.random.rand(3, ))

plt.title('Colorful Voronoi Diagram')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import Delaunay

# Values of x and y
x_values = [23.78197886, 23.76956855, 23.78576776, 23.79935451, 23.86083892, 23.79967071, 23.79613673, 23.82047893,
            23.84539702, 23.70710976, 23.74198834, 23.73833144, 23.73341013, 23.72324368, 23.73389908, 23.73243831,
            23.71056218, 23.78602155, 23.75916729, 23.76842393, 23.85477676, 23.8364697, 23.47528336, 23.84790958,
            23.9894958]
y_values = [90.41684932, 90.422085, 90.34467405, 90.35308546, 90.3653244, 90.42393032, 90.41397396, 90.45268123,
            90.40249605, 90.40977218, 90.38320531, 90.39534173, 90.38366809, 90.41258346, 90.40354883, 90.42639376,
            90.43492837, 90.32978285, 90.38699582, 90.38298348, 90.50546445, 90.54570568, 90.27074127, 90.25773549,
            90.38144177]

# Convert points to numpy array
points = np.column_stack((x_values, y_values))

# Compute Delaunay triangulation
tri = Delaunay(points)

# Perform clustering
clustering = AgglomerativeClustering(n_clusters=3)
clusters = clustering.fit_predict(points)

# Plot Delaunay triangulation with clusters
plt.figure(figsize=(8, 6))
plt.triplot(points[:, 0], points[:, 1], tri.simplices, c='grey')
plt.plot(points[:, 0], points[:, 1], 'o', c=clusters)
plt.title('Delaunay Triangulation with Clusters')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.grid(True)
plt.show()
