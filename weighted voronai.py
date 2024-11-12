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
