import numpy as np
import os

# Replace these paths with your actual save_dir and selected vertices
save_dir = './experiments/hammer/decoder/'
selected_vertices = [1149]  # Replace with your actual vertices

# Construct the filename based on the selected vertices
select_vertices_str = '_'.join(['v%d' % idx for idx in selected_vertices])
filename = f'vertex_probability_{select_vertices_str}.npy'
filepath = os.path.join(save_dir, filename)

# Load the vertex probabilities
vertex_probabilities = np.load(filepath)

# Check if all probabilities are zero
all_zeros = np.all(vertex_probabilities == 0)
print(f'Are all vertex probabilities zero? {all_zeros}')

# Print some statistics
print(f'Min probability: {vertex_probabilities.min()}')
print(f'Max probability: {vertex_probabilities.max()}')
print(f'Mean probability: {vertex_probabilities.mean()}')
