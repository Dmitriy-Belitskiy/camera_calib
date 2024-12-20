import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Function to read .cam file and extract translation vector and rotation matrix
def read_cam_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Extract translation vector and rotation matrix from the first line
        first_line = list(map(float, lines[0].split()))
        translation_vector = np.array(first_line[:3])
        rotation_matrix = np.array(first_line[3:]).reshape(3, 3)
    return translation_vector, rotation_matrix

# Get all .cam files in the current folder
cam_files = [f for f in os.listdir('.') if f.endswith('.cam')]

# Separate L and R files
l_files = sorted([f for f in cam_files if '_L' in f])
r_files = sorted([f for f in cam_files if '_R' in f])

# Arrays to store rotation vectors
rotation_vectors_L = []
rotation_vectors_R = []

# Process L files
for l_file in l_files:
    _, rotation_matrix = read_cam_file(l_file)
    # Convert rotation matrix to rotation vector
    rotation_vector = R.from_matrix(rotation_matrix).as_rotvec()
    rotation_vectors_L.append(rotation_vector)

# Process R files
for r_file in r_files:
    _, rotation_matrix = read_cam_file(r_file)
    # Convert rotation matrix to rotation vector
    rotation_vector = R.from_matrix(rotation_matrix).as_rotvec()
    rotation_vectors_R.append(rotation_vector)

# Convert to numpy arrays
rotation_vectors_L = np.array(rotation_vectors_L)
rotation_vectors_R = np.array(rotation_vectors_R)

# Plot the rotation vectors in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot L vectors
ax.quiver(
    np.zeros(len(rotation_vectors_L)), np.zeros(len(rotation_vectors_L)), np.zeros(len(rotation_vectors_L)),
    rotation_vectors_L[:, 0], rotation_vectors_L[:, 1], rotation_vectors_L[:, 2],
    color='blue', label='L Images'
)

# Plot R vectors
ax.quiver(
    np.zeros(len(rotation_vectors_R)), np.zeros(len(rotation_vectors_R)), np.zeros(len(rotation_vectors_R)),
    rotation_vectors_R[:, 0], rotation_vectors_R[:, 1], rotation_vectors_R[:, 2],
    color='red', label='R Images'
)

# Add labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_title('Rotation Vectors for L and R Images')

# Show the plot
plt.show()
