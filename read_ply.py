#!/usr/bin/env python3
import numpy as np
from plyfile import PlyData, PlyElement

def read_custom_ply(ply_path):
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']
    
    # Access .data for structured array
    vertex_data = vertices.data
    
    # Core data
    points = np.stack([vertex_data['x'], vertex_data['y'], vertex_data['z']], axis=-1)
    colors = np.stack([vertex_data['red'], vertex_data['green'], vertex_data['blue']], axis=-1).astype(np.float32) / 255.0
    density = vertex_data['density']
    adjacency_offset = vertex_data['adjacency_offset']
    
    # SGGX matrix (6 symmetric values)
    sggx = np.stack([
        vertex_data['sxx'], vertex_data['sxy'], vertex_data['sxz'],
        vertex_data['syy'], vertex_data['syz'], vertex_data['szz']
    ], axis=0).T  # Shape: (N, 6)
    
    # Extract SH coefficients dynamically
    sh_coeffs = {}
    for field in vertex_data.dtype.names:
        if field.startswith('color_sh_'):
            idx = int(field.split('_')[-1])
            sh_coeffs[f'sh_{idx}'] = vertex_data[field]
    
    # Adjacency data
    adjacency = plydata['adjacency'].data['adjacency']
    
    print(f"Loaded {len(points)} vertices")
    print(f"SGGX shape: {sggx.shape}")
    print(f"SH coeffs: {list(sh_coeffs.keys())}")
    print(f"Density range: [{density.min():.3f}, {density.max():.3f}]")
    
    return {
        'points': points,
        'colors': colors,
        'density': density,
        'sggx': sggx,
        'adjacency_offset': adjacency_offset,
        'adjacency': adjacency,
        'sh_coeffs': sh_coeffs
    }

# Usage
if __name__ == "__main__":
    ply_path = "output/garden@84b609af/scene.ply"
    data = read_custom_ply(ply_path)
    
    print("\nFirst vertex:")
    print(f"Position: {data['points'][0]}")
    print(f"Color: {data['colors'][0]}")
    print(f"Density: {data['density'][0]:.3f}")
    print(f"SGGX: {data['sggx'][99999]}")
