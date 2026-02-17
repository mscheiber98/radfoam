from plyfile import PlyData
import numpy as np
plydata = PlyData.read('output/garden@23af917e/scene.ply')  # Handles little_endian automatically

# Access vertex data (typical element name)
vertices = plydata['vertex'].data
id_max = np.argmax(vertices['sggx_1'])  # Returns index of maximum valueprint((vertices[9999]['sggx_1']))
print((vertices[id_max]['sggx_1']))
print((vertices[id_max]['sggx_2']))
print((vertices[id_max]['sggx_3']))
print((vertices[id_max]['sggx_4']))
print((vertices[id_max]['sggx_5']))
print((vertices[id_max]['sggx_6']))
print((vertices[id_max]['sggx_7']))
print((vertices[id_max]['sggx_8']))
print((vertices[id_max]['sggx_9']))
print((vertices[id_max]['density']))