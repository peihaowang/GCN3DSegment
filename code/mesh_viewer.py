import threevis
import numpy as np

def display_mesh(mesh):
    vertices = mesh.get("vertices", None)
    faces = mesh.get("faces", None)
    vertex_colors = mesh.get("vertex_colors", (0.5, 0.5, 0.5))
    point_size = mesh.get("point_size", 1.0)
    shading = mesh.get("shading", "flat") # Flat, Phong, Hidden, Wireframe, None
    
    if faces is None:
        if isinstance(vertex_colors, tuple):
            vc = np.array(vertex_colors*len(vertices))
            vc = np.reshape((len(vertices), 3))
            color_attr = threevis.PointAttribute(vc)
        elif isinstance(vertex_colors, str):
            color_attr = vertex_colors
        else:
            color_attr = threevis.PointAttribute(vertex_colors)
        threevis.display_vertices(vertices, point_size=point_size, colors=color_attr)
    else:
        if isinstance(vertex_colors, tuple):
            vc = np.array(vertex_colors*len(vertices))
            vc = np.reshape(vc, (len(vertices), 3))
            color_attr = threevis.PointAttribute(vc)
        else:
            color_attr = threevis.PointAttribute(vertex_colors)
        normals = threevis.calculate_face_normals(vertices, faces)
        threevis.display_faces(
            vertices, faces, 
            normals = normals,
            colors = color_attr,
            shading = shading
        )