import threevis
import numpy as np

def display_mesh(mesh):
    vertices = mesh.get("vertices", None)
    faces = mesh.get("faces", None)
    vertex_colors = mesh.get("vertex_colors", None)
    face_colors = mesh.get("face_colors", None)
    point_size = mesh.get("point_size", 1.0)
    shading = mesh.get("shading", "flat") # Flat, Phong, Hidden, Wireframe, None
    
    if faces is None:
        # Validate vertex color, since face colors are not used on vertices
        if vertex_colors is None:
            vertex_colors = (0.5, 0.5, 0.5)

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
        # Validate face color, since face colors turn out to be the default choice
        if face_colors is None:
            face_colors = (0.5, 0.5, 0.5)

        if isinstance(vertex_colors, tuple):
            vc = np.array(vertex_colors*len(vertices))
            vc = np.reshape(vc, (len(vertices), 3))
            color_attr = threevis.PointAttribute(vc)
        elif vertex_colors is not None:
            color_attr = threevis.PointAttribute(vertex_colors)
        elif isinstance(face_colors, tuple):
            fc = np.array(face_colors*len(faces))
            fc = np.reshape(fc, (len(faces), 3))
            color_attr = threevis.FaceAttribute(fc)
        else:
            color_attr = threevis.FaceAttribute(face_colors)

        normals = threevis.calculate_face_normals(vertices, faces)
        threevis.display_faces(
            vertices, faces, 
            normals = normals,
            colors = color_attr,
            shading = shading
        )