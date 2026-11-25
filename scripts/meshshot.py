"""Quick shaded pictures of a mesh from 4 sides (no GPU/browser needed). usage: meshshot.py <mesh> <out.png>"""
import sys
import numpy as np, trimesh, pymeshlab
from PIL import Image, ImageDraw
m = trimesh.load(sys.argv[1], force="mesh")
ms = pymeshlab.MeshSet(); ms.add_mesh(pymeshlab.Mesh(m.vertices, m.faces))
ms.meshing_decimation_quadric_edge_collapse(targetfacenum=25000, preservetopology=False)
mm = ms.current_mesh(); V, F = mm.vertex_matrix(), mm.face_matrix()
tiles = []
for yaw in (20, 110, 200, 290):
    a = np.radians(yaw); e = np.radians(20)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    P = V @ (Rx @ Ry).T
    s = 360 / (P[:, :2].max(0) - P[:, :2].min(0)).max()
    xy = (P[:, :2] - P[:, :2].min(0)) * s + 20
    xy[:, 1] = 400 - xy[:, 1]
    tri = P[F]; n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    img = Image.new("RGB", (400, 420), (30, 32, 38)); d = ImageDraw.Draw(img)
    for i in np.argsort(tri[:, :, 2].mean(1)):
        sh = int(60 + 180 * abs(n[i] @ np.array([0.3, 0.6, 0.74])))
        d.polygon([tuple(xy[j]) for j in F[i]], fill=(sh, sh, sh))
    tiles.append(img)
out = Image.new("RGB", (1600, 420)); [out.paste(t, (i * 400, 0)) for i, t in enumerate(tiles)]; out.save(sys.argv[2])
