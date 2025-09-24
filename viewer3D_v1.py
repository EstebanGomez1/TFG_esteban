import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import os

class Visualizer3D:
    def __init__(self, save_path="visualizacion3D", elev=70, azim=310):
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.elev = elev
        self.azim = azim

    def draw_bo1(self, ax, center, size, yaw, color='r', alpha=0.3):
        """
        Dibuja una caja 3D con centro, tamaño y rotación.
        center: [x, y, z]
        size: [w, h, l]
        yaw: ángulo de rotación en radianes
        """
        w, l, h = size
        x, y, z = center
        z=z+2
        
        # Ejes: X (largo), Y (ancho), Z (altura)
        """x_corners = [l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]  # largo
        y_corners = [w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]  # ancho
        z_corners = [0,    0,    0,    0,   -h,   -h,   -h,   -h]    # altura desde base hacia abajo"""

        """x_corners = [l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]  # largo
        y_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
        z_corners = [l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2]  # ya está bien centrado"""

        """l, w, h = size
        x_corners = [l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]  # largo
        y_corners = [w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]  # ancho
        z_corners = [0,    0,    0,    0,   -h,   -h,   -h,   -h]    # base al suelo"""

        # largo: eje X, ancho: eje Y, alto: eje Z
        x_corners = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
        y_corners = [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]
        z_corners = [ h/2,  h/2,  h/2,  h/2, -h/2, -h/2, -h/2, -h/2]  # centrado en el centro geométrico


        # Rotar
        corners = np.vstack([x_corners, y_corners, z_corners])  # (3,8)
        rot_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0,            0,           1]
        ])


        corners_rot = rot_matrix @ corners
        corners_rot = corners_rot + np.array([[x], [y], [z]])
        corners_rot = corners_rot.T


        edges = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [1, 2, 6, 5],
            [0, 3, 7, 4]
        ]

        faces = [[corners_rot[i] for i in face] for face in edges]
        ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors='k', alpha=alpha))


    def draw_box(self, ax, center, size, yaw, color='r', alpha=0.3):
        """
        Dibuja una caja 3D con centro, tamaño y rotación (en radianes).
        
        center: (x, y, z)
        size: (w, h, l) → ancho (Y), alto (Z), largo (X)
        yaw: rotación alrededor del eje Z (en radianes)
        """
        #w, h, l = size
        #h, w, l = size
        #l, w, h = size
        w, l, h = size
        x, y, z = center
        z =z+1
        #y = -y
        #z = z*0.5
        #z = z-h/2
        # Define corners en el sistema local
        """x_corners = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
        y_corners = [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]
        z_corners = [ h/2,  h/2,  h/2,  h/2, -h/2, -h/2, -h/2, -h/2]"""

        """x_corners = [l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]  # largo
        y_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
        z_corners = [l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2]  # ya está bien centrado"""

        x_corners = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
        y_corners = [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]
        z_corners = [ h/2,  h/2,  h/2,  h/2, -h/2, -h/2, -h/2, -h/2]


        corners = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)

        # Rotación en plano XY (Z hacia arriba)
        R = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0,            0,           1]
        ])
        corners_rot = R @ corners

        # Traslación al centro global
        corners_rot[0, :] += x
        corners_rot[1, :] += y
        corners_rot[2, :] += z
        corners_rot = corners_rot.T  # (8, 3)

        # Definir caras (índices de las 8 esquinas)
        faces_idx = [
            [0, 1, 2, 3],  # cara superior
            [4, 5, 6, 7],  # cara inferior
            [0, 1, 5, 4],  # frontal
            [2, 3, 7, 6],  # trasera
            [1, 2, 6, 5],  # derecha
            [0, 3, 7, 4]   # izquierda
        ]
        faces = [[corners_rot[i] for i in face] for face in faces_idx]

        # Añadir la caja a la figura
        ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors='k', alpha=alpha))


    def plot(self, points: torch.Tensor, pred_box: torch.Tensor, target_box: torch.Tensor, filename="example.png"):
        fig = plt.figure(figsize=(10, 10))  # Imagen más grande
        ax = fig.add_subplot(111, projection='3d')

        pts = points[:, :3].cpu().numpy()
        #pts[:, 2] = -pts[:, 2]
        # Mostrar puntos con mayor tamaño y opacidad
        pts[:, 2] += 1
        #pts[:, 1] = -pts[:, 1] 
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='blue', s=10, alpha=0.8)

        #print("Yaw (rad):", target_box[6].item(), "Yaw (deg):", np.degrees(target_box[6].item()))

        # Dibujar predicción (rojo)
        #self.draw_box(ax, pred_box[:3].cpu().numpy(), pred_box[3:6].cpu().numpy(), pred_box[6].item(), color='r', alpha=0.4)

        # Dibujar ground truth (verde)
        #self.draw_box(ax, target_box[:3].cpu().numpy(), target_box[3:6].cpu().numpy(), target_box[6].item(), color='g', alpha=0.4)

        # Etiquetas de ejes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Vista desde ángulo personalizado (giro en sentido horario)
        ax.view_init(elev=self.elev, azim=self.azim - 45)

        # Proporción del cubo
        ax.set_box_aspect([1, 1, 1])

        # --- Ejes en los bordes ---
        ax.xaxis._axinfo['tick']['inward_factor'] = 0
        ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
        ax.yaxis._axinfo['tick']['inward_factor'] = 0
        ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
        ax.zaxis._axinfo['tick']['inward_factor'] = 0
        ax.zaxis._axinfo['tick']['outward_factor'] = 0.4

        # Zoom centrado en los puntos
        x_min, y_min, z_min = pts.min(axis=0)
        x_max, y_max, z_max = pts.max(axis=0)

        # Calcular centro y tamaño máximo del cubo (en todos los ejes)
        x_mid = (x_max + x_min) / 2
        y_mid = (y_max + y_min) / 2
        z_mid = (z_max + z_min) / 2
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.6  # 🔍 más zoom

        # Establecer los mismos límites en cada eje centrados
        ax.set_xlim(x_mid - max_range, x_mid + max_range)
        ax.set_ylim(y_mid - max_range, y_mid + max_range)
        ax.set_zlim(z_mid - max_range, z_mid + max_range)


        # Guardar
        out_path = os.path.join(self.save_path, filename)
        #ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        #print(f"Imagen guardada en {out_path}")