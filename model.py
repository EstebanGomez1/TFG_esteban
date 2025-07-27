import torch
import torch.nn as nn
import torch.nn.functional as F
from PTv3.model import PointTransformerV3

class PTv3_deteccion(nn.Module):
    def __init__(self, grid_size: int):
        super(PTv3_deteccion, self).__init__()
        self.grid_size = grid_size
        self.point_encoder = PointTransformerV3(
            in_channels=1,
            enc_depths=(1, 1, 1, 1, 1),
            enc_num_head=(1, 2, 4, 8, 16),
            enc_patch_size=(64, 64, 64, 64, 64),
            enc_channels=(32, 64, 128, 128, 256),
            dec_depths=(1, 1, 1, 1),
            dec_channels=(128, 64, 64, 64),
            dec_num_head=(4, 4, 4, 8),
            dec_patch_size=(64, 64, 64, 64),
            mlp_ratio=4,
            qkv_bias=True,
        )

        self.feature_layer = nn.Sequential(      
            nn.Linear(64, 64),
            nn.ReLU(),      
            nn.Linear(64, 32)
        )

        # 64*23*23   18432
        self.clf_head = nn.Sequential(
            nn.Linear(36864, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        ) 
        self.reg_head = nn.Sequential(
            nn.Linear(36864, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )  
        self.cyc_head_sin = nn.Sequential(
            nn.Linear(36864, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        self.cyc_head_cos = nn.Sequential(
            nn.Linear(36864, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
            #nn.AdaptiveAvgPool2d((1,1))
        )



    def forward(self, ventana: list):
        device = next(self.parameters()).device
        ventana = [f.to(device) for f in ventana]
        ventana = [f.squeeze(0) if f.ndim == 3 and f.shape[0] == 1 else f for f in ventana]
        all_points = torch.cat(ventana, dim=0)
        offset = torch.tensor([f.shape[0] for f in ventana], device=device).cumsum(0)
        points_dict = {
            "feat": all_points[:, 3:],
            "coord": all_points[:, :3],
            "offset": offset,
            "grid_size": self.grid_size,
        }
        point_features = self.point_encoder(points_dict)
        #feats_mean = torch.mean(point_features["feat"], dim=0, keepdim=True)  # [1, D]
        #embedding = self.feature_layer(feats_mean)
        #_-------_
        grid_feat = self.scatter_grid_pooling(
            point_features["coord"].detach(),
            point_features["feat"].detach(), 
            res_x=0.25,
            grid_size= 24, #6/0.25
            device=device
        )
        grid_feat = grid_feat.unsqueeze(0)  # [1, F]

        #############
        """# Visualizar y guardar
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"heatmap_{timestamp}.png"
        grid_np = grid_feat[0, 0].detach().cpu().numpy()

        plt.figure(figsize=(6, 6))
        plt.imshow(grid_np, cmap='hot', origin='lower')
        plt.colorbar(label='Número de puntos')
        plt.title("Distribución de puntos en la rejilla")

        # Asegúrate de que la carpeta exista
        folder = "grids"
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        plt.savefig(filepath)
        plt.close()

        # Extraer coordenadas XY
        points_np = all_points[:, :3].detach().cpu().numpy()
        x, y = points_np[:, 0], points_np[:, 1]

        # Crear figura 2D
        plt.figure(figsize=(6, 6))
        plt.scatter(x, y, c='blue', s=1)  # puedes cambiar color o tamaño de punto
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Nube de puntos proyectada en XY")
        plt.axis('equal')  # para que los ejes tengan la misma escala

        # Guardar imagen
        cloud_img_filename = f"cloud_{timestamp}.png"
        cloud_img_path = os.path.join(folder, cloud_img_filename)
        plt.savefig(cloud_img_path)
        plt.close()"""
        #############



        #embedding = self.feature_layer(grid_feat)
        embedding = self.conv_block(grid_feat)  
        embedding = embedding.view(1, -1)  
        #_-------_
        normalized_embedding = embedding
        logits = self.clf_head(normalized_embedding)     # [batch, 8]
        reg_out = self.reg_head(normalized_embedding)    # [batch, 6]
        sin_out = self.cyc_head_sin(normalized_embedding)
        cos_out = self.cyc_head_cos(normalized_embedding)
        cyc_out = torch.cat([sin_out, cos_out], dim=1) 
        #cyc_out = F.normalize(cyc_out, dim=1)
        return logits, reg_out, cyc_out
    
    def scatter_grid_pooling(self, coords, feats, res_x=0.25, grid_size=24, device="cpu"):
        assert coords.shape[0] == feats.shape[0], "coords y feats deben tener el mismo número de puntos"

        x = coords[:, 0]
        y = coords[:, 1]

        F = feats.shape[1]
        grid = torch.zeros((F, grid_size, grid_size), device=device)

        half_extent = (grid_size * res_x) / 2

        cx = ((x + half_extent) / res_x).long()
        cy = ((y + half_extent) / res_x).long()

        mask = (cx >= 0) & (cx < grid_size) & (cy >= 0) & (cy < grid_size)
        cx = cx[mask]
        cy = cy[mask]
        feats = feats[mask]

        indices = cx * grid_size + cy  
        grid = torch.zeros((F, grid_size * grid_size), device=device)

        grid = grid.index_add(1, indices, feats.T) 

        grid = grid.view(F, grid_size, grid_size)

        return grid