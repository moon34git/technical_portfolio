import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MAEEncoder(nn.Module):
    def __init__(self, mae_model):
        """
        Explicitly reference the MAE's encoder submodules:
        - patch_embed
        - cls_token
        - pos_embed
        - blocks (ModuleList)
        - norm
        """
        super().__init__()
        # These attributes exist in MaskedAutoencoderViT
        self.patch_embed = mae_model.patch_embed
        self.cls_token = mae_model.cls_token
        self.pos_embed = mae_model.pos_embed
        self.blocks = mae_model.blocks
        self.norm = mae_model.norm

    def forward(self, x):
        """
        x: (B, 3, H, W)
        Return shape: (B, num_patches+1, embed_dim),
        because we add a [CLS] token and pass it through the blocks + final norm.
        """
        # 1) Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # 2) Add position embedding (excluding the [CLS] token index 0)
        x = x + self.pos_embed[:, 1:, :]

        # 3) Append CLS token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_token = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # (B, num_patches+1, embed_dim)

        # 4) Pass through transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # 5) Final norm
        x = self.norm(x)
        return x


import torch
import torch.nn as nn

class FineTunedMAE(nn.Module):
    def __init__(self, mae_model, num_classes=1, freeze=True, global_pool=True, norm_layer=nn.LayerNorm):
        """
        Build a classification model using MAE encoder weights.
        
        Args:
            mae_model: Pretrained MAE model (using encoder only)
            num_classes: Number of classes to classify
            global_pool: If True, use average of patch tokens instead of [CLS] token
            norm_layer: Normalization layer (default: nn.LayerNorm)
        """
        super().__init__()
        self.global_pool = global_pool
        self.freeze = freeze

        self.patch_embed = mae_model.patch_embed
        self.cls_token = mae_model.cls_token
        self.pos_embed = mae_model.pos_embed
        self.blocks = mae_model.blocks
        self.norm = mae_model.norm
        
        self.pos_drop = mae_model.pos_drop if hasattr(mae_model, "pos_drop") else nn.Identity()
        
        embed_dim = self.cls_token.shape[-1]

        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
        
        self.head = nn.Linear(embed_dim, num_classes)
        
        if self.freeze:
            self.freeze_encoder()
        
    def freeze_encoder(self):
        """Freeze MAE encoder parameters."""
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        self.cls_token.requires_grad = False
        self.pos_embed.requires_grad = False
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False
        for param in self.norm.parameters():
            param.requires_grad = False
        if not isinstance(self.pos_drop, nn.Identity):
            for param in self.pos_drop.parameters():
                param.requires_grad = False
        if self.global_pool and hasattr(self, 'fc_norm'):
            for param in self.fc_norm.parameters():
                param.requires_grad = False

    def forward_features(self, x):
        """
        ViT-style forward_features:
        - Patch embedding, CLS token addition, positional embedding, dropout
        - Pass through Transformer blocks, then global pooling (or [CLS] token extraction)
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            x = self.fc_norm(x)
        else:
            x = self.norm(x)
            x = x[:, 0]
        return x

    def forward(self, x):
        """
        Final forward: Extract feature vector via forward_features(),
        then pass through head to produce classification logits
        """
        features = self.forward_features(x)
        logits = self.head(features)
        return logits


import torch
import torch.nn as nn
import math


class FineTunedMAE_Shallow(nn.Module):
    """
    Frozen MAE Encoder + Shallow CNN Adapter + Linear Classifier
    """
    def __init__(
        self,
        mae_model,
        num_classes: int = 1,
        freeze: bool = True,
        global_pool: bool = True,
        norm_layer=nn.LayerNorm,
        # cnn_channels: int = 384,        
        cnn_channels: int = 768,
    ):
        super().__init__()
        self.global_pool = global_pool
        self.freeze = freeze


        self.patch_embed = mae_model.patch_embed
        self.cls_token   = mae_model.cls_token
        self.pos_embed   = mae_model.pos_embed
        self.blocks      = mae_model.blocks
        self.norm        = mae_model.norm

        self.pos_drop    = getattr(mae_model, "pos_drop", nn.Identity())

        embed_dim = self.cls_token.shape[-1]

        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)

        ## Adapter
        self.adapter = nn.Sequential(
            nn.Conv2d(embed_dim, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Linear(cnn_channels, num_classes)

        if self.freeze:
            self.freeze_encoder()

    def freeze_encoder(self):
        for m in [self.patch_embed, self.blocks, self.norm]:
            for p in m.parameters():
                p.requires_grad = False
        self.cls_token.requires_grad = False
        self.pos_embed.requires_grad = False
        if not isinstance(self.pos_drop, nn.Identity):
            for p in self.pos_drop.parameters():
                p.requires_grad = False
        if self.global_pool and hasattr(self, "fc_norm"):
            for p in self.fc_norm.parameters():
                p.requires_grad = False

    def get_features(self, x: torch.Tensor):
        B = x.size(0)

        x = self.patch_embed(x)                      # (B, N, D)
        cls_tok = self.cls_token.expand(B, -1, -1)   # (B, 1, D)
        x = torch.cat((cls_tok, x), dim=1) + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            feat = self.fc_norm(x[:, 1:, :].mean(dim=1))
            feat_map = feat.unsqueeze(-1).unsqueeze(-1)
        else:
            N = x.size(1) - 1
            h = w = int(math.sqrt(N))          # 14×14 for 224/16
            patch_tokens = x[:, 1:, :].transpose(1, 2).reshape(B, -1, h, w)
            feat_map = patch_tokens  

        out = self.adapter(feat_map)
        out = out.mean(dim=[2, 3])    

        return feat, out
    

    def forward(self, x: torch.Tensor):
        """
        x: (B, 3, H, W)
        """
        B = x.size(0)

        # --- MAE encoder (frozen) -------------------------
        x = self.patch_embed(x)                      # (B, N, D)
        cls_tok = self.cls_token.expand(B, -1, -1)   # (B, 1, D)
        x = torch.cat((cls_tok, x), dim=1) + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            feat = self.fc_norm(x[:, 1:, :].mean(dim=1))
            feat_map = feat.unsqueeze(-1).unsqueeze(-1)
        else:
            N = x.size(1) - 1
            h = w = int(math.sqrt(N))
            patch_tokens = x[:, 1:, :].transpose(1, 2).reshape(B, -1, h, w)
            feat_map = patch_tokens

        out = self.adapter(feat_map)
        out = out.mean(dim=[2, 3])

        logits = self.head(out)
        return logits

    def extract_representation(self, x: torch.Tensor):
        """
        Extract representation vector (CNN adapter output) for FedProto.
        Output shape: [B, cnn_channels]
        """
        B = x.size(0)

        # --- MAE encoder (frozen) -------------------------
        x = self.patch_embed(x)                      # (B, N, D)
        cls_tok = self.cls_token.expand(B, -1, -1)   # (B, 1, D)
        x = torch.cat((cls_tok, x), dim=1) + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            feat = self.fc_norm(x[:, 1:, :].mean(dim=1))
            feat_map = feat.unsqueeze(-1).unsqueeze(-1)
        else:
            N = x.size(1) - 1
            h = w = int(math.sqrt(N))
            patch_tokens = x[:, 1:, :].transpose(1, 2).reshape(B, -1, h, w)
            feat_map = patch_tokens

        out = self.adapter(feat_map)
        out = out.mean(dim=[2, 3])
        return out
