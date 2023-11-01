# --------------------------------------------------------
# Based from SimMIM codebase
# https://github.com/microsoft/SimMIM
# --------------------------------------------------------


from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .swin_transformer import SwinTransformer
from .vision_transformer import VisionTransformer

from scipy import interpolate
import numpy as np

class SwinTransformerForSimMIM(SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(self, x, mask):
        x = self.patch_embed(x)

        assert mask is not None
        B, L, _ = x.shape

        mask_tokens = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        x = x * (1. - w) + mask_tokens * w

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers[:-1]:
            x = layer(x)
        r = self.layers[-1](x)
        r = self.norm(r)
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)

        r = r.transpose(1, 2)
        B, C, L = r.shape
        H = W = int(L ** 0.5)
        r = r.reshape(B, C, H, W)
        return r, x

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}


class VisionTransformerForSimMIM(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self._trunc_normal_(self.mask_token, std=.02)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward(self, x, mask):
        x = self.patch_embed(x)

        assert mask is not None
        B, L, _ = x.shape

        mask_token = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        x = self.norm(x)

        x = x[:, 1:]
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x


class SimMIM(nn.Module):
    def __init__(self, encoder, encoder_stride, teacher):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride
        self.teacher = teacher

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride ** 2 * self.in_chans, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )

        self.projector = nn.Linear(self.encoder.num_features, self.encoder.num_features)
        self.cos = nn.CosineSimilarity(dim=1)


    def forward(self, x, mask):
        r, zs = self.encoder(x, mask)
        zs = self.projector(zs)
        zt = self.teacher(F.interpolate(x, (224, 224), mode='bilinear', align_corners=True))
        x_rec = self.decoder(r)

        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans

        loss += -(self.cos(zs, zt.detach()).mean())*self.teacher.alpha
        return loss

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


def build_simmim(config, logger):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        encoder = SwinTransformerForSimMIM(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=0,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        encoder_stride = 32
    elif model_type == 'vit':
        encoder = VisionTransformerForSimMIM(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.VIT.PATCH_SIZE,
            in_chans=config.MODEL.VIT.IN_CHANS,
            num_classes=0,
            embed_dim=config.MODEL.VIT.EMBED_DIM,
            depth=config.MODEL.VIT.DEPTH,
            num_heads=config.MODEL.VIT.NUM_HEADS,
            mlp_ratio=config.MODEL.VIT.MLP_RATIO,
            qkv_bias=config.MODEL.VIT.QKV_BIAS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=config.MODEL.VIT.INIT_VALUES,
            use_abs_pos_emb=config.MODEL.VIT.USE_APE,
            use_rel_pos_bias=config.MODEL.VIT.USE_RPB,
            use_shared_rel_pos_bias=config.MODEL.VIT.USE_SHARED_RPB,
            use_mean_pooling=config.MODEL.VIT.USE_MEAN_POOLING)
        encoder_stride = 16
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")
    teacher = SwinTeacher(
        img_size=224,
        patch_size=config.MODEL.SWIN.PATCH_SIZE,
        in_chans=config.MODEL.SWIN.IN_CHANS,
        num_classes=config.MODEL.NUM_CLASSES,
        embed_dim=config.MODEL.SWIN.EMBED_DIM,
        depths=config.MODEL.SWIN.DEPTHS,
        num_heads=config.MODEL.SWIN.NUM_HEADS,
        window_size=7,
        mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
        qkv_bias=config.MODEL.SWIN.QKV_BIAS,
        qk_scale=config.MODEL.SWIN.QK_SCALE,
        drop_rate=config.MODEL.DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        ape=config.MODEL.SWIN.APE,
        patch_norm=config.MODEL.SWIN.PATCH_NORM,
        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        alpha=config.ALPHA)
    load_pretrained(config, teacher, logger)
    model = SimMIM(encoder=encoder, encoder_stride=encoder_stride, teacher=teacher)

    return model

class SwinTeacher(SwinTransformer):
    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)
        self.head = None
        self.alpha = alpha

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers[:-1]:
            x = layer(x)

        # x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x

def load_pretrained(config, model, logger):
    logger.info(f">>>>>>>>>> Fine-tuned from {config.PRETRAINED} ..........")
    checkpoint = torch.load(config.PRETRAINED, map_location='cpu')
    checkpoint_model = checkpoint['model']
    
    if any([True if 'encoder.' in k else False for k in checkpoint_model.keys()]):
        checkpoint_model = {k.replace('encoder.', ''): v for k, v in checkpoint_model.items() if k.startswith('encoder.')}
        logger.info('Detect pre-trained model, remove [encoder.] prefix.')
    else:
        logger.info('Detect non-pre-trained model, pass without doing anything.')

    if config.MODEL.TYPE == 'swin':
        logger.info(f">>>>>>>>>> Remapping pre-trained keys for SWIN ..........")
        checkpoint = remap_pretrained_keys_swin(model, checkpoint_model, logger)
    else:
        raise NotImplementedError
    # MATIAS: If using ImageNet (3 channel) pretrained weights for Sentinel-2 (12 band) data
    if model.patch_embed.proj.weight.shape != checkpoint_model['patch_embed.proj.weight'].shape:
        temp = model.patch_embed.proj.weight.data.cpu()
        if checkpoint_model['patch_embed.proj.weight'].shape[1]==1:
            # greyscale pretrained model
            temp = checkpoint_model['patch_embed.proj.weight'].repeat(1, temp.shape[1],1,1)
        elif checkpoint_model['patch_embed.proj.weight'].shape[1] == 12 and temp.shape[1] == 3: 
            # For 12 band pretrained, the order is CGBR...
            temp[:,:,:,:] = checkpoint_model['patch_embed.proj.weight'][:,[3,2,1],:,:]
        elif checkpoint_model['patch_embed.proj.weight'].shape[1] == 8:
            #SpaceNet superres pretrain
            min_channels = min(temp.shape[1],  checkpoint_model['patch_embed.proj.weight'].shape[1])
            temp[:,:min_channels,:,:] = checkpoint_model['patch_embed.proj.weight'][:,:min_channels,:,:]
        else:
            temp[:,[3,2,1],:,:] = checkpoint_model['patch_embed.proj.weight']
        checkpoint_model['patch_embed.proj.weight'] = temp
    if 'finetune' in config.PRETRAINED:
        logger.info(f">>>>>>>>>> Resizing head from supervised model {config.PRETRAINED} ..........")
        # temp_headw = model.head.weight.data.cpu()
        # temp_headb = model.head.bias.data.cpu()
        # checkpoint_model['head.weight'] = temp_headw
        # checkpoint_model['head.bias'] = temp_headb
        checkpoint_model = {k:v for k, v in checkpoint_model.items() if 'head' not in k}
        # checkpoint_model = {k:v for k, v in checkpoint_model.items() if 'layers.3' not in k}
        # checkpoint_model = {k:v for k, v in checkpoint_model.items() if 'layers.2.blocks.5' not in k}
        # checkpoint_model = {k:v for k, v in checkpoint_model.items() if 'layers.2.blocks.4' not in k}
    msg = model.load_state_dict(checkpoint_model, strict=False)
    if 'finetune' in config.PRETRAINED:
        for param in model.parameters():
            param.requires_grad = False
    #     for param in model.head.parameters():
    #         param.requires_grad = True
    #     for n, param in model.named_parameters():
    #         layers = ['layers.0.blocks.0.attn.relative_position_index', 'layers.0.blocks.1.attn_mask', 'layers.0.blocks.1.attn.relative_position_index', 'layers.1.blocks.0.attn.relative_position_index', 
    #             'layers.1.blocks.1.attn_mask', 'layers.1.blocks.1.attn.relative_position_index', 'layers.2.blocks.0.attn.relative_position_index', 'layers.2.blocks.1.attn_mask', 'layers.2.blocks.1.attn.relative_position_index',
    #              'layers.2.blocks.2.attn.relative_position_index', 'layers.2.blocks.3.attn_mask', 'layers.2.blocks.3.attn.relative_position_index', 'layers.2.blocks.4.attn.relative_position_index', 'layers.2.blocks.5.attn_mask']
    #         if 'layer3' in n or n in layers:
    #             param.requires_grad = True
    logger.info(msg)
    
    del checkpoint
    torch.cuda.empty_cache()
    logger.info(f">>>>>>>>>> loaded successfully '{config.PRETRAINED}'")

def remap_pretrained_keys_swin(model, checkpoint_model, logger):
    state_dict = model.state_dict()
    
    # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_bias_table" in key:
            relative_position_bias_table_pretrained = checkpoint_model[key]
            relative_position_bias_table_current = state_dict[key]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            if nH1 != nH2:
                logger.info(f"Error in loading {key}, passing......")
            else:
                if L1 != L2:
                    logger.info(f"{key}: Interpolate relative_position_bias_table using geo.")
                    src_size = int(L1 ** 0.5)
                    dst_size = int(L2 ** 0.5)

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r ** n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    logger.info("Original positions = %s" % str(x))
                    logger.info("Target positions = %s" % str(dx))

                    all_rel_pos_bias = []

                    for i in range(nH1):
                        z = relative_position_bias_table_pretrained[:, i].view(src_size, src_size).float().numpy()
                        f_cubic = interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(torch.Tensor(f_cubic(dx, dy)).contiguous().view(-1, 1).to(
                            relative_position_bias_table_pretrained.device))

                    new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                    checkpoint_model[key] = new_rel_pos_bias

    # delete relative_position_index since we always re-init it
    # relative_position_index_keys = [k for k in checkpoint_model.keys() if "relative_position_index" in k]
    # for k in relative_position_index_keys:
        # del checkpoint_model[k]

    # delete relative_coords_table since we always re-init it
    # relative_coords_table_keys = [k for k in checkpoint_model.keys() if "relative_coords_table" in k]
    # for k in relative_coords_table_keys:
        # del checkpoint_model[k]

    # delete attn_mask since we always re-init it
    # attn_mask_keys = [k for k in checkpoint_model.keys() if "attn_mask" in k]
    # for k in attn_mask_keys:
        # del checkpoint_model[k]

    return checkpoint_model