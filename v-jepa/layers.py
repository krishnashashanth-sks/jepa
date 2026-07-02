import torch
import torch.nn as nn
import numpy as np

class PatchEmbedding(nn.Module):
  def __init__(self,patch_size,in_channels,embed_dim,T=16,H=224,W=224):
    super().__init__()
    self.patch_size=patch_size
    self.in_channels=in_channels
    self.embed_dim=embed_dim
    if isinstance(patch_size,int):
      self.patch_size=(patch_size,patch_size,patch_size)
    elif len(patch_size)!=3:
      raise ValueError("patch_size must be an int or a tuple/list of 3 dimensions (depth, height, width)")
    self.num_patches_t=T//self.patch_size[0]
    self.num_patches_h=H//self.patch_size[1]
    self.num_patches_w=W//self.patch_size[2]
    if self.num_patches_t==0 or self.num_patches_h==0 or self.num_patches_w==0:
      raise  ValueError("Patch size is larger than video dimensions in one or more axes.")
    self.proj=nn.Conv3d(
        in_channels,
        embed_dim,
        kernel_size=self.patch_size,
        stride=self.patch_size
    )
    self.num_patches=self.num_patches_t,self.num_patches_h,self.num_patches_w
  def forward(self,x):
    B=x.shape(0)
    x=self.proj(x)
    x=x.flatten(2)
    x=x.tranpose(1,2)
    return x
    
class SpatioTemporalPositionalEncoding(nn.Module):
  def __init__(self,embed_dim,num_patches_t,num_patches_h,num_patches_w,dropout=0.0):
    super().__init__()
    self.embed_dim=embed_dim
    self.num_patches_t=num_patches_t
    self.num_patches_h=num_patches_h
    self.num_patche_w=num_patches_w
    self.pos_embed_t=nn.Parameter(torch.zeros(1,num_patches_t,embed_dim))
    self.pos_embed_h=nn.Parameter(torch.zeros(1,num_patches_h,embed_dim))
    self.pos_embed_w=nn.Parameter(torch.zeros(1,num_patches_w,embed_dim))
    self.dropout=nn.Dropout(dropout)
    self._init_weights
  def _init_weights(self):
    nn.init.trunc_normal_(self.pos_embed_t,std=0.02)
    nn.init.trunc_normal_(self.pos_embed_h,std=0.02)
    nn.init.trunc_normal_(self.pos_embed_w,std=0.02)
  def forward(self,x):
    B,N,C=x.shape
    pos_embed=self.pos_embed_t.unsqueeze(2).unsqueeze(3)+ \
    self.pos_embed_h.unsqueeze(1).unsqueeze(3)+\
    self.pos_embed_w.unsqueeze(1).unsqueeze(2)
    pos_embed=pos_embed.view(1,self.num_patches_t*self.num_patches_h*self.num_patches_w,self.embed_dim)
    return self.dropout(x+pos_embed)

class TransformerBlock(nn.Module):
  def __init__(self,embed_dim,num_heads,mlp_ratio=4.,qkv_bias=False,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.):
    super().__init__()
    self.norm1=nn.LayerNorm(embed_dim)
    self.drop_path=nn.Identity()
    self.attn=nn.MultiheadAttention(embed_dim,num_heads,dropout=attn_drop_rate,batch_first=True)
    self.norm2=nn.LayerNorm(embed_dim)
    mlp_hidden_dim=int(embed_dim*mlp_ratio)
    self.mlp=nn.Sequential(
        nn.Linear(embed_dim,mlp_hidden_dim),
        nn.GELU(),
        nn.Dropout(drop_rate),
        nn.Linear(mlp_hidden_dim,embed_dim),
        nn.Dropout(drop_rate)
    )
  def forward(self,x):
    x=x+self.drop_path(self.attn(self.norm1(x),self.norm1(x),self.norm1(x))[0])
    return x+self.drop_path(self.mlp(self.norm2(x)))
  
class VideoEncoder(nn.Module):
    def __init__(self, img_size=(16, 224, 224), patch_size=(2, 16, 16), in_channels=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, has_cls_token=True):
        super().__init__()
        T, H, W = img_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.has_cls_token = has_cls_token

        # Ensure PatchEmbedding, SpatioTemporalPositionalEncoding, and TransformerBlock
        # are defined elsewhere in your project.
        self.patch_embed = PatchEmbedding(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            T=T, H=H, W=W
        )

        num_patches_t = self.patch_embed.num_patches_t
        num_patches_h = self.patch_embed.num_patches_h
        num_patches_w = self.patch_embed.num_patches_w

        if self.has_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = SpatioTemporalPositionalEncoding(
            embed_dim=embed_dim,
            num_patches_t=num_patches_t,
            num_patches_h=num_patches_h,
            num_patches_w=num_patches_w,
            dropout=drop_rate
        )

        # Drop path rate schedule (standard for ViT/Video Transformers)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[i]
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self._init_weights()

    def _init_weights(self):
        if self.has_cls_token:
            # Fix: was referencing self.cls_has_token
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._xavier_uniform_init_linear_layers)

    def _xavier_uniform_init_linear_layers(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        if self.has_cls_token:
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        x = self.pos_embed(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.has_cls_token:
            # Return the CLS token representation
            return x[:, 0]
        # Return average pool if no CLS token, or the whole sequence depending on use-case
        return x.mean(dim=1)

class MaskingGenerator:
  def __init__(self,input_size=(4,14,14),num_masking_patches=75,min_num_patches=16,max_num_patches=784,min_aspect=None,max_aspect=None):
    if not isinstance(input_size,tuple):
      input_size=(input_size,)*3
    self.input_size=input_size
    self.num_masking_patches=num_masking_patches
    self.min_num_patches=min_num_patches
    self.max_num_patches=max_num_patches
    self.min_aspect=min_aspect
    self.max_aspect=max_aspect
  def __repr__(self):
    repr_str = "MaskingGenerator(input_size={}, num_masking_patches={}, min_num_patches={}, max_num_patches={}, min_aspect={}, max_aspect={})".format(
            self.input_size, self.num_masking_patches, self.min_num_patches, self.max_num_patches, self.min_aspect, self.max_aspect
        )
    return repr_str
  def get_random_block(self,max_t,max_h,max_w,min_num_patches,max_num_patches,min_aspect,max_aspect):
    total_patches=max_t*max_h*max_w
    target_num_patches=np.random.randint(min_num_patches,min(max_num_patches+1,total_patches+1))
    t=np.random.randint(1,max_t+1)
    h=np.random.randint(1,max_h+1)
    w=np.random.randint(1,max_w+1)
    current_patches=t*h*w
    if current_patches>target_num_patches:
      dims=[(t, 't'), (h, 'h'), (w, 'w')]
      np.random.shuffle(dims)
      for dim_val,dim_name in dims:
        if current_patches>target_num_patches and dim_val>1:
          reduction_factor=np.sqrt(target_num_patches/current_patches)
          if dim_name=='t':
            t=max(t,int(t*reduction_factor))
          elif dim_name=='h':
            h=max(h,int(h*reduction_factor))
          elif dim_name=='w':
            w=max(w,int(w*reduction_factor))
          current_patches=t*h*w
    elif current_patches<target_num_patches:
      dims= [(t, 't'), (h, 'h'), (w, 'w')]
      np.random.shuffle(dims)
      for dim_val ,dim_name in dims:
        if current_patches<target_num_patches:
          if dim_name=='t' and t<max_t:
            t=min(max_t,int(t*(target_num_patches/current_patches)))
          elif dim_name=='h' and h<max_h:
            h=min(max_h,int(t*(target_num_patches/current_patches)))
          elif dim_name=='w' and  w<max_w:
            w=min(w,int(w*(target_num_patches/current_patches)))
          current_patches=t*h*w
    t=min(t,max_t)
    h=min(h,max_h)
    w=min(w,max_w)
    t_start=np.random.randint(0,max_t-t+1) if max_t-t+1>0 else 0
    h_start=np.random.randint(0,max_h-h+1) if max_h-h+1>0 else 0
    w_start=np.random.randint(0,max_w-w+1) if max_w-w+1>0 else 0
    block_mask=np.zeros(self.input_size,dtype=bool)
    block_mask[t_start:t_start+t,h_start:h_start+h,w_start:w_start+w]=True
    return block_mask
  def __call__(self,video):
    mask=np.zeros(self.input_size,dtype=bool)
    masked_count=0
    while masked_count<self.num_masking_patches:
      block_mask=self.get_random_block(
          self.input_size[0],self.input_size[1],self.input_size[2],
          self.min_num_patches,self.max_num_patches,self.min_aspect,self.max_aspect
      )
      newly_masked_area=block_mask & (~mask)
      mask=mask | block_mask
      masked_count=np.sum(newly_masked_area)
      if masked_count >=self.num_masking_patches:
        break
    flat_mask=mask.flatten()
    masked_indices=np.where(flat_mask)[0]
    if len(masked_indices)>self.num_masking_patches:
      unmask_count=len(masked_indices)-self.num_masking_patches
      unmask_indices_flat=np.random.choice(masked_indices,unmask_count,replace=False)
      flat_mask[unmask_indices_flat]=False
      return flat_mask.reshape(mask.shape)

class MaskedEncoder(nn.Module):
    def __init__(self, video_encoder, masking_generator, mask_ratio=None):
        super().__init__()
        self.video_encoder = video_encoder
        self.masking_generator = masking_generator

        # Override masking generator patches if mask_ratio is provided
        if mask_ratio is not None:
            num_total_patches = np.prod(masking_generator.input_size)
            self.masking_generator.num_masking_patches = int(num_total_patches * mask_ratio)

    def forward(self, videos):
        B, C, T, H, W = videos.shape

        # 1. Patch Embedding: (B, num_patches, embed_dim)
        patch_embeddings = self.video_encoder.patch_embed(videos)
        num_patches = patch_embeddings.shape[1]
        device = videos.device

        # 2. Generate Masks for the batch
        # We collect masks as boolean tensors where True = Masked, False = Visible
        batch_masks = []
        for _ in range(B):
            mask = self.masking_generator(None) # Assuming it returns a numpy array
            batch_masks.append(torch.from_numpy(mask.flatten()).to(device).bool())
        batch_masks = torch.stack(batch_masks) # (B, num_patches)

        # 3. Extract Visible Patches
        # Since different videos might have different numbers of visible patches
        # (if using random masking), we usually force a fixed number for batching.
        # Assuming masking_generator provides a fixed count:
        visible_patches_list = []
        visible_indices_list = []
        masked_indices_list = []

        all_indices = torch.arange(num_patches).to(device)

        for i in range(B):
            mask_i = batch_masks[i]

            # Filter embeddings and indices
            visible_patches_list.append(patch_embeddings[i][~mask_i])
            visible_indices_list.append(all_indices[~mask_i])
            masked_indices_list.append(all_indices[mask_i])

        # 4. Batch the visible patches for the Transformer
        # (B, num_visible, embed_dim)
        visible_patches = torch.stack(visible_patches_list)
        visible_indices = torch.stack(visible_indices_list)
        masked_indices = torch.stack(masked_indices_list)

        # 5. Add Positional Encoding (Specific to visible patches)
        # We pass indices to pos_embed to ensure patches get the correct 3D position
        x = self.video_encoder.pos_embed(visible_patches, indices=visible_indices)

        # 6. Pass through Transformer Blocks
        for block in self.video_encoder.blocks:
            x = block(x)

        x = self.video_encoder.norm(x)

        return x, visible_indices, masked_indices

class PredictionHead(nn.Module):
    def __init__(self, embed_dim, decoder_embed_dim, out_dim=None):
        super().__init__()
        self.decoder_embed_dim = decoder_embed_dim
        # out_dim is usually patch_size^2 * channels (e.g., 16*16*3 = 768)
        self.out_dim = out_dim if out_dim is not None else embed_dim

        # Project encoder features to decoder dimension
        self.proj_visible = nn.Linear(embed_dim, decoder_embed_dim)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Prediction MLP
        self.decoder_pred = nn.Sequential(
            nn.Linear(decoder_embed_dim, decoder_embed_dim),
            nn.GELU(),
            nn.Linear(decoder_embed_dim, self.out_dim)
        )

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, visible_features, visible_indices, masked_indices, num_total_patches):
        B = visible_features.shape[0]
        device = visible_features.device

        # 1. Project visible features to decoder space
        # (B, num_visible, decoder_embed_dim)
        x_visible = self.proj_visible(visible_features)

        # 2. Create the full sequence template filled with mask tokens
        # (B, num_total_patches, decoder_embed_dim)
        full_sequence = self.mask_token.expand(B, num_total_patches, -1).clone()

        # 3. Scatter visible features back into their original positions
        # We use batch-aware indexing to place features correctly
        batch_indices = torch.arange(B, device=device).unsqueeze(-1)
        full_sequence[batch_indices, visible_indices] = x_visible

        # 4. Extract only the segments that were masked for prediction
        # (B, num_masked, decoder_embed_dim)
        masked_tokens = full_sequence[batch_indices, masked_indices]

        # 5. Predict original patch values
        # (B, num_masked, out_dim)
        predicted_values = self.decoder_pred(masked_tokens)

        return predicted_values