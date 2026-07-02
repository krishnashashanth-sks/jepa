import copy
import torch.nn.functional as F
from layers import *

class VJEPA(nn.Module):
  def __init__(self,student_encoder:VideoEncoder,masking_generator:MaskingGenerator,
               prediction_head:PredictionHead,mask_ratio:float=0.6,
               teacher_momentum:float=0.996,freeze_teacher_temp:float=0.0):
    super().__init__()
    self.mask_ratio=mask_ratio
    self.teacher_momentum=teacher_momentum
    self.freeze_teacher_temp=freeze_teacher_temp 

    self.student_encoder=student_encoder
    self.teacher_encoder=copy.deepcopy(student_encoder)
    for p in self.teacher_encoder.parameters(): 
      p.requires_grad=False

    self.masking_generator=masking_generator
    self.prediction_head=prediction_head

    self.student_masked_input_preparer=MaskedEncoder(
        video_encoder=self.student_encoder,
        masking_generator=masking_generator, 
        mask_ratio=self.mask_ratio
    )
    self.num_total_patches=np.prod(self.student_encoder.patch_embed.num_patches)

  @torch.no_grad()
  def _update_teacher_encoder(self):
    for student_p,teacher_p in zip(self.student_encoder.parameters(),self.teacher_encoder.parameters()):
      teacher_p.data=teacher_p.data * self.teacher_momentum+student_p.data *(1.-self.teacher_momentum)

  def forward(self,videos):
    visible_patches_student,visible_indices_student,masked_indices=self.student_masked_input_preparer(videos)

    B = videos.shape[0]
    student_output_visible = []

    for i in range(B):
        current_visible_patches = visible_patches_student[i] # Shape: (num_visible_i, embed_dim)
        current_visible_indices = visible_indices_student[i] # Shape: (num_visible_i)

        input_for_blocks = []
        if self.student_encoder.has_cls_token:
            cls_token = self.student_encoder.cls_token.squeeze(0) # (embed_dim)
            input_for_blocks.append(cls_token.unsqueeze(0)) # (1, embed_dim)

        full_pos_embed_grid = self.student_encoder.pos_embed.pos_embed_t.unsqueeze(2).unsqueeze(3) + \
                              self.student_encoder.pos_embed.pos_embed_h.unsqueeze(1).unsqueeze(3) + \
                              self.student_encoder.pos_embed.pos_embed_w.unsqueeze(1).unsqueeze(2)
        full_pos_embed_flat = full_pos_embed_grid.view(-1, self.student_encoder.embed_dim) # (num_total_patches, embed_dim)

        pos_embed_for_visible = full_pos_embed_flat[current_visible_indices] # (num_visible_i, embed_dim)
        visible_tokens_with_pos = current_visible_patches + pos_embed_for_visible

        input_for_blocks.append(visible_tokens_with_pos)

        x_student_seq = torch.cat(input_for_blocks, dim=0) # (1 + num_visible_i, embed_dim) or (num_visible_i, embed_dim)

        x_student_encoded = x_student_seq.unsqueeze(0) # Add batch dim (B=1 for block processing)
        for block in self.student_encoder.blocks:
            x_student_encoded = block(x_student_encoded)
        x_student_encoded = self.student_encoder.norm(x_student_encoded.squeeze(0)) # Remove batch dim

        student_output_visible.append(x_student_encoded)

    with torch.no_grad(): # Teacher encoder operates without gradients
      teacher_output_full=self.teacher_encoder.forward_features(videos)
      teacher_features_masked_target=[]
      for i in range(B):
        if self.teacher_encoder.has_cls_token:
          current_teacher_patches=teacher_output_full[i,1:]
        else:
          current_teacher_patches=teacher_output_full[i]
        teacher_features_masked_target.append(current_teacher_patches[masked_indices[i]])

    proj_visible_for_pred_head=[]
    visible_indices_for_pred_head = []
    for i in range(B):
      if self.student_encoder.has_cls_token: # Fixed attribute access
        proj_visible_for_pred_head.append(student_output_visible[i][1:])
      else:
        proj_visible_for_pred_head.append(student_output_visible[i])
      visible_indices_for_pred_head.append(visible_indices_student[i])
    predicted_masked_features=self.prediction_head(
        proj_visible_for_pred_head,
        visible_indices_for_pred_head,
        masked_indices,
        B,
        self.num_total_patches
    )
    loss=0.0
    for i in range(B):
      loss+=F.mse_loss(predicted_masked_features[i],teacher_features_masked_target[i])
    loss/=B

    self._update_teacher_encoder()
    return loss,predicted_masked_features,teacher_features_masked_target