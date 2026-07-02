class AJEPADatasetTransform:
  def __init__(self,mel_spectrogram_extractor,weak_augmentation,strong_augmentation,audio_masking):
    self.mel_spectrogram_extractor=mel_spectrogram_extractor
    self.weak_augment=weak_augmentation
    self.strong_augment=strong_augmentation
    self.audio_masking=audio_masking
  def __call__(self,raw_waveform):
    base_mel_spectrogram=self.mel_spectrogram_extractor(raw_waveform)
    weakly_augmented_mel=self.weak_augment(base_mel_spectrogram)
    context_view=self.audio_masking(weakly_augmented_mel)
    target_view=self.strong_augment(base_mel_spectrogram)
    return context_view,target_view