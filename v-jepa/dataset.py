import webdataset as wds
import tempfile # Import tempfile

class WebVideoDataset:
  def __init__(self, data_path_pattern, video_transform, total_size, shard_shuffle=False, num_workers=4):
    self.data_path_pattern = data_path_pattern
    self.video_transform = video_transform
    self.total_size = total_size
    self.shard_shuffle = shard_shuffle
    self.num_workers = num_workers

  def __len__(self):
    return self.total_size

  def create_dataset(self):
    dataset = wds.WebDataset(self.data_path_pattern, shardshuffle=False, empty_check=False)
    dataset = dataset.map_dict(mp4=self._decoder_and_transform_video)
    dataset = dataset.with_length(self.total_size)
    return dataset

  def _decoder_and_transform_video(self, video_bytes):
    # Use tempfile to create a unique temporary file for each worker process
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as temp_f:
      temp_f.write(video_bytes)
      temp_f.flush() # Ensure data is written to disk before being read
      # Pass the name of the temporary file to video_transform
      video_tensor = self.video_transform(temp_f.name)
    # The temporary file is automatically deleted when temp_f is closed
    return video_tensor
