
from huggingface_hub import snapshot_download, hf_hub_download
import os
import tempfile
from pathlib import Path

class HuggingFaceDownloader:
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def download_model(self, model_id, revision="main"):
        """
        Download a complete HuggingFace model
        Returns: Path to downloaded model directory
        """
        try:
            model_path = snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=self.cache_dir,
                local_dir=os.path.join(self.cache_dir, "models", model_id.replace("/", "_")),
                local_dir_use_symlinks=False
            )
            return model_path
        except Exception as e:
            print(f"Error downloading model {model_id}: {e}")
            return None
    
    def download_specific_file(self, model_id, filename, revision="main"):
        """
        Download a specific file from a HuggingFace model
        """
        try:
            file_path = hf_hub_download(
                repo_id=model_id,
                filename=filename,
                revision=revision,
                cache_dir=self.cache_dir
            )
            return file_path
        except Exception as e:
            print(f"Error downloading file {filename} from {model_id}: {e}")
            return None
    
    def get_model_info(self, model_id):
        """
        Get information about a model without downloading it
        """
        from huggingface_hub import model_info
        try:
            info = model_info(model_id)
            return {
                "id": info.id,
                "downloads": info.downloads,
                "likes": info.likes,
                "tags": info.tags,
                "pipeline_tag": info.pipeline_tag,
                "library_name": info.library_name,
                "model_size": getattr(info, 'model_size', None),
                "safetensors": getattr(info, 'safetensors', None)
            }
        except Exception as e:
            print(f"Error getting model info for {model_id}: {e}")
            return None