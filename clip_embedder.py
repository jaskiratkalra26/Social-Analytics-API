import torch
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor
from PIL import Image
import numpy as np

class ClipEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading CLIP model '{model_name}' to {self.device}...")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def get_embedding(self, frames):
        """
        Takes a list of numpy array RGB frames, converts to PIL, runs through CLIP,
        averages the embeddings, and returns a single embedding vector (512,).
        Processes in batches to manage memory.
        """
        if not frames:
            return None

        # Convert numpy frames to PIL images
        pil_images = [Image.fromarray(frame) for frame in frames]
        
        # Batch processing
        BATCH_SIZE = 8
        all_embeddings = []

        for i in range(0, len(pil_images), BATCH_SIZE):
            batch = pil_images[i : i + BATCH_SIZE]
            
            # Prepare inputs
            inputs = self.processor(images=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)

            # Robust handling for output type (Tensor vs ModelOutput)
            if not isinstance(image_features, torch.Tensor):
                # If it's a HuggingFace ModelOutput, try to get the embeddings
                if hasattr(image_features, "image_embeds"):
                    image_features = image_features.image_embeds
                elif hasattr(image_features, "pooler_output"):
                    image_features = image_features.pooler_output
                elif hasattr(image_features, "last_hidden_state"):
                     # Should not happen for get_image_features, but as fallback
                     image_features = image_features.last_hidden_state
                elif isinstance(image_features, (tuple, list)):
                    image_features = image_features[0]

            if isinstance(image_features, torch.Tensor):
                all_embeddings.append(image_features.cpu())
            
        if not all_embeddings:
             print(f"[ERROR] CLIP output collected no embeddings.")
             return None

        # Concatenate all batches 
        all_embeddings = torch.cat(all_embeddings, dim=0)

        # Averaging embeddings for the video (shape: [num_frames, 512]) -> [1, 512]
        avg_embedding = torch.mean(all_embeddings, dim=0).numpy()
        
        return avg_embedding

if __name__ == "__main__":
    # Test
    embedder = ClipEmbedder()
    dummy_frames = [(np.random.rand(224, 224, 3) * 255).astype(np.uint8) for _ in range(3)]
    emb = embedder.get_embedding(dummy_frames)
    print("Embedding shape:", emb.shape)
