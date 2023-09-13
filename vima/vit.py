from vima.transformer import ViTransformerWrapper, Encoder

class ImageEncoder:
    # # Usage:
    # image_encoder = ImageEncoder()
    # img_embeddings = image_encoder.embed_image_data([img1, img2])  # You'd provide your list of image data here.

    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 32,
        encoder_dim: int = 512,
        encoder_depth: int = 6,
        encoder_heads: int = 8,
    ):
        super().__init__()
        
        self.encoder = ViTransformerWrapper(
            image_size=image_size,
            patch_size=patch_size,
            attn_layers=Encoder(
                dim=encoder_dim,
                depth=encoder_depth,
                heads=encoder_heads,
            )
        )

    def embed(self, img):
        encoded = self.encoder(img, return_embeddings=True)
        return encoded
