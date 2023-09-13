import torch 
from torch.nn import Module
from transformers import AutoTokenizer

from vima.transformer import (
    Decoder,
    Transformer,
    ViTransformerWrapper,
    Encoder
)
import logging

from vima.autoregressive import AutoregressiveWrapper

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class VimaTokenizer:
    """
    A tokenizer class for the Vima model

    Attributes:
        processor(CLIPProcessor): The processor to tokenize images
        tokenizer: (AutoTokenizer): The tokenizer to tokenize text 
        im_idx: (int): The Index of the "<image>" token.
        im_end_idx (int): The index of the "</image>" token.
    """
    def __init__(self):
        try:
            # self.processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/gpt-neox-20b",
                additional_special_tokens=["<image>", "</image>", "<audio>", "</audio>", "<video>", "</video>"],
                eos_token="<eos>",
                pad_token="<pad>",
                extra_ids=0,
                model_max_length=8192
            )
        except Exception as e:
            logging.error(f"Failed to initialize VimaTokenizer: {e}")
            raise

        self.im_idx, self.im_end_idx = self.tokenizer.convert_tokens_to_ids(["<image>", "</image>"])

    def tokenize_texts(self, texts: str):
        """
        Tokenize given texts.

        Args:
            Texts (str): The Text to be tokenized

        
        Returns:
            A tuple containing the tokenized texts and only the text tokens.
        """
        try:
            texts =  self.tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).input_ids
            # Add image tokens to text as "<s> <image> </image> text </s>"
            image_tokens = torch.tensor([[self.im_idx, self.im_end_idx]] * texts.shape[0])
            return torch.cat([texts[:, 0:1], image_tokens, texts[:, 1:]], dim=1), texts
        except Exception as e:
            logging.error(f"Failed to tokenize texts: {e}")
            raise

    def tokenize_images(self, images):
        """
        Tokenizes given images.

        Args:
            images: The images to be tokenized

        Returns:
            The tokenized images.
        
        """
        try:
            return self.processor(images=images, return_tensors="pt").pixel_values
        except Exception as e:
            logging.error(f"Failed to tokenize images: {e}")
            raise

    def tokenize(self, sample):
        """
        Tokenizes given sample.

        Args:
            Sample: The sample to be tokenized

        Returns:
            A dictionary containing the tokenized text tokens, images, labels, and attention mask.
        
        """
        try:
            text_tokens, only_text_tokens = self.tokenize_texts(sample["target_text"])
            attention_mask = text_tokens != self.tokenizer.pad_token_id
            dummy_image_features = torch.ones((text_tokens.shape[0], 64))
            attention_mask = torch.cat([dummy_image_features, attention_mask], dim=1)
            return {
                "text_tokens": text_tokens,
                "images": self.tokenize_images(sample["image"]),
                "labels": only_text_tokens,
                "attention_mask": attention_mask,
            }
        except Exception as e:
            logging.error(f"Failed to tokenize sample: {e}")
            raise



class Vima(Module):
    """
    Vima is a transformer-based model architecture. It initializes with 
    a Transformer and AutoregressiveWrapper with default or user-specified parameters.
    """
    def __init__(
            self, 
            num_tokens=50432, 
            max_seq_len=8192, 
            dim=2560, 
            depth=32, 
            dim_head=128, 
            heads=24,
            use_abs_pos_emb=False, 
            alibi_pos_bias=True, 
            alibi_num_heads=12, 
            rotary_xpos=True,
            attn_flash=True, 
            attn_kv_heads = 2,
            qk_norm=True, 
            attn_qk_norm=True, 
            attn_qk_norm_dim_scale=True, 
        ):
        """
        Initialize the model with specified or default parameters.
        Args:
        - num_tokens: Number of tokens in the vocabulary
        - max_seq_len: Maximum sequence length
        - dim: Dimension of the model
        - depth: Depth of the model
        - dim_head: Dimension of the model head
        - heads: Number of heads
        - use_abs_pos_emb: Whether to use absolute position embedding
        - alibi_pos_bias: Alibi position bias
        - alibi_num_heads: Number of alibi heads
        - rotary_xpos: Rotary position
        - attn_flash: Attention flash
        - deepnorm: Deep normalization
        - shift_tokens: Number of tokens to shift
        - attn_one_kv_head: Attention one key/value head
        - qk_norm: Query-key normalization
        - attn_qk_norm: Attention query-key normalization
        - attn_qk_norm_dim_scale: Attention query-key normalization dimension scale
        - embedding_provider: Embedding provider module
        """
        super().__init__()

        try:
            self.Vima = Transformer(
                num_tokens=num_tokens,
                max_seq_len=max_seq_len,
                use_abs_pos_emb=use_abs_pos_emb,
                attn_layers=Decoder(
                    dim=dim,
                    depth=depth,
                    dim_head=dim_head,
                    heads=heads,
                    alibi_pos_bias=alibi_pos_bias,
                    alibi_num_heads=alibi_num_heads,
                    rotary_xpos=rotary_xpos,
                    attn_flash=attn_flash,
                    attn_kv_heads=attn_kv_heads,
                    qk_norm=qk_norm,
                    attn_qk_norm=attn_qk_norm,
                    attn_qk_norm_dim_scale=attn_qk_norm_dim_scale
                )
            )

            self.decoder = AutoregressiveWrapper(self.Vima)

        except Exception as e:
            print("Failed to initialize Vima: ", e)
            raise

    def forward(self, text_tokens, **kwargs):
        """
        Forward pass through the model. It expects the input text_tokens.
        Args:
        - text_tokens: Input tokens
        - kwargs: Other arguments
        Returns:
        - output from the decoder
        """
        try:
            model_input = self.decoder.forward(text_tokens)[0]
            return self.decoder(model_input, padded_x=model_input[0])
        except Exception as e:
            print("Failed in forward method: ", e)
            raise


class VimaMultiModal(Module):
    def __init__(
            self, 
            image_size=256, 
            patch_size=32, 
            encoder_dim=512, 
            encoder_depth=6, 
            encoder_heads=8,
            num_tokens=20000, 
            max_seq_len=1024, 
            decoder_dim=512, 
            decoder_depth=6, 
            decoder_heads=8, 
            alibi_num_heads=4,
            use_abs_pos_emb=False,
            cross_attend=True,
            alibi_pos_bias=True,
            rotary_xpos=True,
            attn_flash=True,
            qk_norm=True
        ):
        super(VimaMultiModal, self).__init__()
        
        self.encoder = ViTransformerWrapper(
            image_size=image_size,
            patch_size=patch_size,
            attn_layers=Encoder(
                dim=encoder_dim,
                depth=encoder_depth,
                heads=encoder_heads
            )
        )

        self.decoder = Transformer(
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            use_abs_pos_emb=use_abs_pos_emb,
            attn_layers=Decoder(
                dim=decoder_dim,
                depth=decoder_depth,
                heads=decoder_heads,
                cross_attend=cross_attend,
                alibi_pos_bias=alibi_pos_bias,
                alibi_num_heads=alibi_num_heads,
                rotary_xpos=rotary_xpos,
                attn_flash=attn_flash,
                qk_norm=qk_norm,
            )
        )

    def forward(self, img, text):
        try:    
            encoded = self.encoder(img, return_embeddings=True)
            return self.decoder(text, context=encoded)
        except Exception as error:
            print(f"Failed in forward method: {error}")
            raise