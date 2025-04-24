import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
import torchvision.transforms as transforms


@dataclass
class Qwen2VLConfig:
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vision_token_id: int = 151654
    image_token_id: int = 151655
    video_token_id: int = 151656
    hidden_act: str = "silu"
    hidden_size: int = 2048
    initializer_range: float = 0.02
    intermediate_size: int = 11008
    max_position_embeddings: int = 128000
    max_window_layers: int = 70
    model_type: str = "qwen2_5_vl"
    num_attention_heads: int = 16
    num_hidden_layers: int = 36
    num_key_value_heads: int = 2
    rms_norm_eps: float = 1e-06
    rope_theta: float = 1000000.0
    sliding_window: int = 32768
    tie_word_embeddings: bool = True
    torch_dtype: str = "bfloat16"
    use_cache: bool = True
    use_sliding_window: bool = False
    vocab_size: int = 151936
    
    # Vision specific config
    vision_config: Dict[str, Any] = None


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        x = self._norm(x).type_as(x)
        x = self.weight * x.to(input_dtype)
        return x


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(self, args: Qwen2VLConfig):
        super().__init__()
        self.n_kv_heads = (
            args.num_attention_heads
            if args.num_key_value_heads is None
            else args.num_key_value_heads
        )
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = self.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.q_proj = nn.Linear(
            args.hidden_size,
            args.num_attention_heads * self.head_dim,
            bias=True,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            args.num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            args.num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=False,
        )
        self.args = args

    def init_kv_cache(
        self,
        max_batch_size: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        cache_shape = (max_batch_size, max_seq_len, self.n_kv_heads, self.head_dim)
        cache_k = torch.zeros(cache_shape, dtype=dtype, device=device)
        cache_v = torch.zeros(cache_shape, dtype=dtype, device=device)
        self.register_buffer("cache_k", cache_k, persistent=False)
        self.register_buffer("cache_v", cache_v, persistent=False)

    def del_kv_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(
        self,
        x: torch.Tensor,
        pos_embed: Tuple[torch.Tensor, torch.Tensor],
        start_pos: Optional[Union[int, torch.Tensor]] = None,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        cos, sin = pos_embed
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin, unsqueeze_dim=2)
        if start_pos is not None:
            # inference mode
            end_pos = start_pos + seqlen
            self.cache_k[:bsz, start_pos:end_pos, :, :] = xk
            self.cache_v[:bsz, start_pos:end_pos, :, :] = xv
            output = torch.nn.functional.scaled_dot_product_attention(
                query=xq.transpose(1, 2),
                key=self.cache_k[:bsz, :end_pos].transpose(1, 2),
                value=self.cache_v[:bsz, :end_pos].transpose(1, 2),
                is_causal=True if seqlen > 1 else False,
                enable_gqa=True,
            ).transpose(1, 2)
        else:
            # training mode
            output = torch.nn.functional.scaled_dot_product_attention(
                query=xq.transpose(1, 2),
                key=xk.transpose(1, 2),
                value=xv.transpose(1, 2),
                is_causal=True,
                enable_gqa=True,
            ).transpose(1, 2)
        output = output.reshape(bsz, seqlen, -1)
        return self.o_proj(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)

    def forward(self, x):
        x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: Qwen2VLConfig):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.dim = args.hidden_size
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.self_attn = Attention(args)
        self.mlp = FeedForward(
            dim=args.hidden_size,
            intermediate_size=args.intermediate_size,
        )
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        pos_embed: Tuple[torch.Tensor, torch.Tensor],
        start_pos: Optional[Union[int, torch.Tensor]] = None,
    ):
        h = x + self.self_attn(self.input_layernorm(x), pos_embed, start_pos=start_pos)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2VLConfig, device: torch.device):
        super().__init__()
        self.config = config
        base = config.rope_theta
        dim = config.hidden_size // config.num_attention_heads
        with torch.autocast(device_type=device.type, dtype=torch.float32):
            inv_freq = 1.0 / (
                base
                ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
            )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, pos):
        inv_freq = self.inv_freq[None, :, None].float().expand(pos.shape[0], -1, 1)
        pos = pos[:, None, :].float()
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq.float().to(x.device) @ pos.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class VisionEncoder(nn.Module):
    """Vision Encoder for Qwen2.5-VL model"""
    def __init__(self, vision_config, device):
        super().__init__()
        self.hidden_size = vision_config["hidden_size"]  # 1280
        self.out_hidden_size = vision_config["out_hidden_size"]  # 2048
        self.patch_size = vision_config["patch_size"]  # 14
        self.window_size = vision_config["window_size"]  # 112
        self.embed_dim = vision_config["hidden_size"]  # 1280
        self.num_heads = vision_config["num_heads"]  # 16
        
        # Simplified vision encoder to match the Qwen2.5-VL architecture
        # In a real implementation, this would have the complete transformer blocks
        # with attention layers, etc.
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, self.hidden_size, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        
        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, (self.window_size // self.patch_size) ** 2, self.hidden_size)
        )
        
        # Vision transformer layers (simplified for this implementation)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=self.num_heads,
                dim_feedforward=vision_config["intermediate_size"],
                activation="gelu",
                batch_first=True
            )
            for _ in range(vision_config["depth"])
        ])
        
        # Output projection to align with text model dimension
        self.output_projection = nn.Linear(self.hidden_size, self.out_hidden_size)
        
        # Normalization
        self.norm = nn.LayerNorm(self.hidden_size)
        
        # Image preprocessing parameters
        self.register_buffer(
            "pixel_mean", 
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1)
        )
        self.register_buffer(
            "pixel_std", 
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1)
        )

    def preprocess_image(self, images):
        """Preprocess images to match model's expected input format"""
        if isinstance(images, list):
            if not all(isinstance(img, Image.Image) for img in images):
                raise ValueError("All elements in the list must be PIL Images")
        elif isinstance(images, Image.Image):
            images = [images]
        else:
            raise ValueError("Input must be a PIL Image or a list of PIL Images")
        
        # Get the device of the model buffers
        device = self.pixel_mean.device
        
        # Convert to tensor and normalize
        processed_images = []
        for img in images:
            # Target size
            img = img.convert("RGB")
            img = img.resize((self.window_size, self.window_size))
            
            # Convert to tensor
            img_tensor = transforms.ToTensor()(img)
            
            # Move to same device as model before normalization
            img_tensor = img_tensor.to(device)
            
            # Normalize
            img_tensor = (img_tensor - self.pixel_mean) / self.pixel_std
            
            processed_images.append(img_tensor)
        
        # Stack into batch
        return torch.stack(processed_images)

    def forward(self, pixel_values):
        """
        Process image inputs and return embeddings
        Args:
            pixel_values: preprocessed image tensors [B, C, H, W]
        Returns:
            image_features: embeddings compatible with text model
        """
        batch_size = pixel_values.shape[0]
        
        # Patch embedding [B, C, H, W] -> [B, hidden_size, H/patch_size, W/patch_size]
        x = self.patch_embed(pixel_values)
        
        # Reshape to sequence [B, hidden_size, H', W'] -> [B, H'*W', hidden_size]
        h = x.shape[2]
        w = x.shape[3]
        x = x.permute(0, 2, 3, 1).reshape(batch_size, h * w, self.hidden_size)
        
        # Add position embeddings
        if x.size(1) <= self.pos_embed.size(1):
            x = x + self.pos_embed[:, :x.size(1), :]
        else:
            # Handle variable sequence lengths
            x = x + self.pos_embed.repeat(1, math.ceil(x.size(1) / self.pos_embed.size(1)), 1)[:, :x.size(1), :]
        
        # Process through transformer blocks
        for blk in self.transformer_blocks:
            x = blk(x)
        
        # Apply normalization
        x = self.norm(x)
        
        # Output projection to match text model dimension
        x = self.output_projection(x)
        
        return x


class Transformer(nn.Module):
    def __init__(self, params: Qwen2VLConfig, device: torch.device):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.num_hidden_layers

        self.embed_tokens = torch.nn.Embedding(params.vocab_size, params.hidden_size)
        
        # Initialize rotary embeddings for text
        with torch.device(device):
            self.rotary_emb = Qwen2RotaryEmbedding(config=params, device=device)

        # Initialize vision encoder
        if params.vision_config:
            self.vision_encoder = VisionEncoder(params.vision_config, device)
        else:
            self.vision_encoder = None

        # Text transformer layers
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.num_hidden_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.hidden_size, eps=params.rms_norm_eps)
        if not params.tie_word_embeddings:
            self.lm_head = nn.Linear(params.hidden_size, params.vocab_size, bias=False)

    def output_proj(self, x):
        if self.params.tie_word_embeddings:
            return x @ self.embed_tokens.weight.T
        else:
            return self.lm_head(x)

    def encode_images(self, images):
        """Process images and return embeddings"""
        if self.vision_encoder is None:
            raise ValueError("Vision encoder not initialized")
            
        # Preprocess images if needed
        if not isinstance(images, torch.Tensor):
            pixel_values = self.vision_encoder.preprocess_image(images)
        else:
            pixel_values = images
            
        # Generate image embeddings
        image_embeds = self.vision_encoder(pixel_values)
        
        return image_embeds

    def prepare_multimodal_inputs(self, tokens, image_embeds=None, image_positions=None):
        """
        Prepare inputs for multimodal (text+image) processing
        
        Args:
            tokens: token ids [B, S]
            image_embeds: image embeddings from vision encoder [B, P, D]
            image_positions: where to insert image tokens in the sequence
              If None, assumes image tokens are already in the sequence
              
        Returns:
            Extended sequence with image embeddings inserted
        """
        batch_size, seq_len = tokens.shape
        
        # If no images, just embed tokens normally
        if image_embeds is None:
            return self.embed_tokens(tokens)
            
        # If image positions not provided, find image token positions
        if image_positions is None:
            image_positions = []
            for b in range(batch_size):
                # Find positions of image tokens in the sequence
                pos = (tokens[b] == self.params.image_token_id).nonzero(as_tuple=True)[0]
                if len(pos) > 0:
                    image_positions.append(pos[0].item())
                else:
                    image_positions.append(None)
        
        # Embed tokens first
        embeds = self.embed_tokens(tokens)
        
        # Insert image embeddings at the specified positions
        for b in range(batch_size):
            if image_positions[b] is not None:
                pos = image_positions[b]
                
                # Get number of image tokens for this example
                img_length = image_embeds[b].shape[0]
                
                # Replace the image token with the image embeddings
                embeds[b, pos:pos+1, :] = image_embeds[b].mean(dim=0, keepdim=True)
        
        return embeds

    def forward(self, tokens: torch.Tensor, images=None):
        """
        Forward pass for training
        
        Args:
            tokens: input token ids [batch_size, seq_len]
            images: optional image inputs (PIL images or tensors)
            
        Returns:
            logits for next token prediction
        """
        _bsz, seqlen = tokens.shape
        
        # Process images if provided
        image_embeds = None
        if images is not None and self.vision_encoder is not None:
            image_embeds = self.encode_images(images)
        
        # Get input embeddings with images integrated
        h = self.prepare_multimodal_inputs(tokens, image_embeds)
        
        # Get position embeddings
        pos = torch.arange(0, seqlen, device=tokens.device, dtype=torch.int32)
        pos_emb = self.rotary_emb(h, pos[None, :])

        # Process through transformer layers with checkpointing
        pipe = []
        for layer in self.layers:
            pipe.append(lambda x, layer=layer: layer(x, pos_emb))
        pipe.append(self.norm.forward)
        pipe.append(self.output_proj)
        
        return torch.utils.checkpoint.checkpoint_sequential(
            pipe, len(pipe), h, use_reentrant=False
        )

    def inference(self, tokens: torch.Tensor, images=None, start_pos: Union[int, torch.Tensor] = 0):
        """
        Forward pass for inference with KV caching
        
        Args:
            tokens: input token ids [batch_size, seq_len]
            images: optional image inputs (PIL images or tensors)
            start_pos: position offset for kv cache
            
        Returns:
            logits for next token prediction
        """
        _bsz, seqlen = tokens.shape
        
        # Process images if provided
        image_embeds = None
        if images is not None and self.vision_encoder is not None:
            image_embeds = self.encode_images(images)
        
        # Get input embeddings with images integrated
        h = self.prepare_multimodal_inputs(tokens, image_embeds)

        # Position embeddings with offset for kv cache
        pos = torch.arange(0, seqlen, device=tokens.device, dtype=torch.int32)[None, :]
        if isinstance(start_pos, torch.Tensor):
            pos = pos + start_pos[:, None]
        else:  # int
            pos.add_(start_pos)
        pos_emb = self.rotary_emb(h, pos)

        # Process through transformer layers
        for layer in self.layers:
            h = layer(h, pos_emb, start_pos=start_pos)

        # Only need the hidden state of the last token
        # to predict the next token
        h = h[:, -1:, :]
        h = self.norm(h)

        output = self.output_proj(h)
        return output

    def init_kv_cache(
        self,
        max_batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        for layer in self.layers:
            layer.self_attn.init_kv_cache(
                max_batch_size, max_seq_len, dtype=dtype, device=device
            )

    def del_kv_cache(self):
        for layer in self.layers:
            layer.self_attn.del_kv_cache()

    def freeze_vision_encoder(self):
        """
        Freeze the parameters of the vision encoder to prevent updates during training.
        This is useful for fine-tuning scenarios where you want to keep the vision encoder fixed.
        """
        if self.vision_encoder is not None:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            print("Vision encoder has been frozen.")
        else:
            print("No vision encoder to freeze.")

    @classmethod
    def from_pretrained(cls, ckpt_path, device: torch.device):
        config_file = Path(ckpt_path) / "config.json"
        with open(config_file, "r") as f:
            config = json.load(f)
        
        # Create vision config
        vision_config = config.get("vision_config", None)
        
        # Create model config
        args = Qwen2VLConfig(
            attention_dropout=config.get("attention_dropout", 0.0),
            bos_token_id=config.get("bos_token_id", 151643),
            eos_token_id=config.get("eos_token_id", 151645),
            vision_start_token_id=config.get("vision_start_token_id", 151652),
            vision_end_token_id=config.get("vision_end_token_id", 151653),
            vision_token_id=config.get("vision_token_id", 151654),
            image_token_id=config.get("image_token_id", 151655),
            video_token_id=config.get("video_token_id", 151656),
            hidden_act=config.get("hidden_act", "silu"),
            hidden_size=config.get("hidden_size", 2048),
            initializer_range=config.get("initializer_range", 0.02),
            intermediate_size=config.get("intermediate_size", 11008),
            max_position_embeddings=config.get("max_position_embeddings", 128000),
            max_window_layers=config.get("max_window_layers", 70),
            model_type=config.get("model_type", "qwen2_5_vl"),
            num_hidden_layers=config.get("num_hidden_layers", 36),
            num_attention_heads=config.get("num_attention_heads", 16),
            num_key_value_heads=config.get("num_key_value_heads", 2),
            vocab_size=config.get("vocab_size", 151936),
            rms_norm_eps=config.get("rms_norm_eps", 1e-6),
            rope_theta=config.get("rope_theta", 1000000.0),
            sliding_window=config.get("sliding_window", 32768),
            use_sliding_window=config.get("use_sliding_window", False),
            use_cache=config.get("use_cache", True),
            tie_word_embeddings=config.get("tie_word_embeddings", True),
            torch_dtype=config.get("torch_dtype", "bfloat16"),
            vision_config=vision_config,
        )
        
        with torch.device("meta"):
            model = cls(params=args, device=device)

        # Move model from meta device to the target device using to_empty()
        model = model.to_empty(device=device)

        import safetensors.torch

        model_weight_files = sorted(Path(ckpt_path).glob("model*.safetensors"))
        weights = {}
        for file in model_weight_files:
            weights.update(safetensors.torch.load_file(file, device="cpu"))
        
        # Remove "model." prefix from keys
        weights = {k.replace("model.", ""): v for k, v in weights.items()}
        
        # Handle missing or unexpected keys for this implementation
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in weights.items() if k in model_dict}
        
        # Load the weights that match the model structure
        model.load_state_dict(pretrained_dict, strict=False)
        
        return model 