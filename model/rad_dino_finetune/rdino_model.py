import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.upernet.decoder import UPerNetDecoder

# rad_dino_model.py
class RADDINO_Model(nn.Module):
    def __init__(
        self,
        n_classes=1,
        use_lora=True,
        apply_mlp_lora=False,
        r=4,
        alpha=8,
        freeze_encoder=True,
        include_segmentation=False,
        img_dim=(518, 518),
    ):
        """RAD-DINO model for medical image classification.
        
        Parameters:
            n_classes (int): Number of output classes
            use_lora (bool): Whether to use LoRA fine-tuning
            r (int): LoRA rank parameter
            freeze_encoder (bool): Whether to freeze the encoder
            include_segmentation (bool): Whether to include segmentation head
            img_dim (tuple): Input image dimensions
        """
        super().__init__()
        self.img_dim = img_dim
        self.use_lora = use_lora
        self.include_segmentation = include_segmentation
        self.lora_r = r
        self.lora_alpha = alpha
        self.apply_mlp_lora = apply_mlp_lora
        
        # Load RAD-DINO model and processor
        self.rad_dino = AutoModel.from_pretrained("microsoft/rad-dino")
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
        
        # Get encoder dimensions
        self.encoder_dim = self.rad_dino.config.hidden_size  # 768 for base model
        
        # Freeze encoder if specified
        if freeze_encoder:
            self._freeze_encoder()
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(self.encoder_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )

        print("###########RADDINO##########")
        print(self.rad_dino)
        # print(dir(self.rad_dino))
        # for name, module in self.rad_dino.named_modules():
        #     print(name, module)

        
        # Apply LoRA if specified
        if self.use_lora:
            self._apply_lora(r)
            # Conditionally apply LoRA to MLP layers
            if self.apply_mlp_lora:
                self._apply_lora_to_mlp(r)
            
        # Segmentation decoder (if needed)
        if include_segmentation:
            from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
            self.decoder = UnetDecoder(
                encoder_channels=[self.encoder_dim, self.encoder_dim, self.encoder_dim, self.encoder_dim, self.encoder_dim],
                decoder_channels=[256, 128, 64, 32, 16],
                n_blocks=5,
                use_batchnorm=True,
                center=False,
                attention_type=None
            )
            # Segmentation head
            self.segmentation_head = nn.Conv2d(16, n_classes, kernel_size=1)
    
    def _freeze_encoder(self):
        """Freeze all encoder parameters"""
        for param in self.rad_dino.parameters():
            param.requires_grad = False
    
    def _apply_lora(self, r):
        """Apply LoRA to attention layers"""
        import math
        
        # Find attention blocks in RAD-DINO
        self.lora_layers = []
        self.w_a = []
        self.w_b = []
        
        # Apply LoRA to all transformer blocks in RAD-DINO
        for i, block in enumerate(self.rad_dino.encoder.layer):
            self.lora_layers.append(i)
            
            # Apply LoRA to query projection in self-attention
            q_proj = block.attention.attention.query
            dim = q_proj.in_features
            
            # Create LoRA layers for query
            w_a_q = nn.Linear(dim, r, bias=False)
            w_b_q = nn.Linear(r, dim, bias=False)
            
            # Initialize weights
            nn.init.kaiming_uniform_(w_a_q.weight, a=math.sqrt(5))
            nn.init.zeros_(w_b_q.weight)
            
            # Store LoRA layers
            self.w_a.append(w_a_q)
            self.w_b.append(w_b_q)
            
            # Apply LoRA to value projection
            v_proj = block.attention.attention.value
            
            # Create LoRA layers for value
            w_a_v = nn.Linear(dim, r, bias=False)
            w_b_v = nn.Linear(r, dim, bias=False)
            
            # Initialize weights
            nn.init.kaiming_uniform_(w_a_v.weight, a=math.sqrt(5))
            nn.init.zeros_(w_b_v.weight)
            
            # Store LoRA layers
            self.w_a.append(w_a_v)
            self.w_b.append(w_b_v)
            
            # Register modules
            for idx, (w_a, w_b) in enumerate(zip([w_a_q, w_a_v], [w_b_q, w_b_v])):
                self.add_module(f"w_a_{i}_{idx}", w_a)
                self.add_module(f"w_b_{i}_{idx}", w_b)
            
            # Modify forward pass for query and value projections
            original_q_forward = q_proj.forward
            original_v_forward = v_proj.forward
            
            # Create new forward methods
            def make_forward(original_forward, w_a, w_b, alpha, r):
                scale = alpha / r
                
                def lora_forward(x):
                    return original_forward(x) + scale * w_b(w_a(x)) # original_forward(x) + w_b(w_a(x))
                return lora_forward
            
            # Replace forward methods
            q_proj.forward = make_forward(original_q_forward, w_a_q, w_b_q, self.lora_alpha, r)
            v_proj.forward = make_forward(original_v_forward, w_a_v, w_b_v, self.lora_alpha, r)

    def _apply_lora_to_mlp(self, r):
        import math
        self.lora_mlp_modules = []
        for i, block in enumerate(self.rad_dino.encoder.layer):
            fc1 = block.mlp.fc1
            # stash original forward
            fc1._original_forward = fc1.forward

            # create adapters
            w_a_fc1 = nn.Linear(fc1.in_features, r, bias=False)
            w_b_fc1 = nn.Linear(r, fc1.out_features, bias=False)
            nn.init.kaiming_uniform_(w_a_fc1.weight, a=math.sqrt(5))
            nn.init.zeros_(w_b_fc1.weight)

            self.add_module(f"lora_mlp_w_a_{i}", w_a_fc1)
            self.add_module(f"lora_mlp_w_b_{i}", w_b_fc1)
            self.lora_mlp_modules.append((i, w_a_fc1, w_b_fc1))

            # wrap forward
            def make_lora_fc1_forward(original_forward, w_a, w_b, alpha, r):
                scale = alpha / r
                def lora_forward(x):
                    return original_forward(x) + scale * w_b(w_a(x))
                return lora_forward

            fc1.forward = make_lora_fc1_forward(
                fc1._original_forward, w_a_fc1, w_b_fc1, self.lora_alpha, r
            )
    
    def preprocess(self, x):
        """Convert input images to format expected by RAD-DINO"""
        # RAD-DINO expects 3-channel images
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # RAD-DINO was trained on 518x518 images
        if x.shape[2] != self.img_dim[0] or x.shape[3] != self.img_dim[1]:
            x = nn.functional.interpolate(
                x, size=self.img_dim, mode='bilinear', align_corners=True
            )
        
        return x
    
    def forward(self, x, output_attentions=False):
        # Store original size for segmentation output
        original_size = x.shape[2:]
        
        # Preprocess for RAD-DINO
        x_processed = self.preprocess(x)
        
        # Get RAD-DINO embeddings
        encoder_output = self.rad_dino(x_processed, output_hidden_states=True)
        # encoder_output = self.rad_dino(
        #     x_processed,
        #     output_hidden_states=True,
        #     output_attentions=output_attentions
        # )

        if output_attentions:
            return encoder_output
        
        # Classification path using CLS token
        cls_token = encoder_output.pooler_output
        cls_logits = self.classification_head(cls_token)
        
        # Return early if segmentation not requested
        if not self.include_segmentation:
            return cls_logits
        
        # Segmentation path using patch tokens
        patch_tokens = encoder_output.last_hidden_state[:, 1:]  # Skip CLS token
        
        # Reshape patch tokens to spatial grid
        seq_len = patch_tokens.shape[1]
        height = width = int(seq_len**0.5)
        patch_features = patch_tokens.reshape(
            patch_tokens.shape[0], height, width, self.encoder_dim
        ).permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Create multi-scale features for decoder
        features = [patch_features]
        for i in range(4):
            # Downsample previous feature map
            downsampled = nn.functional.avg_pool2d(features[-1], kernel_size=2, stride=2)
            features.append(downsampled)
        
        # Decoder and segmentation head
        decoder_features = self.decoder(*features)
        seg_logits = self.segmentation_head(decoder_features)
        
        # Resize to match input dimensions
        if seg_logits.shape[2:] != original_size:
            seg_logits = nn.functional.interpolate(
                seg_logits, size=original_size, 
                mode='bilinear', align_corners=True
            )
        
        # Return both task outputs
        return {
            "classification": cls_logits,
            "segmentation": seg_logits
        }
    
    def save_lora_weights(self, path):
        """Save LoRA weights to file"""
        if not self.use_lora:
            return
        
        weights_dict = {}
        
        # Save LoRA weights
        for i, layer_idx in enumerate(self.lora_layers):
            # Save query LoRA weights
            w_a_q = getattr(self, f"w_a_{layer_idx}_0")
            w_b_q = getattr(self, f"w_b_{layer_idx}_0")
            weights_dict[f"w_a_{layer_idx}_q"] = w_a_q.weight
            weights_dict[f"w_b_{layer_idx}_q"] = w_b_q.weight
            
            # Save value LoRA weights
            w_a_v = getattr(self, f"w_a_{layer_idx}_1")
            w_b_v = getattr(self, f"w_b_{layer_idx}_1")
            weights_dict[f"w_a_{layer_idx}_v"] = w_a_v.weight
            weights_dict[f"w_b_{layer_idx}_v"] = w_b_v.weight

        # Save MLP LoRA weights if applied
        if self.apply_mlp_lora:
            for i, (layer_idx, w_a_fc1, w_b_fc1) in enumerate(self.lora_mlp_modules):
                weights_dict[f"lora_mlp_w_a_{i}"] = w_a_fc1.weight
                weights_dict[f"lora_mlp_w_b_{i}"] = w_b_fc1.weight
                
        # Save classification head
        weights_dict.update({f"cls_head.{k}": v for k, v in self.classification_head.state_dict().items()})
        
        # Save segmentation head if present
        if self.include_segmentation:
            weights_dict.update({f"seg_decoder.{k}": v for k, v in self.decoder.state_dict().items()})
            weights_dict.update({f"seg_head.{k}": v for k, v in self.segmentation_head.state_dict().items()})
        
        # Save weights
        torch.save(weights_dict, path)
    
    def load_lora_weights(self, path, device):
        """Load LoRA weights from file"""
        if not self.use_lora:
            return
        if isinstance(path, dict):
            weights_dict = path
        else:
            weights_dict = torch.load(path, map_location=device)
        
        # Now load the LoRA weights.
        for i, layer_idx in enumerate(self.lora_layers):
            # Load query LoRA weights
            w_a_q = getattr(self, f"w_a_{layer_idx}_0")
            w_b_q = getattr(self, f"w_b_{layer_idx}_0")
            w_a_q.weight.data = weights_dict[f"w_a_{layer_idx}_q"]
            w_b_q.weight.data = weights_dict[f"w_b_{layer_idx}_q"]
            
            # Load value LoRA weights
            w_a_v = getattr(self, f"w_a_{layer_idx}_1")
            w_b_v = getattr(self, f"w_b_{layer_idx}_1")
            w_a_v.weight.data = weights_dict[f"w_a_{layer_idx}_v"]
            w_b_v.weight.data = weights_dict[f"w_b_{layer_idx}_v"]

        # Load MLP LoRA weights if applied
        if self.apply_mlp_lora:
            for i, (layer_idx, w_a_fc1, w_b_fc1) in enumerate(self.lora_mlp_modules):
                w_a_fc1.weight.data = weights_dict[f"lora_mlp_w_a_{i}"]
                w_b_fc1.weight.data = weights_dict[f"lora_mlp_w_b_{i}"]
        
        # Load classification head
        cls_head_dict = {k.replace("cls_head.", ""): v for k, v in weights_dict.items() if k.startswith("cls_head.")}
        self.classification_head.load_state_dict(cls_head_dict)
        
        # Load segmentation head if present
        if self.include_segmentation and any(k.startswith("seg_decoder.") for k in weights_dict):
            seg_decoder_dict = {k.replace("seg_decoder.", ""): v for k, v in weights_dict.items() if k.startswith("seg_decoder.")}
            self.decoder.load_state_dict(seg_decoder_dict)
            
            seg_head_dict = {k.replace("seg_head.", ""): v for k, v in weights_dict.items() if k.startswith("seg_head.")}
            self.segmentation_head.load_state_dict(seg_head_dict)

    def save_model_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_model_checkpoint(self, checkpoint, device):
        # If checkpoint is already a dict, use it directly.
        if isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            state_dict = torch.load(checkpoint, map_location=device)
        
        self.load_state_dict(state_dict)

#########################################################################################
## Not using due to conflicts and errors --> building custom-built lora ##
#########################################################################################
# use PEFT from HuggingFace
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
class RADDINOPeft_Model(nn.Module): # took out args in __init__
    def __init__(
        self,
        n_classes=1,
        use_peft=True,
        r=4,
        include_segmentation=False,
        img_dim=(518, 518)
    ):
        """RAD-DINO model with PEFT.
        
        Parameters:
            n_classes (int): Number of output classes
            use_peft (bool): Whether to use PEFT fine-tuning
            r (int): LoRA rank parameter
            include_segmentation (bool): Whether to include segmentation head
            img_dim (tuple): Input image dimensions
        """
        super().__init__()
        self.img_dim = img_dim
        self.use_peft = use_peft
        self.include_segmentation = include_segmentation
        
        # Load RAD-DINO model and processor
        self.rad_dino = AutoModel.from_pretrained("microsoft/rad-dino")
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
        
        # Get encoder dimensions
        self.encoder_dim = self.rad_dino.config.hidden_size  # 768 for base model
        
        # Apply PEFT if specified
        if self.use_peft:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=r,
                lora_alpha=16, # 32
                lora_dropout=0.1,
                target_modules=["query", "value"], # In RADDINO, linear layers inside the self‑attention are named "query" and "value" (not "q_proj" or "v_proj")
            )
            self.rad_dino = get_peft_model(self.rad_dino, peft_config)
        else:
            # Freeze encoder if not using PEFT
            for param in self.rad_dino.parameters():
                param.requires_grad = False
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(self.encoder_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )
        
        # Segmentation decoder (if needed)
        if include_segmentation:
            from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
            self.decoder = UnetDecoder(
                encoder_channels=[self.encoder_dim, self.encoder_dim, self.encoder_dim, self.encoder_dim, self.encoder_dim],
                decoder_channels=[256, 128, 64, 32, 16],
                n_blocks=5,
                use_batchnorm=True,
                center=False,
                attention_type=None
            )
            # Segmentation head
            self.segmentation_head = nn.Conv2d(16, n_classes, kernel_size=1)
    
    def preprocess(self, x):
        """Convert input images to format expected by RAD-DINO"""
        # RAD-DINO expects 3-channel images
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # RAD-DINO was trained on 518x518 images
        if x.shape[2] != self.img_dim[0] or x.shape[3] != self.img_dim[1]:
            x = nn.functional.interpolate(
                x, size=self.img_dim, mode='bilinear', align_corners=True
            )
        
        return x
    
    def forward(self, x):
        # Store original size for segmentation output
        original_size = x.shape[2:]
        
        # Preprocess for RAD-DINO
        x_processed = self.preprocess(x)
        
        # Get RAD-DINO embeddings
        encoder_output = self.rad_dino(pixel_values=x_processed, output_hidden_states=True)
        
        # Classification path using CLS token
        cls_token = encoder_output.pooler_output
        cls_logits = self.classification_head(cls_token)
        
        # Return early if segmentation not requested
        if not self.include_segmentation:
            return cls_logits
        
        # Segmentation path using patch tokens
        patch_tokens = encoder_output.last_hidden_state[:, 1:]  # Skip CLS token
        
        # Reshape patch tokens to spatial grid
        seq_len = patch_tokens.shape[1]
        height = width = int(seq_len**0.5)
        patch_features = patch_tokens.reshape(
            patch_tokens.shape[0], height, width, self.encoder_dim
        ).permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Create multi-scale features for decoder
        features = [patch_features]
        for i in range(4):
            # Downsample previous feature map
            downsampled = nn.functional.avg_pool2d(features[-1], kernel_size=2, stride=2)
            features.append(downsampled)
        
        # Decoder and segmentation head
        decoder_features = self.decoder(*features)
        seg_logits = self.segmentation_head(decoder_features)
        
        # Resize to match input dimensions
        if seg_logits.shape[2:] != original_size:
            seg_logits = nn.functional.interpolate(
                seg_logits, size=original_size, 
                mode='bilinear', align_corners=True
            )
        
        # Return both task outputs
        return {
            "classification": cls_logits,
            "segmentation": seg_logits
        }
    
    def save_model(self, path):
        """Save model weights"""
        if self.use_peft:
            # Save PEFT model
            self.rad_dino.save_pretrained(path)
            
            # Save classification head
            torch.save(self.classification_head.state_dict(), f"{path}/classification_head.pt")
            
            # Save segmentation components if present
            if self.include_segmentation:
                torch.save(self.decoder.state_dict(), f"{path}/segmentation_decoder.pt")
                torch.save(self.segmentation_head.state_dict(), f"{path}/segmentation_head.pt")
            else:
                # Save entire model
                torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        """Load model weights"""
        if self.use_peft:
            # Load PEFT model
            self.rad_dino = PeftModel.from_pretrained(self.rad_dino, path)
            
            # Load classification head
            self.classification_head.load_state_dict(torch.load(f"{path}/classification_head.pt"))
            
            # Load segmentation components if present
            if self.include_segmentation:
                self.decoder.load_state_dict(torch.load(f"{path}/segmentation_decoder.pt"))
                self.segmentation_head.load_state_dict(torch.load(f"{path}/segmentation_head.pt"))
        else:
            # Load entire model
            self.load_state_dict(torch.load(path))


