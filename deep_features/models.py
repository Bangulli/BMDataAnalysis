import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
from pathlib import Path

from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
def build_model_from_plans(
    plans_path: Path,
    dataset_json_path: Path,
    configuration: str = "3d_fullres",
    deep_supervision: bool = True,
):
    """
    Builds and returns a model based on the nnUNet plans file.

    Args:
        plans_path (Path): Path to the nnUNetPlans.json file.
        configuration (str): Configuration name to use from the plans. Default is "default".

    Returns:
        torch.nn.Module: The model built based on the plans.
    """
    plans_path = Path(plans_path)  # Ensure compatibility with Path
    if not plans_path.exists():
        raise ValueError(f"Invalid plans path: {plans_path}")

    # Load the plans file
    plans = load_json(str(plans_path))
    dataset_json = load_json(dataset_json_path)
    plans_manager = PlansManager(plans)

    # Get the configuration manager
    config_manager = plans_manager.get_configuration(configuration)

    # Determine the number of input channels
    num_input_channels = determine_num_input_channels(
        plans_manager,
        config_manager,
        dataset_json,
    )
    label_manager = plans_manager.get_label_manager(dataset_json)
    # Build and return the network
    return get_network_from_plans(
        config_manager.network_arch_class_name,
        config_manager.network_arch_init_kwargs,
        config_manager.network_arch_init_kwargs_req_import,
        num_input_channels,
        label_manager.num_segmentation_heads,
        allow_init=True,
        deep_supervision=deep_supervision,
    )
def build_encoder_decoder_from_plans(
    plans_path: Path,
    dataset_json_path: Path,
    configuration: str = "3d_fullres",
):
    """
    Builds and returns the encoder from an nnUNet model based on the plans file.

    Args:
        plans_path (Path): Path to the nnUNetPlans.json file.
        dataset_json_path (Path): Path to the dataset.json file.
        configuration (str): Configuration name to use from the plans.

    Returns:
        torch.nn.Module: The encoder extracted from the nnUNet model.
    """
    model = build_model_from_plans(plans_path, dataset_json_path, configuration, deep_supervision=False)
    
    # The model has an encoder/decoder attribute
    return model
def nnUNetBackbone(feature_dim=128, feature_dim_local=32, local_pool=10):
    """
    Builds the nnUNet encoder and adds a projection head.
    
    Args:
        feature_dim (int): The dimension of the feature space after the projection head.
        feature_dim_local (int): The dimension of the feature space after the local projection head.

    Returns:
        nn.Module: The nnUNet encoder with a projection head.
    """
    # Load the encoder/decoder from nnUNet plans
    model = build_encoder_decoder_from_plans(
        '/home/lorenz/BMDataAnalysis/deep_features/nnUNET/nnUNetPlans.json',
        "/home/lorenz/BMDataAnalysis/deep_features/nnUNET/dataset.json",
        configuration="3d_fullres"
    )
    # Add a projection head to the encoder
    class NNUNetWithProjection(nn.Module):
        def __init__(self, model, feature_dim, feature_dim_local=None, local_pool=10):
            super().__init__()
            self.encoder = model.encoder
            self.model = model
            self.encoder_projection = nn.Sequential(
                nn.Linear(encoder_output_dim, feature_dim*4), 
                nn.ReLU(),
                nn.Linear(feature_dim*4, feature_dim*2),
                nn.ReLU(),
                nn.Linear(feature_dim*2, feature_dim)
            )
            if feature_dim_local is not None:
                self.decoder_projection = nn.Sequential(
                    nn.Conv3d(in_channels=decoder_output_dim, out_channels=feature_dim_local*4, kernel_size=1), 
                    nn.ReLU(),
                    nn.Conv3d(in_channels=feature_dim_local*4, out_channels=feature_dim_local*2, kernel_size=1),
                    nn.ReLU(),
                    nn.Conv3d(in_channels=feature_dim_local*2, out_channels=feature_dim_local, kernel_size=1)
                )
            self.feature_output = None  # To store the hooked feature output
            self.hook_registered = False

        # def forward(self, x):
        #     """Forward through encoder and projection head."""
        #     x = self.encoder(x)[-1]  # Forward through nnUNet encoder
        #     x = self.avg_pool3d(x)
        #     x = torch.flatten(x, start_dim=1)
        #     x = self.encoder_projection(x)
        #     return F.normalize(x, dim=1)

        def forward(self, x):
            """Efficient forward pass through encoder and projection head."""
            x = self.encoder(x)[-1]  # Extract deepest feature map
            x = F.adaptive_avg_pool3d(x, 1)  
            x = x.squeeze(-1).squeeze(-1).squeeze(-1)  # Remove singleton dims
            x = self.encoder_projection(x)
            return F.normalize(x, dim=1)

        def forward_noproj(self, x):
            x = self.encoder(x)[-1]
            # x = self.avg_pool3d(x)  
            x = F.adaptive_avg_pool3d(x, 1)  
            x = torch.flatten(x, start_dim=1)
            return F.normalize(x, dim=1)


        def _hook_fn(self, module, input, output):
            self.feature_output = output  # Save the output of the hooked layer

        def register_hook(self):
            """Registers the hook once."""
            if not self.hook_registered:
                # self.model.decoder.transpconvs[-1].register_forward_hook(self._hook_fn)
                self.model.decoder.stages[-1].convs[-1].all_modules[-1].register_forward_hook(self._hook_fn)
                self.hook_registered = True

        def forward_decoder(self, x):
            original_size = x.shape[2:]  # Store original spatial dimensions (D, H, W)
            x = self._pad_if_needed(x)   # Ensure correct input size

            self.register_hook()  # Ensure hook is registered only once
            self.feature_output = None  # Reset before forward pass

            _ = self.model(x)  # Forward pass through the model
            if self.feature_output is None:
                raise RuntimeError("Hook did not capture any features. Check if the layer is correct.")

            features = self.feature_output  # Get the hooked feature map
            reconstructed_proj = self.decoder_projection(features)

            # Unpad the reconstructed projection to match the original input size
            reconstructed_proj = self._unpad_if_needed(reconstructed_proj, original_size)
            return reconstructed_proj
            
        def _pad_if_needed(self, x):
            """Pad input tensor if its dimensions are not divisible by the downsampling factor."""
            spatial_dims = x.shape[2:]  # Get spatial dimensions (D, H, W)
            downsampling_factor = 2 ** len(self.encoder.stages)  # Assuming 2x downsampling per stage
            pad_sizes = [(downsampling_factor - dim % downsampling_factor) % downsampling_factor 
                        for dim in spatial_dims]

            if any(pad_sizes):  # Apply padding only if needed
                padding = [p for size in pad_sizes[::-1] for p in (size // 2, size - size // 2)]
                self.pad_layer = nn.ConstantPad3d(padding, 0)  # Use ConstantPad3d for efficiency
                x = self.pad_layer(x)  # Apply padding
                self.pad_sizes = pad_sizes  # Store pad sizes for unpadding
            else:
                self.pad_sizes = [0, 0, 0]  # No padding needed
            return x

        def _unpad_if_needed(self, x, original_size):
            """Unpad the output tensor to match the original input size."""
            if any(self.pad_sizes):  # Unpad only if padding was applied
                for i, (pad, orig) in enumerate(zip(self.pad_sizes, original_size)):
                    start = pad // 2
                    x = x.narrow(dim=2 + i, start=start, length=orig)  # More efficient than slicing
            return x

    encoder_output_dim = model.encoder.stages[-1][0].convs[-1].conv.out_channels
    # decoder_output_dim = model.decoder.transpconvs[-1].out_channels 
    decoder_output_dim = model.decoder.stages[-1].convs[-1].conv.out_channels
    return NNUNetWithProjection(model, feature_dim, feature_dim_local, local_pool)

class GaussianSmoothing3D(nn.Module):
    def __init__(self, channels, kernel_size=5, sigma=1.0):
        super(GaussianSmoothing3D, self).__init__()
        
        # Create a 3D Gaussian kernel
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        
        # Convert to a 5D tensor (out_channels, in_channels, D, H, W)
        kernel = kernel.expand(channels, 1, kernel_size, kernel_size, kernel_size)
        
        # Define Conv3D layer (padding='same' ensures output size is the same)
        self.conv = nn.Conv3d(in_channels=channels, out_channels=channels, 
                              kernel_size=kernel_size, stride=1, padding=kernel_size//2, 
                              groups=channels, bias=False)
        
        # Load the Gaussian kernel into the Conv3D layer
        self.conv.weight.data = kernel
        self.conv.weight.requires_grad = False  # Freeze weights

    def _create_gaussian_kernel(self, kernel_size, sigma):
        """Create a 3D Gaussian kernel"""
        coords = torch.arange(kernel_size) - kernel_size // 2
        grid_x, grid_y, grid_z = torch.meshgrid(coords, coords, coords, indexing="ij")
        kernel = torch.exp(-(grid_x**2 + grid_y**2 + grid_z**2) / (2 * sigma**2))
        kernel /= kernel.sum()  # Normalize the kernel
        return kernel.view(1, 1, kernel_size, kernel_size, kernel_size)

    def forward(self, x):
        return self.conv(x)
    
class MOCO_local_global(nn.Module):
    def __init__(self, backbone, feature_dim=128, feature_dim_local=32, queue_size=65536, local_queue_size=262144, local_pool=10, momentum=0.999, use_global=True, use_local=False):
        super(MOCO_local_global, self).__init__()
        self.encoder_q = backbone(feature_dim, feature_dim_local, local_pool)
        self.encoder_k = backbone(feature_dim, feature_dim_local, local_pool)
        self.gaussian_blur = GaussianSmoothing3D(channels=feature_dim_local, sigma=0.5 * (1 / local_pool))
        self.gaussian_blur_mask = GaussianSmoothing3D(channels=1, sigma=0.5 * (1 / local_pool))

        # Initialize the momentum encoder with the same weights
        self.encoder_k.load_state_dict(self.encoder_q.state_dict())
        for param in self.encoder_k.parameters():
            param.requires_grad = False  # Momentum encoder is not trained directly

        if use_global:
            # Global queue
            self.register_buffer("queue", torch.randn(feature_dim, queue_size))
            self.queue = F.normalize(self.queue, dim=0)  # Normalize embeddings
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue_size = queue_size

        if use_local:
            # Local queue
            self.register_buffer("local_queue", torch.randn(feature_dim_local, local_queue_size))
            self.local_queue = F.normalize(self.local_queue, dim=0)  # Normalize embeddings
            self.register_buffer("local_queue_ptr", torch.zeros(1, dtype=torch.long))
            self.local_queue_size = local_queue_size
            self.local_pool = local_pool

        self.momentum = momentum
        self.use_local = use_local
        self.use_global = use_global

    def apply_smoothing_and_interp(self, recorder, data, gaussian_blur):
        # Data shape: N,C,H,W,D.
        # len(recorder) = N 
        # Apply Gaussian smoothing and interpolation to data
        # gaussian_blur = GaussianSmoothing3D(channels=data.shape[1], sigma=sigma).to(data.device)
        smoothed_data = torch.stack([
            F.interpolate(
                gaussian_blur(recorder[i].invert_transforms(data[i]).unsqueeze(0)),  # Ensure batch dim
                scale_factor=1 / self.local_pool, 
                mode='trilinear', 
                align_corners=False
            ).squeeze(0)  # Remove batch dim
            for i in range(len(data))
        ], dim=0).to(data.device)
        return smoothed_data

    @torch.no_grad()
    def update_momentum_encoder(self):
        """Update momentum encoder parameters."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    @torch.no_grad()
    def dequeue_and_enqueue_global(self, keys):
        """Update the global queue with the latest embeddings."""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # Ensure queue size is divisible by batch size

        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size  # Circular queue
        self.queue_ptr[0] = ptr

    def dequeue_and_enqueue_local(self, keys):
        """Update the local queue with the latest voxel-wise embeddings."""
        batch_size = keys.shape[0]
        ptr = int(self.local_queue_ptr)  # Current position in queue

        # Determine how many elements can be written before wrapping
        remaining = self.local_queue_size - ptr  

        if batch_size <= remaining:
            # No wrapping needed, directly insert the batch
            self.local_queue[:, ptr:ptr + batch_size] = keys.T
        else:
            # Wrap around: fill remaining slots, then continue from the beginning
            self.local_queue[:, ptr:] = keys.T[:, :remaining]
            self.local_queue[:, :batch_size - remaining] = keys.T[:, remaining:]

        # Update the pointer with wrapping
        ptr = (ptr + batch_size) % self.local_queue_size  
        self.local_queue_ptr[0] = ptr

    def forward(self, x_q, x_k, recorderq=None, recorderk=None):
        # Compute query features (main encoder)
        if self.use_global:
            q = self.encoder_q(x_q)  # Global contrastive learning
        if self.use_local:

            qd = self.encoder_q.forward_decoder(x_q)  # Local (voxel-wise) contrastive learning

            maskq = torch.ones_like(qd[:,0:1], dtype=torch.float32)
            # Apply the same transformations to the mask
            maskq = self.apply_smoothing_and_interp(recorderq, maskq, self.gaussian_blur_mask)

            maskq = (maskq > 0.75).float()  # Binarize mask

            # Apply smoothing and interpolation to query features
            qd = self.apply_smoothing_and_interp(recorderq, qd, self.gaussian_blur)
            # Reshape and normalize query features
            qd = qd.permute(0, 2, 3, 4, 1).reshape(-1, qd.size(1))  # Reshape to [NHWD, C]
            maskq = maskq.permute(0, 2, 3, 4, 1).reshape(-1, maskq.size(1))  # Reshape to [NHWD, C]
            qd = F.normalize(qd, dim=1)

        with torch.no_grad(): # gradients are needed when computing the query (q and qd), but not the key k and kd.
            self.update_momentum_encoder()  # Update momentum encoder
            if self.use_global:
                k = self.encoder_k(x_k)

            if self.use_local:
                kd = self.encoder_k.forward_decoder(x_k)  # Local (voxel-wise) contrastive learning

                maskk = torch.ones_like(kd[:,0:1], dtype=torch.float32)
                # Apply the same transformations to the mask
                maskk = self.apply_smoothing_and_interp(recorderk, maskk, self.gaussian_blur_mask)
                maskk = (maskk > 0.75).float()  # Binarize mask


                # Apply smoothing and interpolation to key features
                kd = self.apply_smoothing_and_interp(recorderk, kd, self.gaussian_blur)

                # Reshape and normalize key features
                kd = kd.permute(0, 2, 3, 4, 1).reshape(-1, kd.size(1))  # Reshape to [NHWD, C]
                maskk = maskk.permute(0, 2, 3, 4, 1).reshape(-1, maskk.size(1))  # Reshape to [NHWD, C]
                kd = F.normalize(kd, dim=1)

        if self.use_global:
            # Compute similarity scores for global contrastive learning
            positive_logits = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # Global positive logits
            negative_logits = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # Global negative logits

            # Concatenate logits for global contrastive learning
            logits = torch.cat([positive_logits, negative_logits], dim=1)
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)  # Positive pair index = 0

            # Temperature scaling
            logits /= 0.07  # Common default value for temperature in MOCO

        if self.use_local:
            # Combine masks to keep only valid locations (where both maskq and maskk are 1)
            valid_mask = (maskq * maskk).squeeze(1)  # Element-wise multiplication of both masks         
            # Apply the valid mask to query and key features (zeroing out invalid locations)
            qd = qd[valid_mask > 0]
            kd = kd[valid_mask > 0]

            # Compute similarity scores for local contrastive learning
            positive_logits_d = torch.einsum('nc,nc->n', [qd, kd]).unsqueeze(-1)  # Global positive logits
            negative_logits_d = torch.einsum('nc,ck->nk', [qd, self.local_queue.clone().detach()])  # Global negative logits
            
            # Concatenate logits for local contrastive learning
            logits_d = torch.cat([positive_logits_d, negative_logits_d], dim=1)
            labels_d = torch.zeros(logits_d.shape[0], dtype=torch.long).to(qd.device)  # Positive pair index = 0   
            logits_d /= 0.07    

        with torch.no_grad():
            # Dequeue and enqueue
            if self.use_global:
                self.dequeue_and_enqueue_global(k)
            else:
                logits = None
                labels = None
            if self.use_local:
                self.dequeue_and_enqueue_local(kd)
            else:
                logits_d = None
                labels_d = None


            return logits, labels, logits_d, labels_d

    def extract_features(self, volume, local=False):
        """Extract features up to the avg_pool3d layer or decoder."""
        with torch.no_grad():
            if local:
                return self.encoder_q.forward_decoder(volume)
            else:
                return self.encoder_q.forward_noproj(volume)
            
def build_encoder_from_plans(
    plans_path: Path,
    dataset_json_path: Path,
    configuration: str = "3d_fullres",
):
    """
    Builds and returns the encoder from an nnUNet model based on the plans file.

    Args:
        plans_path (Path): Path to the nnUNetPlans.json file.
        dataset_json_path (Path): Path to the dataset.json file.
        configuration (str): Configuration name to use from the plans.

    Returns:
        torch.nn.Module: The encoder extracted from the nnUNet model.
    """
    model = build_model_from_plans(plans_path, dataset_json_path, configuration, deep_supervision=False)
    
    # Assuming the model has an encoder attribute or similar access point
    encoder = model.encoder if hasattr(model, 'encoder') else model
    return encoder

def nnUNetEncoderBackbone(feature_dim=128):
    """
    Builds the nnUNet encoder and adds a projection head.
    
    Args:
        feature_dim (int): The dimension of the feature space after the projection head.

    Returns:
        nn.Module: The nnUNet encoder with a projection head.
    """
    # Load the encoder from nnUNet plans
    encoder = build_encoder_from_plans(
        '/home/lorenz/BMDataAnalysis/deep_features/nnUNET/nnUNetPlans.json',
        "/home/lorenz/BMDataAnalysis/deep_features/nnUNET/dataset.json",
        configuration="3d_fullres"
    )

    # Add a projection head to the encoder
    class NNUNetWithProjection(nn.Module):
        def __init__(self, encoder, feature_dim):
            super().__init__()
            self.encoder = encoder
            encoder_output_dim = encoder.stages[-1][0].convs[-1].conv.out_channels 
            self.avg_pool3d = nn.AdaptiveAvgPool3d(output_size=1)
            self.projection = nn.Sequential(
                nn.Linear(encoder_output_dim, feature_dim),  
                nn.ReLU(),
                nn.Linear(feature_dim, feature_dim)
            )

        def forward(self, x):
            x = self.encoder(x)[-1]  # Forward through nnUNet encoder
            x = self.avg_pool3d(x)  
            x = torch.flatten(x, start_dim=1)
            x = self.projection(x)
            return F.normalize(x, dim=1)

        def forward_noproj(self, x):
            #print(x.shape)
            x = self.encoder(x)[-1]
            #print(x.shape)
            x = self.avg_pool3d(x)  
            x = torch.flatten(x, start_dim=1)
            #print(x.shape)
            return F.normalize(x, dim=1)

    return NNUNetWithProjection(encoder, feature_dim)
          
class CNN3DClassifier(nn.Module):
    def __init__(self, feature_dim=128, num_classes=3, model_type="tiny", n_proj_layer=1):
        super(CNN3DClassifier, self).__init__()

        self.backbone = nnUNetEncoderBackbone(feature_dim)

        if n_proj_layer==1:
            self.fc_classification = nn.Linear(feature_dim, num_classes)
        elif n_proj_layer==2:
            self.fc_classification = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, num_classes)
            )
        else:
            raise ValueError(f"N_LAYER_PROJ should be 1 or 2, but is {n_proj_layer}.")


    def forward(self, x):
        features = self.backbone(x)
        logits = self.fc_classification(features)
        return logits
    
    def extract_features(self, x):
        return self.backbone.forward_noproj(x)
