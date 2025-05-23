import os
import math
# from config.config_pretrain_small import *
from .config_pretrain import *
from anatcl import AnatCL
import torch
from .models import nnUNetBackbone, MOCO_local_global, CNN3DClassifier

# def get_vincent_encoder():
#     torch.cuda.empty_cache()
#     torch.backends.cudnn.benchmark = True
#     os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

#     if not (USE_GLOBAL or USE_LOCAL):
#         raise ValueError(f"USE_LOCAL or USE_GLOBAL should be True.")

#     if RESUME_TRAINING and not os.path.exists(MODEL_CHECKPOINT_PATH):
#         raise ValueError(f"Trying to resume training but {MODEL_CHECKPOINT_PATH} does not exist.")


#     # Instantiate the backbone (e.g., CNN3DBackbone_tiny) 
#     momentum = 0.999

#     if MODEL_TYPE == "nnunet":
#         backbone = nnUNetBackbone

#     else:
#         raise ValueError(f"Unknown model_type '{MODEL_TYPE}'. Choose from 'tiny', 'med', 'deep' or 'nnunet.")

#     pooled_size = math.floor(CROP_DATALOADER/LOCAL_POOL)
#     local_queue_size = BATCH_SIZE*pooled_size*pooled_size*pooled_size*4 # batch_size*pooled_size^3 is the number of local samples if there is no padding
#     print(f"Global and local queue sizes: {QUEUE_SIZE}, {local_queue_size}")

#     model = MOCO_local_global(backbone=backbone, feature_dim=FEATURE_DIM, feature_dim_local=FEATURE_DIM_LOCAL, queue_size=QUEUE_SIZE, local_queue_size=local_queue_size, local_pool=LOCAL_POOL, momentum=0.999, use_global=USE_GLOBAL, use_local=USE_LOCAL)

#     if len(GPU_IDS) > 1: # Not working, to fix
#         model = torch.nn.DataParallel(model, device_ids=GPU_IDS)
#     model.device = DEVICE
#     model = model.to(DEVICE)

#     checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     return model

class ModelAdaptor():
    def __init__(self, model):
        self.model = model
    
    def extract_features(self, x):
        #print(x)
        x = self.model(x)
        #print(x)
        return x

def get_vincent_encoder():
    checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint['model_state_dict']

    # Adjust state_dict to match model keys if necessary
    adjusted_state_dict = {k.replace("encoder_q.", "backbone."): v for k, v in state_dict.items()}

    # Initialize model
    pretrained_model = CNN3DClassifier(
        feature_dim=FEATURE_DIM, num_classes=4, 
        model_type=MODEL_TYPE, n_proj_layer=1
    ).to(DEVICE)

    # Load state_dict into the model
    load_result = pretrained_model.load_state_dict(adjusted_state_dict, strict=False)

    # Log missing and unexpected keys
    if load_result.missing_keys:
        print("Missing keys (expected by model but not in loaded state_dict):")
        for key in load_result.missing_keys:
            print(f"  - {key}")

    if load_result.unexpected_keys:
        print("\nUnexpected keys (in loaded state_dict but not expected by model):")
        for key in load_result.unexpected_keys:
            print(f"  - {key}")

    # Log correctly loaded keys
    loaded_keys = set(adjusted_state_dict.keys()) - set(load_result.unexpected_keys)
    print("\nCorrectly loaded keys (present in both state_dict and model):")
    for key in sorted(loaded_keys):
        print(f"  - {key}")

    pretrained_model.device = DEVICE
    
    return pretrained_model

def get_anatCL_encoder(): #https://arxiv.org/abs/2408.07079
    model = AnatCL(descriptor="global", fold=0, pretrained=True)
    model = model.to("cuda")
    
    model.eval()
    model = ModelAdaptor(model)
    model.device = "cuda"
    return model