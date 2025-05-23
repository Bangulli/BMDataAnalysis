import torch
RESUME_TRAINING = True
DATA_DIR = '/mnt/nas6/data/Target/mrct1000_iso_npy'
MODEL_TYPE = "nnunet" # tiny, medium, deep (not for local CL...), nnunet
USE_GLOBAL = True
USE_LOCAL = True
if USE_LOCAL and USE_GLOBAL:
    model_parts = "local_global"
elif USE_LOCAL:
    model_parts = "local"
elif USE_GLOBAL:
    model_parts = "global"
# METADATA_KEY="PatientID"
QUEUE_SIZE = 2048  
FEATURE_DIM = 32 # dim of the last projection (Linear(encoder_output_dim, feature_dim*4), Linear(feature_dim*4, feature_dim*2), Linear(feature_dim*2, feature_dim))
FEATURE_DIM_LOCAL = 8 
BATCH_SIZE = 4
NUM_EPOCHS = 100
CROP_SIZE = 112
CROP_DATALOADER = 156
LEARNING_RATE = 1e-4
STEP_KNN = 5
STEP_SAVE_MODEL = 1
LOCAL_POOL = 10

MODEL_CHECKPOINT_PATH = f'/home/lorenz/BMDataAnalysis/deep_features/cnn3d_nnunet_local_global_100ep_checkpoint.pth'
MAPPING_PATH = '/home/vincent/repos/ssl-bm/mapping/pretrain_class_mappings'

GPU_IDS = [1]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PROGRESS_TASKS = [  
    "SequenceCategory","Sequence", "Manufacturer", "PatientID", "PatientSex", #"ManufacturerModelName", 
    "Modality"#, "PatientWeight","PatientAge"
    ] # 



DISCARD_PATIENTS = [
    'sub-PAT-0351','sub-PAT-0863','sub-PAT-0039','sub-PAT-0292','sub-PAT-0054','sub-PAT-0578','sub-PAT-0183','sub-PAT-1032',
    'sub-PAT-0061','sub-PAT-0254','sub-PAT-0806','sub-PAT-0893','sub-PAT-1080','sub-PAT-0055','sub-PAT-0721','sub-PAT-0012',
    'sub-PAT-0220','sub-PAT-0840','sub-PAT-0268','sub-PAT-0995','sub-PAT-0114','sub-PAT-0496','sub-PAT-0029','sub-PAT-0585',
    'sub-PAT-0460','sub-PAT-1135','sub-PAT-0056','sub-PAT-0247','sub-PAT-0899'
    ]