
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from torch.utils.data import DataLoader, DistributedSampler

# ── SageMaker Data Parallel ───────────────────────────────────────────────────
import smdistributed.dataparallel.torch.distributed as smdist
from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as SmDDP

smdist.init_process_group()
local_rank = smdist.get_local_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

# ── Hyperparameters (passed in by SageMaker estimator) ───────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--epochs',     type=int,   default=120)
parser.add_argument('--lr',         type=float, default=1e-4)
parser.add_argument('--batch_size', type=int,   default=16)
parser.add_argument('--num_classes',type=int,   default=13)
# SageMaker injects these automatically
parser.add_argument('--model_dir',  type=str,   default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
parser.add_argument('--train',      type=str,   default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
parser.add_argument('--val',        type=str,   default=os.environ.get('SM_CHANNEL_VAL',   '/opt/ml/input/data/val'))
args = parser.parse_args()

# ── Checkpoint path (SageMaker injects SM_CHECKPOINT_DIR for spot recovery) ──
checkpoint_dir  = os.environ.get('SM_CHECKPOINT_DIR', '/opt/ml/checkpoints')
checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
os.makedirs(checkpoint_dir, exist_ok=True)

# ── Paste your full model architecture here (copy from goobie5000.ipynb) ─────
# PatchEmbed3D, Attention, MLP, Block, TemporalViTEncoder,
# LNWrapper, FPNBlock, ConvTransformerNeck, ConvModule,
# FCNHead, AuxFCNHead, PrithviCropClassifier
# (copy all class definitions from Step 5A of your notebook)

# ── Build model ───────────────────────────────────────────────────────────────
model = PrithviCropClassifier(num_classes=args.num_classes).to(device)
model = SmDDP(model)  # Wrap with SageMaker Data Parallel

# ── Optimizer ─────────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
start_epoch = 0

# ── Resume from checkpoint if spot instance was interrupted ──────────────────
if os.path.exists(checkpoint_path):
    print(f"[RESUME] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    print(f"[RESUME] Resuming from epoch {start_epoch}")

# ── Dataset (replace with your actual geospatial dataset class) ──────────────
# Your dataset should load HLS/NAIP chips from args.train and args.val
# Example: dataset = PrithviGeospatialDataset(args.train)
dataset = YourGeospatialDataset(args.train)  # ← replace with your dataset class

sampler    = DistributedSampler(
    dataset,
    num_replicas=smdist.get_world_size(),
    rank=smdist.get_rank()
)
dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

# ── Training loop ─────────────────────────────────────────────────────────────
for epoch in range(start_epoch, args.epochs):
    sampler.set_epoch(epoch)  # Critical: ensures different shuffle per epoch per GPU
    model.train()
    epoch_loss = 0.0

    for batch in dataloader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)

    # ── Save checkpoint every epoch (rank 0 only to avoid file conflicts) ────
    if smdist.get_rank() == 0:
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.module.state_dict(),  # .module unwraps SmDDP
            'optimizer_state_dict': optimizer.state_dict(),
            'loss':                 avg_loss,
        }, checkpoint_path)
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | Checkpoint saved.")

# ── Save final model artifacts (SageMaker uploads SM_MODEL_DIR to S3) ─────────
if smdist.get_rank() == 0:
    final_model_path = os.path.join(args.model_dir, 'prithvi_crop_classifier.pt')
    torch.save(model.module.state_dict(), final_model_path)
    print(f"[OK] Final model saved to: {final_model_path}")

