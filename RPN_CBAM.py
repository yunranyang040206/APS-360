# -*- coding: utf-8 -*-
"""RPN_with_CBMA.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1QHhVc4bZ_By98NndyVI9QrUk1pZlS8p4
"""

! pip install ijson

!unzip 'trainA_original_700.zip' # initial image tensors without resize

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Input image size (height, width)
ISIZE = (720, 1280)

# ImageNet statistics (for VGG16)
# imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
# imagenet_std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def normalize_tensor(img):
    """Normalize a tensor image (C, H, W) with values in [0,255]."""
    img = img / 255.0
    return img

def unnormalize_tensor(img):
    """Convert a normalized tensor back to a displayable numpy image."""
    img = img * 255.0
    return img.clamp(0, 255).byte().cpu().numpy()

# Global anchor parameters
ratios = [0.5, 1, 2]
anchor_scales = [8, 16, 32]

import ijson


def extract_first_n_labels(json_file_path, n):
    labels = []
    with open(json_file_path, 'rb') as f:
        parser = ijson.items(f, 'item')
        for i, item in enumerate(parser):
            if i >= n:
                break
            filtered_labels = [
                {"category": li.get("category"), "box2d": li.get("box2d")}
                for li in item.get("labels", []) if "box2d" in li
            ]
            labels.append({
                "name": item.get("name"),
                "timestamp": item.get("timestamp"),
                "labels": filtered_labels
            })
    return labels

def standardize_filename(path_or_name):
    base = os.path.basename(path_or_name)
    base, _ = os.path.splitext(base)
    return base

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, labels, pt_dir='pt_files'):
        self.image_dir = image_dir
        self.pt_dir = pt_dir
        os.makedirs(self.pt_dir, exist_ok=True)
        self.image_files = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
        self.label_dict = {}
        for item in labels:
            key = standardize_filename(item["name"])
            self.label_dict[key] = item

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        pt_path = os.path.join(
            self.pt_dir,
            os.path.basename(image_path)
            .replace('.jpg', '.pt')
            .replace('.png', '.pt')
            .replace('.jpeg', '.pt')
        )
        if os.path.exists(pt_path):
            image_tensor = torch.load(pt_path)
        else:
            image = Image.open(image_path).convert('RGB')
            if image.size != (ISIZE[1], ISIZE[0]):  # PIL: (width, height)
                image = image.resize((ISIZE[1], ISIZE[0]))
            image_tensor = transforms.PILToTensor()(image).float()
            torch.save(image_tensor, pt_path)
        image_tensor = normalize_tensor(image_tensor)

        base_key = standardize_filename(image_path)
        matched = self.label_dict.get(base_key, None)
        if matched is None or "labels" not in matched:
            target = {"boxes": torch.zeros((0, 4), dtype=torch.float32),
                      "labels": torch.zeros((0,), dtype=torch.int64),
                      "names": [],
                      "index": idx}
        else:
            boxes = []
            cats = []
            for obj in matched["labels"]:
                if "box2d" in obj:
                    b2d = obj["box2d"]
                    boxes.append([float(b2d["y1"]), float(b2d["x1"]),
                                  float(b2d["y2"]), float(b2d["x2"])])
                    cats.append(obj["category"])
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4), dtype=torch.float32)
            labels_tensor = torch.tensor([1] * len(cats), dtype=torch.int64)
            target = {"boxes": boxes_tensor, "labels": labels_tensor, "names": cats, "index": idx}
        return {"image": image_tensor, "boxes": target["boxes"], "labels": target["labels"],
                "names": target["names"], "index": target["index"]}

# Custom collate function (your version)
def custom_collate_fn(batch):
    images = [item["image"] for item in batch]
    boxes = [item["boxes"] for item in batch]
    labels = [item["labels"] for item in batch]
    names = [item["names"] for item in batch]
    indices = [item["index"] for item in batch]
    return {"images": torch.stack(images, 0),
            "boxes": boxes,
            "labels": labels,
            "names": names,
            "indices": indices}

def create_corner_rect(bb, color='red'):
    bb = np.array(bb, dtype=np.float32)
    return plt.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], color=color,
                         fill=False, lw=3)

def show_corner_bbs(im, bbs):
    # im expected to be (C, H, W) tensor; convert to numpy image for plotting
    im_np = unnormalize_tensor(im)
    plt.imshow(np.transpose(im_np, (1, 2, 0)))
    for bb in bbs:
        plt.gca().add_patch(create_corner_rect(bb))
    plt.show()

def create_ground_truth_rect(bb, color='blue'):
    # Ground truth boxes are already in the format: [x1, y1, x2, y2]
    bb = np.array(bb, dtype=np.float32)
    return plt.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1],
                         color=color, fill=False, lw=3)

def show_ground_truth_bbs(im, bbs):
    # im is expected to be a (C, H, W) tensor; convert it to a NumPy image for plotting
    im_np = unnormalize_tensor(im)
    plt.imshow(np.transpose(im_np, (1, 2, 0)))
    for bb in bbs:
        plt.gca().add_patch(create_ground_truth_rect(bb))
    plt.show()

# -----------------------
# Vectorized IoU Computation
# -----------------------

def compute_iou_vectorized(anchors, gt_boxes):
    """
    Compute IoU between anchors (N,4) and gt_boxes (M,4).
    Boxes in [y1,x1,y2,x2] format.
    Returns IoU matrix of shape (N, M).
    """
    anchors = anchors.astype(np.float32)
    gt_boxes = gt_boxes.astype(np.float32)
    inter_y1 = np.maximum(anchors[:, None, 0], gt_boxes[None, :, 0])
    inter_x1 = np.maximum(anchors[:, None, 1], gt_boxes[None, :, 1])
    inter_y2 = np.minimum(anchors[:, None, 2], gt_boxes[None, :, 2])
    inter_x2 = np.minimum(anchors[:, None, 3], gt_boxes[None, :, 3])
    inter_h = np.maximum(inter_y2 - inter_y1, 0)
    inter_w = np.maximum(inter_x2 - inter_x1, 0)
    inter_area = inter_h * inter_w
    anchor_area = (anchors[:,2]-anchors[:,0])*(anchors[:,3]-anchors[:,1])
    gt_area = (gt_boxes[:,2]-gt_boxes[:,0])*(gt_boxes[:,3]-gt_boxes[:,1])
    union = anchor_area[:,None] + gt_area[None,:] - inter_area
    iou = inter_area / union
    return iou

# -----------------------
# Revised bbox_generation Function (Vectorized and Padded)
# -----------------------

def bbox_generation(images, targets, X_FM, Y_FM):
    """
    Compute regression targets and classification labels for all anchors.
    All anchors (generated over the full feature map) receive a target;
    anchors outside the image are ignored (label = -1).
    Returns:
       anchor_locations_all_merge: (B, total_anchors, 4)
       anchor_labels_all_merge: (B, total_anchors)
       anchors: (total_anchors, 4)
    """
    num_batch = len(images)
    C, H_IMG, W_IMG = images[0].shape

    # Generate full grid anchors over feature map
    total_positions = X_FM * Y_FM
    num_anchor_per_pos = len(ratios) * len(anchor_scales)
    total_anchors = total_positions * num_anchor_per_pos

    # Generate grid centers (using strides)
    sub_sampling_x = float(W_IMG) / X_FM
    sub_sampling_y = float(H_IMG) / Y_FM

    shift_x = np.arange(sub_sampling_x, (X_FM+1)*sub_sampling_x, sub_sampling_x)
    shift_y = np.arange(sub_sampling_y, (Y_FM+1)*sub_sampling_y, sub_sampling_y)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)  # shape (Y_FM, X_FM)
    centers = np.stack([shift_y.ravel() - sub_sampling_y/2, shift_x.ravel() - sub_sampling_x/2], axis=1)  # (total_positions, 2)

    # For each center, generate anchors
    anchors = []
    for center in centers:
        cy, cx = center
        for ratio in ratios:
            for scale in anchor_scales:
                h = sub_sampling_y * scale * np.sqrt(ratio)
                w = sub_sampling_x * scale * np.sqrt(1.0/ratio)
                y1 = cy - h/2.
                x1 = cx - w/2.
                y2 = cy + h/2.
                x2 = cx + w/2.
                anchors.append([y1, x1, y2, x2])
    anchors = np.array(anchors, dtype=np.float32)  # shape (total_anchors, 4)

    # Create ground-truth arrays for all anchors (padded to total_anchors)
    # Initialize labels to -1 (ignore) and loc targets to zeros.
    anchor_labels_all = []
    anchor_locs_all = []
    pos_iou_threshold = 0.7
    neg_iou_threshold = 0.3
    n_sample = 256
    pos_ratio = 0.5

    # Compute valid indices: those anchors that are completely inside the image.
    valid_idx = np.where((anchors[:,0] >= 0) & (anchors[:,1] >= 0) &
                         (anchors[:,2] <= H_IMG) & (anchors[:,3] <= W_IMG))[0]

    for i in range(num_batch):
        # Create full target arrays for this image.
        labels = -1 * np.ones((anchors.shape[0],), dtype=np.int32)
        locs = np.zeros((anchors.shape[0], 4), dtype=np.float32)
        gt_boxes_tensor = targets[i]["boxes"]
        if gt_boxes_tensor.numel() > 0:
            gt_boxes = gt_boxes_tensor.cpu().numpy()  # shape (M,4)
            # Compute IoU for valid anchors only.
            valid_anchors = anchors[valid_idx]
            ious = compute_iou_vectorized(valid_anchors, gt_boxes)  # (N_valid, M)
            max_ious = np.max(ious, axis=1)
            argmax_ious = np.argmax(ious, axis=1)
            # Set targets for valid anchors.
            valid_labels = -1 * np.ones((valid_anchors.shape[0],), dtype=np.int32)
            valid_labels[max_ious >= pos_iou_threshold] = 1
            valid_labels[max_ious < neg_iou_threshold] = 0
            # Ensure every GT box gets at least one positive anchor.
            gt_max_ious = np.max(ious, axis=0)  # (M,)
            for j in range(gt_boxes.shape[0]):
                inds = np.where(ious[:, j] == gt_max_ious[j])[0]
                valid_labels[inds] = 1

            # Subsample positives and negatives in valid region.
            pos_inds = np.where(valid_labels == 1)[0]
            neg_inds = np.where(valid_labels == 0)[0]
            if len(pos_inds) > int(pos_ratio * n_sample):
                disable = np.random.choice(pos_inds, size=(len(pos_inds) - int(pos_ratio * n_sample)), replace=False)
                valid_labels[disable] = -1
            remaining = n_sample - np.sum(valid_labels == 1)
            if len(neg_inds) > remaining:
                disable = np.random.choice(neg_inds, size=(len(neg_inds) - remaining), replace=False)
                valid_labels[disable] = -1

            # Compute regression targets for positive valid anchors.
            valid_locs = np.zeros((valid_anchors.shape[0], 4), dtype=np.float32)
            pos_valid_inds = np.where(valid_labels == 1)[0]
            if len(pos_valid_inds) > 0:
                pos_anchors = valid_anchors[pos_valid_inds]
                anchor_heights = pos_anchors[:,2] - pos_anchors[:,0]
                anchor_widths = pos_anchors[:,3] - pos_anchors[:,1]
                anchor_ctr_y = pos_anchors[:,0] + 0.5 * anchor_heights
                anchor_ctr_x = pos_anchors[:,1] + 0.5 * anchor_widths

                target_gt = gt_boxes[argmax_ious[pos_valid_inds]]
                gt_heights = target_gt[:,2] - target_gt[:,0]
                gt_widths = target_gt[:,3] - target_gt[:,1]
                gt_ctr_y = target_gt[:,0] + 0.5 * gt_heights
                gt_ctr_x = target_gt[:,1] + 0.5 * gt_widths

                dy = (gt_ctr_y - anchor_ctr_y) / anchor_heights
                dx = (gt_ctr_x - anchor_ctr_x) / anchor_widths
                dh = np.log(gt_heights / anchor_heights)
                dw = np.log(gt_widths / anchor_widths)
                valid_locs[pos_valid_inds, :] = np.stack([dy, dx, dh, dw], axis=1)
            # Assign computed valid targets into the full arrays.
            labels[valid_idx] = valid_labels
            locs[valid_idx, :] = valid_locs
        # Append for current image.
        anchor_labels_all.append(labels)
        anchor_locs_all.append(locs)
    anchor_labels_all_merge = np.stack(anchor_labels_all, axis=0)  # (B, total_anchors)
    anchor_locs_all_merge = np.stack(anchor_locs_all, axis=0)        # (B, total_anchors, 4)
    return anchor_locs_all_merge, anchor_labels_all_merge, anchors

class CBAM(nn.Module):
    def __init__(self, channels, reduction=4, kernel_size=3):  # Reduced reduction ratio
        super().__init__()
        # Channel attention with more conservative reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels//reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        # Spatial attention with smaller kernel
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)

    def forward(self, x):
        # Channel attention
        b, c, _, _ = x.size()
        y_avg = self.avg_pool(x).view(b, c)
        y_max = self.max_pool(x).view(b, c)
        y = self.fc(y_avg) + self.fc(y_max)  # More efficient than conv implementation
        scale = self.sigmoid(y).view(b, c, 1, 1)
        x = x * scale.expand_as(x)

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(y))
        return x * scale

def visualize_attention(feat, title=""):
    """Visualize the attention maps from CBAM"""
    plt.figure(figsize=(10,5))
    # Channel attention
    avg_pool = torch.mean(feat, dim=1, keepdim=True)
    max_pool, _ = torch.max(feat, dim=1, keepdim=True)
    plt.subplot(1,2,1)
    plt.imshow(avg_pool[0].cpu().detach().numpy(), cmap='jet')
    plt.title(f"{title} - Channel Avg")
    plt.subplot(1,2,2)
    plt.imshow(max_pool[0].cpu().detach().numpy(), cmap='jet')
    plt.title(f"{title} - Channel Max")
    plt.show()

vgg_model = torchvision.models.vgg16(pretrained=True).to(device)
vgg_model.eval()
for param in vgg_model.features.parameters():
    param.requires_grad = False
req_features = [layer for layer in list(vgg_model.features)[:30]]

class EnhancedRPN(nn.Module):
    def __init__(self, in_channels=512, mid_channels=256, n_anchor=9):
        super(EnhancedRPN, self).__init__()
        # Convolutional layers with reduced capacity
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)

        # Improved CBAM modules
        self.cbam1 = CBAM(mid_channels, reduction=4, kernel_size=3)
        self.cbam2 = CBAM(mid_channels, reduction=4, kernel_size=3)

        # Output layers
        self.reg_layer = nn.Conv2d(mid_channels, n_anchor*4, kernel_size=1)
        self.cls_layer = nn.Conv2d(mid_channels, n_anchor*2, kernel_size=1)

        # Skip connection - always create but use identity if same dims
        self.skip_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        if in_channels == mid_channels:
            # If dimensions match, make it an identity mapping
            nn.init.eye_(self.skip_conv.weight)
            nn.init.zeros_(self.skip_conv.bias)
            self.skip_conv.weight.requires_grad = False
            self.skip_conv.bias.requires_grad = False

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in [self.conv1, self.conv2, self.conv3, self.reg_layer, self.cls_layer]:
            nn.init.normal_(layer.weight, std=0.01)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        # Skip conv already initialized appropriately

    def forward(self, x):
        residual = self.skip_conv(x)  # Always use skip_conv

        # First block with CBAM
        x = F.relu(self.conv1(x))
        x = self.cbam1(x)

        # Second block with CBAM
        x = F.relu(self.conv2(x))
        x = self.cbam2(x)

        # Final block with residual connection
        x = F.relu(self.conv3(x) + residual)

        # Predictions
        pred_anchor_locs = self.reg_layer(x)
        pred_cls_scores = self.cls_layer(x)

        # Reshape outputs
        batch_size = x.shape[0]
        pred_anchor_locs = pred_anchor_locs.permute(0,2,3,1).contiguous().view(batch_size, -1, 4)
        pred_cls_scores = pred_cls_scores.permute(0,2,3,1).contiguous().view(batch_size, -1, 2)

        # Apply tanh to regression outputs and softmax to classification
        pred_anchor_locs = torch.tanh(pred_anchor_locs) * 2
        objectness_score = F.softmax(pred_cls_scores, dim=-1)[:,:,1]

        return pred_anchor_locs, pred_cls_scores, objectness_score

# Initialize model
rpn_model = EnhancedRPN(in_channels=512, mid_channels=256).to(device)
optimizer = torch.optim.Adam(rpn_model.parameters(), lr=0.001, weight_decay=1e-4)

"""## Training/Validation Functions"""

def pred_bbox_to_xywh(bbox, anchors):
    anchors = anchors.astype(np.float32)
    anc_height = anchors[:,2] - anchors[:,0]
    anc_width  = anchors[:,3] - anchors[:,1]
    anc_ctr_y = anchors[:,0] + 0.5 * anc_height
    anc_ctr_x = anchors[:,1] + 0.5 * anc_width
    bbox_np = bbox.detach().cpu().numpy()
    dy = bbox_np[:,0]
    dx = bbox_np[:,1]
    dh = bbox_np[:,2]
    dw = bbox_np[:,3]
    ctr_y = dy * anc_height + anc_ctr_y
    ctr_x = dx * anc_width + anc_ctr_x
    h = np.exp(dh) * anc_height
    w = np.exp(dw) * anc_width
    roi = np.zeros(bbox_np.shape, dtype=np.float32)
    roi[:,0] = ctr_x - 0.5 * w
    roi[:,1] = ctr_y - 0.5 * h
    roi[:,2] = ctr_x + 0.5 * w
    roi[:,3] = ctr_y + 0.5 * h
    return roi


def train_epochs(req_features, rpn_model, optimizer, train_dl, epochs=20, rpn_lambda=10, iou_threshold=0.5, top_k=20):
    rpn_model.train()
    epoch_train_recalls = []  # Track recall instead of error
    epoch_train_errors = []   # Still keep error for backward compatibility

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        total_samples = 0
        sum_loss = 0.0
        sum_loss_cls = 0.0
        sum_loss_loc = 0.0
        batch_recalls = []  # Track recall per batch

        for batch in train_dl:
            images = batch["images"].to(device)
            targets = [{"boxes": b, "labels": l} for b, l in zip(batch["boxes"], batch["labels"])]
            B = images.shape[0]
            total_samples += B

            # Forward through frozen backbone
            imgs = images.clone()
            with torch.no_grad():
                feat = imgs
                for m in req_features:
                    feat = m(feat)
            X_FM, Y_FM = feat.shape[2], feat.shape[3]

            # Compute GT targets
            gt_locs_np, gt_scores_np, anchors = bbox_generation([img for img in images], targets, X_FM, Y_FM)
            gt_locs = torch.from_numpy(gt_locs_np.astype(np.float32)).to(device)
            gt_scores = torch.from_numpy(gt_scores_np.astype(np.float32)).to(device)

            # Forward RPN
            pred_locs, pred_scores, objectness_score = rpn_model(feat)

            # Compute losses
            cls_loss = F.cross_entropy(pred_scores.view(-1, 2),
                                     gt_scores.view(-1).long(),
                                     ignore_index=-1)

            pos_mask = gt_scores > 0
            if pos_mask.sum() > 0:
                pred_pos = pred_locs[pos_mask]
                gt_pos = gt_locs[pos_mask]
                diff = torch.abs(gt_pos - pred_pos)
                loc_loss = torch.where(diff < 1, 0.5 * diff**2, diff - 0.5)
                loc_loss = loc_loss.sum() / pos_mask.sum().float()
            else:
                loc_loss = torch.tensor(0.0, device=device)

            loss = cls_loss + rpn_lambda * loc_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate losses
            sum_loss += loss.item()
            sum_loss_cls += cls_loss.item()
            sum_loss_loc += (rpn_lambda * loc_loss).item()

            with torch.no_grad():
                batch_recall = 0.0
                count = 0
                for i in range(B):
                    rois = pred_bbox_to_xywh(pred_locs[i], anchors)
                    if top_k is not None:
                        k = min(top_k, objectness_score[i].shape[0])
                        top_k_idx = torch.topk(objectness_score[i], k=k).indices
                        proposals = rois[top_k_idx.cpu().numpy()]

                    gt_boxes = batch["boxes"][i]
                    if not isinstance(gt_boxes, np.ndarray):
                        gt_boxes = gt_boxes.cpu().numpy()

                    if len(gt_boxes) > 0:
                        matched = 0
                        for gt in gt_boxes:
                            gt_converted = np.array([gt[1], gt[0], gt[3], gt[2]])
                            ious = compute_iou_vectorized(proposals, np.expand_dims(gt_converted, axis=0))
                            best_iou = np.max(ious) if ious.size > 0 else 0.0
                            if best_iou >= iou_threshold:
                                matched += 1
                        recall = matched / len(gt_boxes)
                        batch_recall += recall
                        count += 1

                if count > 0:
                    batch_recalls.append(batch_recall / count)

        # Store epoch metrics
        epoch_recall = np.mean(batch_recalls) if batch_recalls else 0.0
        epoch_train_recalls.append(epoch_recall)
        epoch_train_errors.append(1 - epoch_recall)  # For compatibility

        print(f"Epoch {epoch+1}: Loss {sum_loss/total_samples:.3f} | "
              f"Recall: {epoch_recall:.3f} | Error: {1-epoch_recall:.3f}")

        if (epoch+1) % 5 == 0:
            torch.save(rpn_model.state_dict(), f"./rpn_epoch_{epoch+1}.pth")

    # Improved plotting - show both recall and error trends
    plt.figure(figsize=(12, 5))

    # Plot 1: Recall
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), epoch_train_recalls, 'b-o', label='Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Training Recall Over Epochs')
    plt.grid(True)

    # Plot 2: Error (1 - Recall)
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), epoch_train_errors, 'r-o', label='Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error (1 - Recall)')
    plt.title('Training Error Over Epochs')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return rpn_model


def validate(rpn_model, data_loader, n_images=7, top_k=20, iou_threshold=0.5):
    rpn_model.eval()
    errors = []    # List to store error (1 - recall) per image
    recalls = []   # List to store recall per image
    avg_ious = []  # Track average best IoU per image

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= 1:  # Only process one batch for validation
                break

            images = batch["images"][:n_images].to(device)
            targets = [{"boxes": b, "labels": l} for b, l in zip(batch["boxes"], batch["labels"])][:n_images]

            # Forward pass through backbone features
            imgs = images.clone()
            for m in req_features:
                imgs = m(imgs)
            X_FM, Y_FM = imgs.shape[2], imgs.shape[3]
            _, _, anchors = bbox_generation([img for img in images], targets, X_FM, Y_FM)
            pred_locs, pred_scores, objectness_score = rpn_model(imgs)

            for i in range(min(n_images, images.shape[0])):
                # Get proposals for image i (output in [y1, x1, y2, x2])
                rois = pred_bbox_to_xywh(pred_locs[i], anchors)
                if top_k is not None:
                    k = min(top_k, objectness_score[i].shape[0])
                    top_k_idx = torch.topk(objectness_score[i], k=k).indices
                    proposals = rois[top_k_idx.cpu().numpy()]
                    print(f"\nImage {i}: Showing top {k} proposals")
                else:
                    proposals = rois

                # Plot proposals and ground truth
                show_corner_bbs(images[i], proposals)
                gt_boxes = targets[i]["boxes"].cpu().numpy()  # Format: [x1, y1, x2, y2]

                if len(gt_boxes) > 0:
                    show_ground_truth_bbs(images[i], gt_boxes)

                    # Compute metrics
                    matched = 0
                    image_ious = []
                    for gt in gt_boxes:
                        # Convert GT box from [x1, y1, x2, y2] to [y1, x1, y2, x2]
                        gt_converted = np.array([gt[1], gt[0], gt[3], gt[2]])
                        ious = compute_iou_vectorized(proposals, np.expand_dims(gt_converted, axis=0))
                        best_iou = np.max(ious) if ious.size > 0 else 0.0
                        image_ious.append(best_iou)
                        if best_iou >= iou_threshold:
                            matched += 1

                    recall = matched / len(gt_boxes)
                    avg_iou = np.mean(image_ious)
                    error = 1 - recall

                    recalls.append(recall)
                    errors.append(error)
                    avg_ious.append(avg_iou)

                    print(f"Image {i} Metrics:")
                    print(f"- Recall: {recall:.3f}")
                    print(f"- Error (1-Recall): {error:.3f}")
                    print(f"- Avg Best IoU: {avg_iou:.3f}")
                    print(f"- GT Boxes: {len(gt_boxes)}")
                    print(f"- Proposals: {len(proposals)}")
                else:
                    print(f"Image {i}: No ground truth boxes available")

    # Improved validation plotting
    if errors:
        plt.figure(figsize=(15, 5))

        # Plot 1: Recall
        plt.subplot(1, 3, 1)
        plt.plot(range(len(recalls)), recalls, 'b-o')
        plt.ylim(0, 1.05)
        plt.xlabel('Image Index')
        plt.ylabel('Recall')
        plt.title(f'Recall (IoU ≥ {iou_threshold})')
        plt.grid(True)

        # Plot 2: Error
        plt.subplot(1, 3, 2)
        plt.plot(range(len(errors)), errors, 'r-o')
        plt.ylim(0, 1.05)
        plt.xlabel('Image Index')
        plt.ylabel('Error (1-Recall)')
        plt.title('Error per Image')
        plt.grid(True)

        # Plot 3: Average IoU
        plt.subplot(1, 3, 3)
        plt.plot(range(len(avg_ious)), avg_ious, 'g-o')
        plt.ylim(0, 1.05)
        plt.xlabel('Image Index')
        plt.ylabel('Avg Best IoU')
        plt.title('Proposal Quality')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    # Print summary statistics
    if recalls:
        print("\nValidation Summary:")
        print(f"Mean Recall: {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
        print(f"Mean Error: {np.mean(errors):.3f} ± {np.std(errors):.3f}")
        print(f"Mean Avg IoU: {np.mean(avg_ious):.3f} ± {np.std(avg_ious):.3f}")

    rpn_model.train()
    return errors, recalls, avg_ious

"""## Prepare Dataset for Training"""

image_dir = 'trainA_original_700'
pt_dir = 'trainA_testing2'
json_file_path = 'bdd100k_labels_images_train.json'

# Extract labels from JSON (adjust number as desired)
all_labels = extract_first_n_labels(json_file_path, 20000)

# Create the custom dataset using your method
dataset = CustomDataset(image_dir, all_labels, pt_dir)

# Split using random_split (70% train, 15% val, 15% test)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Testing dataset size: {len(test_dataset)}")

# Create DataLoaders using your custom collate function
batch_size = 8
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=2)
val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=2)

"""## Training Test"""

small_train_dataset = torch.utils.data.Subset(train_dataset, list(range(100)))
small_train_loader = torch.utils.data.DataLoader(small_train_dataset, batch_size=batch_size, shuffle=True,
                                           collate_fn=custom_collate_fn, num_workers=2)

# Train the RPN using the training DataLoader
trained_rpn = train_epochs(req_features, rpn_model, optimizer, small_train_loader, epochs=30, rpn_lambda=10)

# Validate (visualize predictions) on both training and validation sets
print("Validation on training data:")
validate(trained_rpn, train_loader)
print("Validation on validation data:")
validate(trained_rpn, val_loader)