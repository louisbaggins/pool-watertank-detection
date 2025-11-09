import json, cv2, numpy as np, torch, albumentations as A
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

DATASET_ROOT = Path("/content/drive/MyDrive/TCC/DATASET")

class CocoDetDataset(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None, category_id=1, class_id_out=1):
        """
        category_id: id da classe (1 para 'pool' OU 1 para 'water_tank' em datasets separados).
        """
        self.img_dir = Path(img_dir)
        self.coco = COCO(str(ann_file))
        self.img_ids = list(self.coco.imgs.keys())
        self.transforms = transforms
        self.category_id = category_id
        self.class_id_out = class_id_out

    def __len__(self): return len(self.img_ids)

    def __getitem__(self, idx):

      img_id = self.img_ids[idx]
      info = self.coco.loadImgs([img_id])[0]
      img_path = self.img_dir / info["file_name"]

      img = cv2.imread(str(img_path))
      if img is None:
          raise FileNotFoundError(f"Imagem não encontrada: {img_path}")
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # HWC, uint8
      h, w = img.shape[:2]

      ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=False)
      anns = self.coco.loadAnns(ann_ids)

      boxes, labels = [], []
      for a in anns:
          if a.get("category_id", self.category_id) != self.category_id:
              continue

          bbox = a.get("bbox", None)
          if bbox is None:
              seg = a.get("segmentation", [])
              if not seg:
                  continue
              seg = np.array(seg[0], dtype=np.float32).reshape(-1, 2)
              x1, y1 = seg[:, 0].min(), seg[:, 1].min()
              x2, y2 = seg[:, 0].max(), seg[:, 1].max()
              bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
          x, y, bw, bh = bbox
          x1, y1, x2, y2 = x, y, x + bw, y + bh

          # clip e valida
          x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
          y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))
          if x2 <= x1 or y2 <= y1:
              continue

          boxes.append([x1, y1, x2, y2])
          labels.append(self.class_id_out)  # única classe neste dataset

      # ---- tensores com shapes certos, inclusive para vazio ----
      if len(boxes) == 0:
          boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
          labels_t = torch.zeros((0,),   dtype=torch.int64)
      else:
          boxes_t  = torch.as_tensor(boxes,  dtype=torch.float32)
          labels_t = torch.as_tensor(labels, dtype=torch.int64)

      target = {
          "boxes":    boxes_t.clone(),   # usa *estes* (não recria do zero)
          "labels":   labels_t.clone(),
          "image_id": torch.tensor([img_id]),
          # "area" e "iscrowd" não são necessários para treino no torchvision
      }

      # ---- Albumentations (em NumPy uint8) ----
      img_np = img
      if self.transforms:
          bxs_in = boxes_t.numpy().tolist() if boxes_t.numel() else []
          cls_in = labels_t.numpy().tolist() if labels_t.numel() else []

          t = self.transforms(image=img_np, bboxes=bxs_in, class_labels=cls_in)
          img_np = t["image"]

          bxs_out = t.get("bboxes", [])
          cls_out = t.get("class_labels", [])

          # blindagem: se vier bool, troca por lista vazia
          if isinstance(bxs_out, (bool, np.bool_)): bxs_out = []
          if isinstance(cls_out, (bool, np.bool_)): cls_out = []

          if len(bxs_out) == 0:
              target["boxes"]  = torch.zeros((0, 4), dtype=torch.float32)
              target["labels"] = torch.zeros((0,),   dtype=torch.int64)
          else:
              target["boxes"]  = torch.as_tensor(bxs_out, dtype=torch.float32)
              target["labels"] = torch.as_tensor(cls_out, dtype=torch.int64)

      # ---- imagem: SEMPRE float32 em [0,1], CHW ----
      img_np = img_np.astype(np.float32)
      if img_np.max() > 1.0:
          img_np /= 255.0
      if img_np.ndim == 2:
          img_np = np.repeat(img_np[..., None], 3, axis=2)

      img_t = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()  # CHW, float32
      return img_t, target





