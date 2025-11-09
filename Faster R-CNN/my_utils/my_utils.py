from collections import defaultdict
import numpy as np, torch, cv2, os, pathlib
from torchvision.ops import box_iou, nms


def inflate_boxes_xyxy(boxes, scale=1.5):
  c = (boxes[:, :2] + boxes[:, 2:]) / 2
  hw = (boxes[:, 2:] - boxes[:, :2]) / 2
  hw2 = hw * scale 
  out = boxes.clone()
  out[:, :2] = c - hw2
  out[:, 2:] = c + hw2
  return out    
     
@torch.no_grad()
def predict_tta(model, img, iou_thr=0.5):
    """Realiza predição com Test-Time Augmentation (flip horizontal) e aplica NMS final."""
    # normaliza caso ainda esteja uint8
    
    if img.dtype != torch.float32:
        img = img.float() / 255.0

    out1 = model([img])[0]
    img_h = torch.flip(img, dims=[2])  # flip horizontal
    out2 = model([img_h])[0]

    # desflipa boxes da segunda predição
    W = img.shape[2]
    boxes_h = out2["boxes"].clone()
    boxes_h[:, [0, 2]] = W - boxes_h[:, [2, 0]]

    # junta predições e faz NMS
    boxes = torch.cat([out1["boxes"], boxes_h])
    scores = torch.cat([out1["scores"], out2["scores"]])
    labels = torch.cat([out1["labels"], out2["labels"]])
    keep = nms(boxes, scores, iou_thr)

    return {
        "boxes": boxes[keep],
        "scores": scores[keep],
        "labels": labels[keep],
    }

@torch.no_grad()
def evaluate_and_visualize(model, loader, out_dir="/content/out",
                           score_thr=0.05, iou_thr=0.50, max_images_to_save=200,
                           debug_first_n=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    scores_by_c = defaultdict(list)
    tps_by_c    = defaultdict(list)
    fps_by_c    = defaultdict(list)
    npos_by_c   = defaultdict(int)
    seen_classes = set()
    saved = 0

    img_idx = 0
    for imgs, targets in loader:
        imgs_cuda = [im.to(device) for im in imgs]
        preds = [predict_tta(model, im) for im in imgs_cuda]
        preds = [{k: v.detach().cpu() for k,v in p.items()} for p in preds]
        tgts  = [{k: v.detach().cpu() for k,v in t.items()} for t in targets]

        for img_tensor, pred, tgt in zip(imgs, preds, tgts):
            img_idx += 1
            gt_boxes  = tgt["boxes"]
            gt_labels = tgt["labels"]

            # --- GT stats ---
            unique_gt = gt_labels.unique().tolist()
            for c in unique_gt:
                c = int(c)
                seen_classes.add(c)
                npos_by_c[c] += int((gt_labels == c).sum().item())

            # --- filtra preds por score ---
            keep = pred["scores"] >= score_thr
         
            pb = pred["boxes"][keep]
            pb = inflate_boxes_xyxy(pb, scale=1.5)
            pl = pred["labels"][keep]
            ps = pred["scores"][keep]

            # Dentro do evaluate_and_visualize, depois do filtro por score:


            # --- DEBUG opcional: primeiras N imagens ---
            if img_idx <= debug_first_n:
                print(f"[dbg img#{img_idx}] GT uniq={unique_gt} | #GT={len(gt_boxes)} | #pred(>=thr)={len(pb)}")
                if len(pl) > 0:
                    print(f"           preds uniq labels={sorted(set(pl.tolist()))} | min/max score=({float(ps.min()):.3f},{float(ps.max()):.3f})")

            # --- matching por classe ---
            # garante que avaliamos as classes presentes em GT ou preds
            classes_here = set(unique_gt) | set(pl.tolist())
            for c in classes_here:
                c = int(c)
                pb_c = pb[pl == c]
                ps_c = ps[pl == c]
                gt_c = gt_boxes[gt_labels == c]

                if len(pb_c) == 0:
                    continue

                # ordena por score (desc) para acumular corretamente
                order = torch.argsort(ps_c, descending=True)
                pb_c = pb_c[order]
                ps_c = ps_c[order]

                matched_gt = torch.zeros(len(gt_c), dtype=torch.bool)
                for j, box in enumerate(pb_c):
                    scores_by_c[c].append(float(ps_c[j]))

                    if len(gt_c) == 0:
                        # Não há GT dessa classe nesta imagem
                        tps_by_c[c].append(0); fps_by_c[c].append(1)
                        continue

                    # IoU confiável (espera [x1,y1,x2,y2])
                    ious = box_iou(box.unsqueeze(0), gt_c).squeeze(0)  # [num_gt]
                    best_iou, best_idx = (ious.max().item(), int(ious.argmax().item()))
                    if best_iou >= iou_thr and not matched_gt[best_idx]:
                        tps_by_c[c].append(1); fps_by_c[c].append(0)
                        matched_gt[best_idx] = True
                    else:
                        tps_by_c[c].append(0); fps_by_c[c].append(1)

                if img_idx <= debug_first_n:
                    tp_img = int(np.sum(tps_by_c[c][-len(pb_c):]))
                    fp_img = int(np.sum(fps_by_c[c][-len(pb_c):]))
                    print(f"           [cls={c}] GT={len(gt_c)} | pred={len(pb_c)} | matched_TP={tp_img} | FP={fp_img}")

            # ====== visualização ======
            if saved < max_images_to_save:
                im = img_tensor.detach().cpu()
                if im.dtype == torch.float32:
                    im = (im*255).clamp(0,255).byte()
                im = im.permute(1,2,0).numpy()  # HWC, RGB
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

                # GT (verde fino)
                for b in gt_boxes:
                    x1,y1,x2,y2 = map(int, b.tolist())
                    cv2.rectangle(im, (x1,y1), (x2,y2), (0,255,0), 1, lineType=cv2.LINE_AA)

                # Preds (verde forte)
                for b, lab, sc in zip(pb, pl, ps):
                    x1,y1,x2,y2 = map(int, b.tolist())
                    cv2.rectangle(im, (x1,y1), (x2,y2), (50,220,50), 2, lineType=cv2.LINE_AA)
                    txt = f"{int(lab.item())} {float(sc):.2f}"
                    cv2.putText(im, txt, (x1, max(y1-5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,220,50), 2, cv2.LINE_AA)

                out_path = os.path.join(out_dir, f"pred_{saved:05d}.jpg")
                cv2.imwrite(out_path, im)
                saved += 1

    def voc_ap(rec, prec):
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i-1] = np.maximum(mpre[i-1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[i+1] - mrec[i]) * mpre[i+1]))

    results = {}
    for c in sorted(seen_classes):
        scores = np.array(scores_by_c[c], dtype=np.float32)
        tps    = np.array(tps_by_c[c],    dtype=np.int32)
        fps    = np.array(fps_by_c[c],    dtype=np.int32)
        npos   = int(npos_by_c[c])

        if scores.size == 0 or npos == 0:
            results[c] = {"AP@0.50": 0.0, "npos": npos}
            continue

        order = np.argsort(-scores)
        tps = tps[order]; fps = fps[order]

        cum_tp = np.cumsum(tps)
        cum_fp = np.cumsum(fps)
        rec = cum_tp / max(npos, 1)
        prec = cum_tp / np.maximum(cum_tp + cum_fp, 1)

        results[c] = {"AP@0.50": voc_ap(rec, prec), "npos": npos}

    return results

def collate_fn(batch):
    imgs, targets = list(zip(*batch))
    return list(imgs), list(targets)

import cv2, numpy as np, torch, albumentations as A

def get_transforms(train=True, img_size=1024):
    if train:
        tfms = [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=(114,114,114)),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.10, rotate_limit=10,
                               border_mode=cv2.BORDER_CONSTANT, value=(114,114,114), p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.CLAHE(p=0.2),
        ]
    else:
        return None
    return A.Compose(
        tfms,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_visibility=0.0,
        )
    )

import matplotlib.pyplot as plt
import numpy as np
import torch, cv2

ID_TO_NAME = globals().get("ID_TO_NAME", {1: "Piscinas", 2: "Caixa D'água"})

def show_predictions(
    model,
    loader,
    device,
    n=2,
    score_thr=0.25,
    figsize=(12, 8)
):
    model.eval()
    shown = 0

    for imgs, targets in loader:
        imgs_cuda = [im.to(device) for im in imgs]
        with torch.no_grad():
            preds = model(imgs_cuda)

        preds = [{k: v.detach().cpu() for k,v in p.items()} for p in preds]
        tgts  = [{k: v.detach().cpu() for k,v in t.items()} for t in targets]

        for img_t, pred, tgt in zip(imgs, preds, tgts):
            if shown >= n:
                return

            im = img_t.detach().cpu()
            if im.dtype == torch.float32:
                im = (im * 255).clamp(0, 255).byte()
            im = im.permute(1, 2, 0).numpy()               
            im = np.ascontiguousarray(im, dtype=np.uint8)  

            gt_boxes  = tgt.get("boxes", torch.zeros((0,4)))
            gt_labels = tgt.get("labels", torch.zeros((0,), dtype=torch.int64))
            for b, lab in zip(gt_boxes, gt_labels):
                x1,y1,x2,y2 = map(int, b.tolist())
                cv2.rectangle(im, (x1,y1), (x2,y2), (0,255,0), 1, lineType=cv2.LINE_AA)

            keep = pred["scores"] >= score_thr
            pb = pred["boxes"][keep]
            pl = pred["labels"][keep]
            ps = pred["scores"][keep]
            for b, lab, sc in zip(pb, pl, ps):
                x1,y1,x2,y2 = map(int, b.tolist())
                cv2.rectangle(im, (x1,y1), (x2,y2), (50,220,50), 2, lineType=cv2.LINE_AA)
                txt = f"{ID_TO_NAME.get(int(lab.item()), str(int(lab.item())))} {float(sc):.2f}"
                cv2.putText(im, txt, (x1, max(y1-5,0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,220,50), 2, cv2.LINE_AA)

            plt.figure(figsize=figsize)
            plt.imshow(im)
            plt.axis("off")
            plt.title(f"GT (verde fino) | Preds (verde forte) | score ≥ {score_thr}")
            plt.show()

            shown += 1

import os, pathlib, numpy as np, torch, pandas as pd
from collections import defaultdict
from torchvision.ops import box_iou

@torch.no_grad()
def evaluate_ap_multi_iou(
    model,
    loader,
    device,
    score_thr=0.25,
    iou_list=(0.50, 0.75, 0.90),
    iou_grid=np.arange(0.50, 0.96, 0.05),   # para mAP COCO
    out_csv_path=None
):
    model.eval()

    # 1) Coleta única de GT e predições por imagem (evita re-inferir)
    cache = []             # lista de (gt_boxes, gt_labels, pb, pl, ps)
    all_classes = set()
    for imgs, targets in loader:
        imgs_cuda = [im.to(device) for im in imgs]
        outs = model(imgs_cuda)
        outs = [{k: v.detach().cpu() for k,v in o.items()} for o in outs]
        tgts = [{k: v.detach().cpu() for k,v in t.items()} for t in targets]

        for tgt, pred in zip(tgts, outs):
            gt_boxes  = tgt["boxes"]
            gt_labels = tgt["labels"]
            all_classes.update(gt_labels.unique().tolist())

            keep = pred["scores"] >= score_thr
            pb = pred["boxes"][keep]
            pl = pred["labels"][keep]
            ps = pred["scores"][keep]

            cache.append((gt_boxes, gt_labels, pb, pl, ps))

    classes = sorted(int(c) for c in all_classes) if all_classes else [1]

    def _ap_from_scores(scores, tps, fps, npos):
        if len(scores) == 0 or npos == 0:
            return 0.0
        order = np.argsort(-np.asarray(scores, dtype=np.float32))
        tps = np.asarray(tps, dtype=np.int32)[order]
        fps = np.asarray(fps, dtype=np.int32)[order]
        tp_cum = np.cumsum(tps)
        fp_cum = np.cumsum(fps)
        rec = tp_cum / max(npos, 1)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        # integração contínua (envelope)
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i-1] = max(mpre[i-1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
        return ap

    def _accumulate_for_iou(iou_thr):
        scores_by_c = defaultdict(list)
        tps_by_c    = defaultdict(list)
        fps_by_c    = defaultdict(list)
        npos_by_c   = defaultdict(int)

        for (gt_boxes, gt_labels, pb, pl, ps) in cache:
            # conta GT por classe
            for c in classes:
                npos_by_c[c] += int((gt_labels == c).sum().item())

            # por classe, match greedily por IoU
            for c in classes:
                pb_c = pb[pl == c]
                ps_c = ps[pl == c]
                gt_c = gt_boxes[gt_labels == c]

                if len(pb_c) == 0:
                    continue

                order = torch.argsort(ps_c, descending=True)
                pb_c = pb_c[order]
                ps_c = ps_c[order]

                matched_gt = torch.zeros(len(gt_c), dtype=torch.bool)
                for j, box in enumerate(pb_c):
                    scores_by_c[c].append(float(ps_c[j]))
                    if len(gt_c) == 0:
                        tps_by_c[c].append(0); fps_by_c[c].append(1)
                        continue
                    ious = box_iou(box.unsqueeze(0), gt_c).squeeze(0)  # [num_gt]
                    best_iou, best_idx = (ious.max().item(), int(ious.argmax().item()))
                    if best_iou >= iou_thr and not matched_gt[best_idx]:
                        tps_by_c[c].append(1); fps_by_c[c].append(0)
                        matched_gt[best_idx] = True
                    else:
                        tps_by_c[c].append(0); fps_by_c[c].append(1)

        ap_per_class = {}
        for c in classes:
            ap_per_class[c] = _ap_from_scores(
                scores_by_c[c], tps_by_c[c], fps_by_c[c], npos_by_c[c]
            )
        return ap_per_class, npos_by_c

    # 2) AP por IoU (0.50, 0.75, 0.90)
    table_rows = []
    aps_by_iou = {}
    npos_ref = None
    for thr in iou_list:
        ap_c, npos_c = _accumulate_for_iou(thr)
        aps_by_iou[thr] = ap_c
        if npos_ref is None:
            npos_ref = npos_c  # guarda #GT por classe

    # 3) mAP COCO (média de APs em 0.50:0.95)
    ap_grid_per_class = {c: [] for c in classes}
    for thr in iou_grid:
        ap_c, _ = _accumulate_for_iou(float(thr))
        for c in classes:
            ap_grid_per_class[c].append(ap_c.get(c, 0.0))
    map_coco_per_class = {c: float(np.mean(ap_grid_per_class[c])) if len(ap_grid_per_class[c]) else 0.0
                          for c in classes}

    # 4) Monta DataFrame (uma linha por classe)
    for c in classes:
        row = {
            "class_id": c,
            "AP@0.50": aps_by_iou.get(0.50, {}).get(c, 0.0),
            "AP@0.75": aps_by_iou.get(0.75, {}).get(c, 0.0),
            "AP@0.90": aps_by_iou.get(0.90, {}).get(c, 0.0),
            "mAP@[0.50:0.95]": map_coco_per_class.get(c, 0.0),
            "GT_objs": int(npos_ref.get(c, 0)) if npos_ref else 0
        }
        table_rows.append(row)

    df = pd.DataFrame(table_rows).sort_values(by="class_id").reset_index(drop=True)

    # 5) imprime
    print("==== Resultados por classe ====")
    for _, r in df.iterrows():
        print(
            f"Classe {int(r['class_id'])} | "
            f"mAP[.50:.95]={r['mAP@[0.50:0.95]']:.3f} | "
            f"AP@0.50={r['AP@0.50']:.3f} | AP@0.75={r['AP@0.75']:.3f} | AP@0.90={r['AP@0.90']:.3f} "
            f"(GT={int(r['GT_objs'])})"
        )

    # 6) salva CSV (se solicitado)
    if out_csv_path:
        pathlib.Path(os.path.dirname(out_csv_path)).mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv_path, index=False, sep=";")
        print(f"\n📊 CSV salvo em: {out_csv_path}")

    return df

