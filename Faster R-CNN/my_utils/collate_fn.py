def collate_fn(batch):
    imgs, targets = list(zip(*batch))
    return list(imgs), list(targets)