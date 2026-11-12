"""Validate provided files, cache uint8 images, and save a fixed 45k/5k split.

Uses only supplied competition files. No standard CIFAR download or label lookup.
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / 'data'  # dataset files, shared by both pipelines


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cache = ROOT / 'cache'
    cache.mkdir(exist_ok=True)
    split_dir = ROOT / 'splits'
    split_dir.mkdir(exist_ok=True)
    configuration = json.loads((ROOT / 'configs/a0_resnet18_baseline.json').read_text())
    source_hashes = {name: digest(DATA_DIR / name) for name in ('train_labels.csv','test.csv','sample_submission.csv')}
    meta_path = cache / 'data_manifest.json'
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        assert old['source_csv_sha256'] == source_hashes, 'Source CSV changed; inspect before regenerating data.'
        assert old['split_seed'] == configuration['split_seed']
        assert old['validation_per_class'] == configuration['validation_per_class']
        print('Existing validated cache retained. Remove only cache/ and splits/ if you intentionally change the data.', flush=True)
        return
    records = {}
    image_hashes = {}
    for split, filename, folder in [('train','train_labels.csv','train_images'),('test','test.csv','test_images')]:
        rows = list(csv.DictReader((DATA_DIR / filename).open(newline='')))
        assert len({r['id'] for r in rows}) == len(rows), f'Duplicate {split} IDs'
        assert len({r['filename'] for r in rows}) == len(rows)
        images = np.empty((len(rows),3,32,32),dtype=np.uint8)
        hashes = []
        for i,row in enumerate(rows):
            # Reject unexpected paths before reading outside the image directory.
            assert Path(row['filename']).name == row['filename']
            with Image.open(DATA_DIR/folder/row['filename']) as im:
                assert im.size == (32,32) and im.mode == 'RGB', (row['filename'],im.size,im.mode)
                arr = np.asarray(im)
                images[i] = arr.transpose(2,0,1)
                hashes.append(hashlib.sha256(arr.tobytes()).hexdigest())
        np.save(cache/f'{split}_images.npy',images)
        records[split] = rows
        image_hashes[split] = hashes
        print(f'Validated and cached {len(rows):,} {split} images',flush=True)
    train_rows = records['train']
    labels = np.array([int(r['label']) for r in train_rows],dtype=np.int64)
    assert set(labels) == set(range(10))
    # Exact duplicate grouping protects the validation split if duplicates appear.
    grouped = defaultdict(list)
    for i,h in enumerate(image_hashes['train']):
        grouped[h].append(i)
    for indices in grouped.values():
        assert len(set(labels[indices])) == 1, 'Identical training pixels have conflicting labels; inspect data.'
    rng = np.random.default_rng(configuration['split_seed'])
    val_indices = []
    for label in range(10):
        groups = [indices for indices in grouped.values() if labels[indices[0]] == label]
        rng.shuffle(groups)
        chosen = []
        for indices in groups:
            if len(chosen)+len(indices) <= configuration['validation_per_class']:
                chosen.extend(indices)
        assert len(chosen) == configuration['validation_per_class']
        val_indices.extend(chosen)
    val_indices = np.array(sorted(val_indices),dtype=np.int64)
    train_indices = np.setdiff1d(np.arange(len(labels)),val_indices)
    assert not (set(image_hashes['train'][i] for i in train_indices) & set(image_hashes['train'][i] for i in val_indices))
    np.save(cache/'labels.npy',labels)
    for split,indices in [('train',train_indices),('validation',val_indices)]:
        np.save(cache/f'{split}_indices.npy',indices)
        with (split_dir/f'{split}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['row_index','id','filename','label'])
            writer.writeheader()
            for i in indices:
                writer.writerow(dict(row_index=int(i),**train_rows[i]))
    # Pixel normalization is fit only on the 45,000-image training subset.
    images=np.load(cache/'train_images.npy',mmap_mode='r')
    total=np.zeros(3,dtype=np.float64)
    squares=np.zeros(3,dtype=np.float64)
    pixels=0
    for start in range(0,len(train_indices),1000):
        x=images[train_indices[start:start+1000]].astype(np.float64)/255
        total+=x.sum((0,2,3));squares+=(x*x).sum((0,2,3))
        pixels+=x.shape[0]*32*32
    mean=total/pixels
    std=np.sqrt(squares/pixels-mean**2)
    sample=list(csv.DictReader((DATA_DIR/'sample_submission.csv').open()))
    assert list(sample[0]) == ['id','label']
    assert [r['id'] for r in sample] == [r['id'] for r in records['test']]
    manifest=dict(source_csv_sha256=source_hashes,split_seed=configuration['split_seed'],
        validation_per_class=configuration['validation_per_class'],train_images=len(train_indices),
        validation_images=len(val_indices),test_images=len(records['test']),
        class_counts={str(k):int(v) for k,v in sorted(Counter(labels.tolist()).items())},
        train_mean=mean.tolist(),train_std=std.tolist(),
        exact_duplicate_training_groups=sum(len(v)>1 for v in grouped.values()),
        train_test_exact_pixel_overlap=len(set(image_hashes['train'])&set(image_hashes['test'])),
        split_sha256={s:digest(split_dir/f'{s}.csv') for s in ('train','validation')},
        cache_sha256={p.name:digest(p) for p in cache.glob('*.npy')})
    meta_path.write_text(json.dumps(manifest,indent=2))
    (split_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':
    main()
