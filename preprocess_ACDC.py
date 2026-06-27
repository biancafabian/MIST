"""
Preprocess raw ACDC dataset for MIST training.

Raw ACDC structure (from the challenge download):
  <RAW_DIR>/training/
    patient001/
      patient001_frame01.nii.gz       <- image at ED frame
      patient001_frame01_gt.nii.gz    <- label at ED frame
      patient001_frame14.nii.gz       <- image at ES frame
      patient001_frame14_gt.nii.gz    <- label at ES frame
      Info.cfg                        <- contains "ED: 1" and "ES: 14" lines
    patient002/
      ...
    patient100/
      ...

Output structure:
  <OUT_DIR>/
    train/        2D slices .npz  keys: 'img' (H,W), 'label' (H,W)
    valid/        2D slices .npz  keys: 'img' (H,W), 'label' (H,W)
    test/         3D volumes .npz keys: 'img' (S,H,W), 'label' (S,H,W)
    lists_ACDC/
      train.txt
      valid.txt
      test.txt

Split (TransUNet / MIST standard):
  Train : patient001 - patient070  (70 patients)
  Valid : patient071 - patient080  (10 patients)
  Test  : patient081 - patient100  (20 patients)

Normalisation: per-volume min-max to [0, 1].
Both ED and ES frames are included for train/valid.
Test saves each frame as a separate 3D volume .npz.

Usage:
  python preprocess_ACDC.py --raw_dir /path/to/ACDC_raw/training --out_dir ./data/ACDC
"""

import os
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm


TRAIN_IDS = list(range(1,  71))   # patients 001-070
VALID_IDS = list(range(71, 81))   # patients 071-080
TEST_IDS  = list(range(81, 101))  # patients 081-100


def read_ed_es(patient_dir):
    """Parse Info.cfg and return (ed_frame, es_frame) as ints."""
    cfg = os.path.join(patient_dir, 'Info.cfg')
    ed = es = None
    with open(cfg) as fh:
        for line in fh:
            if line.startswith('ED:'):
                ed = int(line.split(':')[1].strip())
            elif line.startswith('ES:'):
                es = int(line.split(':')[1].strip())
    assert ed is not None and es is not None, f'Could not parse ED/ES from {cfg}'
    return ed, es


def normalize_volume(vol):
    """Min-max normalise a float32 volume to [0, 1]."""
    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax - vmin < 1e-8:
        return np.zeros_like(vol, dtype=np.float32)
    return ((vol - vmin) / (vmax - vmin)).astype(np.float32)


def process_patient(patient_id, split, raw_dir, out_dir,
                    train_names, valid_names, test_names):
    pname      = f'patient{patient_id:03d}'
    pdir       = os.path.join(raw_dir, pname)

    if not os.path.isdir(pdir):
        print(f'  WARNING: {pdir} not found — skipping')
        return

    ed_frame, es_frame = read_ed_es(pdir)

    for frame_num in [ed_frame, es_frame]:
        frame_str = f'{frame_num:02d}'
        img_path  = os.path.join(pdir, f'{pname}_frame{frame_str}.nii.gz')
        lbl_path  = os.path.join(pdir, f'{pname}_frame{frame_str}_gt.nii.gz')

        if not os.path.exists(img_path):
            print(f'  WARNING: {img_path} not found — skipping')
            continue
        if not os.path.exists(lbl_path):
            print(f'  WARNING: {lbl_path} not found — skipping')
            continue

        # nibabel loads as (H, W, S); transpose to (S, H, W)
        img_vol = nib.load(img_path).get_fdata().astype(np.float32)
        lbl_vol = nib.load(lbl_path).get_fdata().astype(np.uint8)
        img_vol = np.transpose(img_vol, (2, 0, 1))
        lbl_vol = np.transpose(lbl_vol, (2, 0, 1))

        img_vol = normalize_volume(img_vol)

        if split == 'test':
            fname = f'{pname}_frame{frame_str}.npz'
            np.savez_compressed(
                os.path.join(out_dir, 'test', fname),
                img=img_vol, label=lbl_vol)
            test_names.append(fname)

        else:
            subdir = os.path.join(out_dir, split)
            for si in range(img_vol.shape[0]):
                fname = f'{pname}_frame{frame_str}_slice{si:03d}.npz'
                np.savez_compressed(
                    os.path.join(subdir, fname),
                    img=img_vol[si], label=lbl_vol[si])
                if split == 'train':
                    train_names.append(fname)
                else:
                    valid_names.append(fname)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', required=True,
                        help='Path to ACDC training/ directory containing patient001…patient100')
    parser.add_argument('--out_dir', default='./data/ACDC',
                        help='Output directory (default: ./data/ACDC)')
    args = parser.parse_args()

    raw_dir = args.raw_dir
    out_dir = args.out_dir

    for sub in ['train', 'valid', 'test', 'lists_ACDC']:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    train_names, valid_names, test_names = [], [], []

    print(f'Train — patients 001-070')
    for pid in tqdm(TRAIN_IDS):
        process_patient(pid, 'train', raw_dir, out_dir,
                        train_names, valid_names, test_names)
    print(f'  → {len(train_names)} slices')

    print(f'Valid — patients 071-080')
    for pid in tqdm(VALID_IDS):
        process_patient(pid, 'valid', raw_dir, out_dir,
                        train_names, valid_names, test_names)
    print(f'  → {len(valid_names)} slices')

    print(f'Test  — patients 081-100')
    for pid in tqdm(TEST_IDS):
        process_patient(pid, 'test', raw_dir, out_dir,
                        train_names, valid_names, test_names)
    print(f'  → {len(test_names)} volumes')

    lists_dir = os.path.join(out_dir, 'lists_ACDC')
    with open(os.path.join(lists_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_names) + '\n')
    with open(os.path.join(lists_dir, 'valid.txt'), 'w') as f:
        f.write('\n'.join(valid_names) + '\n')
    with open(os.path.join(lists_dir, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_names) + '\n')

    print(f'\nDone.')
    print(f'  train.txt : {len(train_names)} entries')
    print(f'  valid.txt : {len(valid_names)} entries')
    print(f'  test.txt  : {len(test_names)} entries')
    print(f'  Output    : {out_dir}')


if __name__ == '__main__':
    main()
