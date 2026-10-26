"""Materialize immutable split IDs as symlinks to author-obtained images/labels."""
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'data_prep/config.json')
    a=p.parse_args();c=json.loads(a.config.read_text());out=Path(c['output'])/'training'
    assert not out.exists(),'EXISTING_TRAINING_DATA_PRESERVED'
    for rotation in ('bbro','near30','walledgarden'):
        folder=out/rotation
        for split in ('train','val'):
            with (ROOT/'splits'/rotation/(split+'.csv')).open(newline='') as f:rows=list(csv.DictReader(f))
            for im in rows:
                for kind,source in [('images',Path(c['lb_images'])/im['file_name']),
                                    ('labels',Path(c['output'])/'LB/labels'/(im['id']+'.txt'))]:
                    assert source.is_file(),source.name
                    dest=folder/kind/split/source.name;dest.parent.mkdir(parents=True,exist_ok=True)
                    dest.symlink_to(source.resolve())
        (folder/'dataset.yaml').write_text('path: '+json.dumps(str(folder.resolve()))+'\ntrain: images/train\nval: images/val\nnames:\n  0: beet\n  1: weed\n')
    print('TRAINING_SPLITS_READY (fixed shared IDs; no target field supplied to trainer)')


if __name__=='__main__':main()
