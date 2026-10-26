"""Recompute group assignments and compare them with the published fixed IDs."""
import csv
from collections import defaultdict
from pathlib import Path
import random
from grouping import group_partition
ROOT=Path(__file__).resolve().parents[1]


def closest_groups(groups,target,seed):
    order=sorted(groups);random.Random(seed).shuffle(order);reachable={0:()}
    for group in order:
        size=len(groups[group])
        for total,subset in tuple(reachable.items()):reachable.setdefault(total+size,subset+(group,))
    best=min((total for total in reachable if total>0),key=lambda t:(abs(t-target),t))
    return tuple(sorted(reachable[best])),best


def read_csv(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def verify():
    allrows=read_csv(ROOT/'splits/LB_IMAGE_GROUPS.csv');byid={r['id']:r for r in allrows}
    for heldout,counts in [('bbro',(1200,200)),('near30',(1203,200)),('walledgarden',(1200,200))]:
        groups=defaultdict(list)
        for row in allrows:
            if row['field']!=heldout:groups[row['group_id']].append(row['id'])
        if heldout=='bbro':
            val_g,_=closest_groups(groups,200,0)
            train_g,_=closest_groups({g:ids for g,ids in groups.items() if g not in val_g},1200,1)
            selected={s:sorted(i for g in gs for i in groups[g]) for s,gs in [('train',train_g),('val',val_g)]}
        else:selected=group_partition(groups,*counts,seed=0)
        gs={}
        for s,count in zip(('train','val'),counts):
            fixed=read_csv(ROOT/'splits'/heldout/(s+'.csv'))
            assert sorted(selected[s])==sorted(r['id'] for r in fixed),'FIXED_SPLIT_DIFFERS'
            assert len(fixed)==count
            gs[s]={r['group_id'] for r in fixed}
            assert all(byid[i]['field']!=heldout for i in selected[s])
            assert {i for g in gs[s] for i in groups[g]}==set(selected[s]),'PARTIAL_GROUP'
        assert not gs['train']&gs['val']
        target=read_csv(ROOT/'splits'/heldout/'target.csv')
        assert {r['id'] for r in target}=={r['id'] for r in allrows if r['field']==heldout}
        assert not {r['id'] for r in target}&(set(selected['train'])|set(selected['val']))
        print('FIXED_SPLIT_VERIFIED',heldout,*counts,'groups',len(gs['train']),len(gs['val']),
              'target_groups',len({r['group_id'] for r in target}))


if __name__=='__main__':verify()
