"""Original formal split/date functions; scientific logic unchanged."""
from collections import defaultdict
import datetime as dt
from pathlib import Path
import random
import re
LB = re.compile('^(bbro|near30|walledgarden)_\\1_(\\d{2})_(\\d{2})_(\\d{4})_([iv])_(\\d+)(?:_(\\d+))?\\.png$')

def lb_meta(name):
    m=LB.fullmatch(Path(name).name)
    if not m:raise ValueError('LB_FILENAME_UNPARSED: '+name)
    field,day,month,year,mode,ident,frame=m.groups()
    date=dt.date(int(year),int(month),int(day)).isoformat()
    assert (mode=='i')==(frame is None)
    group=f'{field}/{date}/'+('single_photos' if mode=='i' else f'video_{int(ident)}')
    return {'field':field,'capture_date':date,'date_source':'LB_filename','group_id':group}


def image_date(im,dataset,overrides=None):
    if im['id'] in (overrides or {}):return overrides[im['id']],'configured_verified_mapping'
    text=' '.join(str(im.get(k,'')) if k!='image_path' else Path(im[k]).name for k in ('file_name','image_path','id') if im.get(k))
    if dataset in ('bbro','near30','walledgarden','source_holdout'):
        return lb_meta(Path(im.get('image_path',im.get('file_name',im['id']))).name)['capture_date'],'LB_filename'
    for key in ('capture_date','date'):
        if im.get(key):
            try:return dt.date.fromisoformat(im[key]).isoformat(),'audited_manifest_'+str(im.get('date_source',key))
            except ValueError:pass
    m=re.search(r'(20\d\d)-(\d\d)-(\d\d)',text)
    if m:
        return dt.date(*map(int,m.groups())).isoformat(),'filename_or_audited_group_sequence'
    if dataset=='PhenoBench':
        m=re.search(r'(?:^|[/ _])(\d\d)-(\d\d)(?:_|\b)',text)
        if m:
            month,day=map(int,m.groups());dt.date(2000,month,day)
            return f'{month:02d}-{day:02d}','filename_month_day;collection_year_unspecified'
    if dataset=='OD_SugarBeets':
        for m in re.finditer(r'(?<!\d)(\d{10,16})(?:\.\d+)?(?!\d)',text):
            number=int(m.group(1));seconds=number/(10**6 if number>=10**15 else 1000 if number>=10**12 else 1)
            try:date=dt.datetime.fromtimestamp(seconds,dt.timezone.utc).date()
            except (ValueError,OverflowError,OSError):continue
            if dt.date(2016,3,1)<=date<=dt.date(2016,6,30):
                return date.isoformat(),'inferred_unix_timestamp_UTC'
        group=str(im.get('group_id',''));m=re.search(r'(20\d\d-\d\d-\d\d)',group)
        if m:
            date=dt.date.fromisoformat(m.group(1)).isoformat()
            return date,('audited_inferred_timestamp_group' if any(s in group.lower() for s in ('infer','timestamp','unix')) else 'audited_sequence_group_date')
    return None,'date_unresolved'


def group_partition(groups,train_count,val_count,seed):
    """Exact joint size DP over indivisible groups; uses sizes only, never labels."""
    total=sum(len(v) for v in groups.values())
    assert total>=train_count+val_count,'INSUFFICIENT_POOL'
    remaining=total-train_count-val_count
    if 0<remaining<min(map(len,groups.values())):raise ValueError('EXACT_WHOLE_GROUP_COUNTS_IMPOSSIBLE')
    order=sorted(groups);random.Random(seed).shuffle(order)
    parent={(0,0):None};goal=(train_count,val_count)
    for j,g in enumerate(order):
        size=len(groups[g])
        for (a,b) in list(parent):
            for key,split in [((a+size,b),'train'),((a,b+size),'val')]:
                if key[0]<=train_count and key[1]<=val_count and key not in parent:
                    parent[key]=((a,b),j,split)
        if goal in parent:break
    if goal not in parent:raise ValueError('EXACT_WHOLE_GROUP_PARTITION_NOT_FOUND')
    assigned={'train':[],'val':[]};state=goal
    while parent[state] is not None:
        prev,j,split=parent[state];assigned[split].append(order[j]);state=prev
    return {s:sorted(i for g in gs for i in groups[g]) for s,gs in assigned.items()}


def audit_split(all_rows,splits,heldout,counts):
    by_id={r['id']:r for r in all_rows};all_groups=defaultdict(set)
    for r in all_rows:all_groups[r['group_id']].add(r['id'])
    selected={k:set(v) for k,v in splits.items()}
    assert [len(selected[k]) for k in ('train','val')]==list(counts)
    assert not selected['train']&selected['val']
    source_groups=[]
    for s in ('train','val'):
        assert all(by_id[i]['field']!=heldout for i in selected[s]),'TARGET_FIELD_LEAKAGE'
        gs={by_id[i]['group_id'] for i in selected[s]};source_groups.append(gs)
        assert all(all_groups[g]<=selected[s] for g in gs),'PARTIAL_GROUP'
    assert not source_groups[0]&source_groups[1]
    target={r['id'] for r in all_rows if r['field']==heldout}
    assert not target&(selected['train']|selected['val'])
    return {'train_images':len(selected['train']),'source_holdout_images':len(selected['val']),
            'target_images':len(target),'whole_groups':True,'target_zero_overlap':True,
            'train_source_holdout_group_overlap':False,'source_groups':[len(g) for g in source_groups]}

