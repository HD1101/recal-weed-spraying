"""Unchanged formal per-instance PhenoBench conversion."""
def pheno_boxes(semantics,instances,add,filename,audit=None,tie_break='lowest_semantic_id'):
    import numpy as np
    if tie_break!='lowest_semantic_id':raise ValueError('UNKNOWN_PHENO_MAJORITY_TIE_BREAK')
    if semantics.shape!=instances.shape:raise ValueError('PHENO_MASK_SHAPE_MISMATCH')
    if not np.isin(semantics,[0,1,2,3,4]).all():raise ValueError('UNKNOWN_PHENO_SEMANTIC_LABEL: '+filename)
    if audit is None:audit={}
    for key in ('input_instances','discarded_majority_background_count','mixed_semantics_instances_count','majority_tie_count'):
        audit.setdefault(key,0)
    for key in ('discarded_majority_background_instances','majority_ties'):
        audit.setdefault(key,[])
    boxes=[]; degenerate=[]
    ids=np.unique(instances)
    for instance in ids[ids!=0]:
        audit['input_instances']+=1
        yy,xx=np.where(instances==instance)
        values,counts=np.unique(semantics[yy,xx],return_counts=True)
        winners=values[counts==counts.max()]
        # Majority is determined on raw semantics, before merging partial labels.
        majority=int(winners[0])  # sorted np.unique: deterministic lowest label for ties
        audit['mixed_semantics_instances_count']+=int(len(values)>1)
        if len(winners)>1:
            audit['majority_tie_count']+=1
            audit['majority_ties'].append({'file':filename,'instance_id':int(instance),
                'tied_semantics':list(map(int,winners)),'selected_semantic':majority,
                'semantic_pixel_counts':{str(int(v)):int(n) for v,n in zip(values,counts)}})
        if majority==0:
            audit['discarded_majority_background_count']+=1
            audit['discarded_majority_background_instances'].append({'file':filename,'instance_id':int(instance)})
            continue
        cls=0 if majority in (1,3) else 1
        x1,y1=int(xx.min()),int(yy.min()); x2,y2=int(xx.max())+add,int(yy.max())+add
        if x2<=x1 or y2<=y1:
            degenerate.append({'file':filename,'instance_id':int(instance),'class_id':cls});continue
        boxes.append((cls,x1,y1,x2,y2))
    return boxes,degenerate

