"""Original cross-seed aggregation, unchanged."""
import statistics
def cross_seed(rows,identity):
    buckets={}
    for row in rows:buckets.setdefault(tuple(row.get(k,'') for k in identity),[]).append(row)
    output=[]
    for key,items in sorted(buckets.items()):
        seeds=sorted({int(x['seed']) for x in items});assert len(items)==len(seeds),'DUPLICATE_SEED_RESULT'
        r={**dict(zip(identity,key)),'seeds_present':','.join(map(str,seeds)),'n_seeds':len(seeds),
           'all_three_seeds':seeds==[0,1,2]}
        for k in sorted(set().union(*(x.keys() for x in items))):
            if k in identity or k in ('seed','model','rotation','dataset','method','scheme'):continue
            values=[]
            for x in items:
                if x.get(k) not in ('',None):
                    try:values.append(float(x[k]))
                    except (ValueError,TypeError):values=[];break
            if values:r.update({k+'_seed_mean':statistics.fmean(values),k+'_seed_min':min(values),k+'_seed_max':max(values),k+'_valid_seeds':len(values)})
        output.append(r)
    return output

