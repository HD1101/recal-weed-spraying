"""Original descriptive coverage/IoU diagnostics; no model import."""
import statistics
def any_match_recall(records):
    import numpy as np
    hits = total = 0
    for record in records:
        g = [b for b in record.ground_truth if b.class_id == 1]
        p = [v.box for v in record.predictions if v.box.class_id == 1 and v.confidence >= .001]
        total += len(g)
        if not g or not p:
            continue
        gt = np.asarray([[b.x1,b.y1,b.x2,b.y2] for b in g], dtype=float)
        pred = np.asarray([[b.x1,b.y1,b.x2,b.y2] for b in p], dtype=float)
        wh = np.maximum(0, np.minimum(gt[:,None,2:],pred[None,:,2:])-
                        np.maximum(gt[:,None,:2],pred[None,:,:2]))
        intersection = wh[...,0]*wh[...,1]
        union = np.prod(gt[:,2:]-gt[:,:2],axis=1)[:,None] + np.prod(pred[:,2:]-pred[:,:2],axis=1)[None,:] - intersection
        hits += int(((intersection/union).max(axis=1) >= .5).sum())
    return hits/total if total else None


def compute_rows(records, nozzles):
    from wcr import aggregate, image_metrics, spray_union, validate_records
    validate_records(records)
    assert all(p.confidence >= .001 for r in records for p in r.predictions)
    recall = any_match_recall(records)
    median = statistics.median(sum(p.box.class_id==1 for p in r.predictions) for r in records)
    rows = []
    for n in nozzles:
        metrics = [image_metrics(r,spray_union(r,n,0.0)) for r in records]
        pooled = aggregate(metrics)
        losses = [(m.weed_total-m.weed_sprayed)/m.weed_total for m in metrics if m.weed_total]
        rows.append({'nozzles':n, 'minimum_grid_threshold':0.0, 'prediction_confidence_floor':.001,
                     'images':len(records), 'images_with_weeds':len(losses), 'weed_total':pooled['weed_total'],
                     'wcr_per_weed_at_min':pooled['wcr_per_weed'],
                     'weed_image_mean_miss_rate_at_min':statistics.fmean(losses) if losses else None,
                     'spray_area_fraction_at_min':pooled['spray_area_fraction'],
                     'iou50_detection_recall':recall, 'weed_predictions_per_image_median':median})
    return rows

