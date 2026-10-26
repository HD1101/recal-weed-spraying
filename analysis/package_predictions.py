"""Build a prediction-only release from a read-only local formal backup.

No model runs. Ground truth and machine paths are excluded by an explicit
schema whitelist. Old/new hashes are written ONLY to --private-record.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import tarfile

ROOT=Path(__file__).resolve().parents[1]
IMAGE_KEYS=('id','file_name','width','height','group_id','predictions')
TOP_KEYS=('schema_version','selected_checkpoint','checkpoint_sha256','FP32','TTA','confidence_floor','max_det')
PATTERN=re.compile(r'^runs/(bbro|near30|walledgarden)__(asset_detr|rtdetr_r18|yolo26m)__seed([012])/attempt01/predictions01/(source_holdout|bbro|near30|walledgarden|PhenoBench|OD_SugarBeets)\.json$')


def digest(data):return hashlib.sha256(data).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def encoded(obj):return json.dumps(obj,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()+b'\n'


def prediction_digest(images):
    h=hashlib.sha256()
    for im in images:
        # This binds image order, identity, dimensions, groups and every
        # prediction box/class/confidence, independently of removed metadata.
        h.update(encoded({k:im[k] for k in IMAGE_KEYS if k in im}))
    return h.hexdigest()


def sanitize(raw):
    assert raw['schema_version']==1
    assert (raw['confidence_floor'],raw['max_det'],raw['FP32'],raw['TTA'])==(.001,300,True,False)
    assert raw['selected_checkpoint']=='last'
    result={k:raw[k] for k in TOP_KEYS};images=[]
    for im in raw['images']:
        row={k:im[k] for k in IMAGE_KEYS}
        assert isinstance(row['id'],str) and row['id'] and isinstance(row['group_id'],str) and row['group_id']
        assert row['file_name']==Path(row['file_name']).name
        for value in (row['id'],row['file_name'],row['group_id']):
            assert not value.startswith(('/', '\\')) and '..' not in Path(value).parts
            assert not re.match(r'^[A-Za-z]:',value)
        W,H=row['width'],row['height'];assert type(W) is int and type(H) is int and min(W,H)>0
        assert len(row['predictions'])<=300
        for pred in row['predictions']:
            assert set(pred)=={'xyxy','class_id','confidence'}
            assert type(pred['class_id']) is int and pred['class_id'] in (0,1)
            x1,y1,x2,y2=pred['xyxy']
            assert all(math.isfinite(v) for v in [x1,y1,x2,y2,pred['confidence']])
            assert 0<=x1<x2<=W and 0<=y1<y2<=H and .001<=pred['confidence']<=1
        images.append(row)
    assert len({im['id'] for im in images})==len(images)
    result['images']=images
    assert prediction_digest(raw['images'])==prediction_digest(images)
    return result


def format_readme():
    return '''# ReCal exported predictions — release format v1

This is a local prediction-data release prepared for a separate Zenodo record.
DOI: [prediction DOI to be added]. It contains 27 model/held-out-field/seed
configurations and 108 prediction JSONs, plus export hashes and format metadata.
No images, masks, original/converted ground-truth annotations, auxiliary crop
annotations, model weights or training code are included.

## Layout

`runs/<rotation>__<model>__seed<0|1|2>/attempt01/predictions01/` contains:
`source_holdout.json`, `<rotation>.json`, `PhenoBench.json`,
`OD_SugarBeets.json`, and `EXPORT_COMPLETE.json`. The rotations are bbro,
near30 and walledgarden; models are asset_detr, rtdetr_r18 and yolo26m.

## JSON fields

Top-level: schema_version=1; selected_checkpoint="last"; checkpoint_sha256;
FP32=true; TTA=false; confidence_floor=0.001; max_det=300; images (ordered list).
Each image has id (exact GT-join identifier), file_name (basename used for date
and mask joins), width/height (original-image pixels), group_id (required for
grouped calibration), and predictions. Each prediction has xyxy (four floating
coordinates in original-image pixels), class_id (0=sugar beet, 1=weed), and
confidence in [0.001,1]. An empty list is an explicit detector output, not GT.

Coordinates are clipped to image bounds. The frozen native preprocessing and
YOLO default one-to-many class-aware NMS outputs are retained; no inference,
thresholding, NMS, box truncation or reordering was rerun for this release.
Original-image order and prediction values/classes/confidences are unchanged.
Image dimensions and groups are necessary format metadata, not object labels.

EXPORT_COMPLETE.json hashes the FOUR RELEASE JSONs. Historical paths and
embedded auxiliary GT were removed; therefore original JSON byte hashes no
longer apply. The old/new hash mapping is kept privately and is not distributed.
SHA256SUMS lists the new file hashes (excluding itself).

## Reproduction

Extract the archive with `tar -xzf ReCal_Predictions_20261020_v1.tar.gz`.
Verify `shasum -a 256 -c SHA256SUMS` from the extracted root. Set
`predictions_root` in the companion ReCal analysis configuration to this root.
The analysis code verifies the release EXPORT_COMPLETE hashes and joins GT by
exact image ID. Obtain original annotations and PhenoBench semantics separately
from their providers; predictions alone do not define recall, miss rates or
crop exposure. No GPU is required for the CPU analysis.

## License and citation

The ReCal authors license this prediction-data release under Creative Commons
Attribution 4.0 International (CC BY 4.0):
https://creativecommons.org/licenses/by/4.0/ . Cite the ReCal paper and this
version's Zenodo prediction DOI when available; identify changes in reused
copies. This grant does not relicense the original datasets or model software.

Suggested citation (fill after author publication): ReCal authors. ReCal
exported predictions, v1 (2026). Zenodo. [prediction DOI to be added].
Original dataset references: Lincoln Beet (Darbyshire et al., 2023), PhenoBench
(Weyler et al., 2024), SugarBeets2016 (Chebrolu et al., 2017), and OD-SugarBeets /
WeedNet-R (Guo et al., 2023). Source images and original annotations are NOT
redistributed here.
'''


def build(source,source_manifest,output,private_record):
    source,output,private_record=Path(source),Path(output),Path(private_record)
    assert not output.resolve().is_relative_to(ROOT.resolve()),'PACKAGE_MUST_BE_OUTSIDE_CODE_REPOSITORY'
    assert not private_record.resolve().is_relative_to(ROOT.resolve()),'HASH_MAPPING_IS_PRIVATE'
    assert not private_record.resolve().is_relative_to(output.resolve()),'HASH_MAPPING_MUST_NOT_ENTER_RELEASE_FOLDER'
    assert not output.exists(),'EXISTING_RELEASE_PRESERVED'
    assert not private_record.exists(),'EXISTING_PRIVATE_RECORD_PRESERVED'
    expected={r['archive_path']:r for r in json.loads(Path(source_manifest).read_text())['files']}
    folder=output/'staging/recal-predictions-v1';folder.mkdir(parents=True)
    mapping=[];exports={};removed_keys={};totals={'images':0,'predictions':0}
    with tarfile.open(source,'r|gz') as archive:
        for member in archive:
            match=PATTERN.fullmatch(member.name)
            if not match:continue
            data=archive.extractfile(member).read();old=digest(data)
            assert old==expected[member.name]['sha256'] and len(data)==expected[member.name]['bytes']
            raw=json.loads(data);clean=sanitize(raw);payload=encoded(clean)
            target=folder/member.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
            images=len(clean['images']);count=sum(len(im['predictions']) for im in clean['images'])
            totals['images']+=images;totals['predictions']+=count
            for im in raw['images']:
                for key in set(im)-set(IMAGE_KEYS):removed_keys[key]=removed_keys.get(key,0)+1
            mapping.append({'original_member':member.name,'original_bytes':len(data),'original_sha256':old,
              'release_member':member.name,'release_bytes':len(payload),'release_sha256':digest(payload),
              'prediction_payload_sha256':prediction_digest(clean['images']),'images':images,'prediction_boxes':count,
              'image_order_and_prediction_values_unchanged':True})
            parent=str(Path(member.name).parent)
            exports.setdefault(parent,{'schema_version':1,'selected_checkpoint':'last',
               'checkpoint_sha256':clean['checkpoint_sha256'],'release_format':'ReCal prediction schema 1','files':{}})['files'][target.name]=digest(payload)
            if len(mapping)%4==0:print('RELEASE_CONFIG_COMPLETE',len(mapping)//4,'/27',flush=True)
    assert len(mapping)==108 and len(exports)==27
    for parent,export in exports.items():
        assert len(export['files'])==4
        (folder/parent/'EXPORT_COMPLETE.json').write_bytes(encoded(export))
    (folder/'README.md').write_text(format_readme())
    (folder/'LICENSE-DATA').write_text('ReCal prediction data, copyright (c) 2026 ReCal authors.\nLicensed under Creative Commons Attribution 4.0 International (CC BY 4.0).\nhttps://creativecommons.org/licenses/by/4.0/\nNo license in original datasets or model implementations is granted.\n')
    metadata={'format':'ReCal predictions','version':1,'configurations':27,'prediction_json_files':108,
              'license':'CC-BY-4.0','prediction_DOI':'[prediction DOI to be added]',
              'ground_truth_included':False,'images_or_masks_included':False,**totals}
    (folder/'FORMAT.json').write_bytes(encoded(metadata))
    checksum=''.join(sha(p)+'  '+p.relative_to(folder).as_posix()+'\n' for p in sorted(folder.rglob('*')) if p.is_file())
    (folder/'SHA256SUMS').write_text(checksum)
    private_record.parent.mkdir(parents=True,exist_ok=True)
    private_record.write_text(json.dumps({'private_only':True,'original_to_release':mapping,
       'removed_image_metadata_keys':removed_keys,'prediction_counts':totals,
       'source_archive_sha256':sha(source),'no_scientific_values_changed':True},indent=2)+'\n')
    archive_path=output/'ReCal_Predictions_20261020_v1.tar.gz'
    with archive_path.open('xb') as dst,gzip.GzipFile(fileobj=dst,mode='wb',filename='',mtime=0,compresslevel=6) as gz:
        with tarfile.open(fileobj=gz,mode='w|') as tar:
            for p in sorted(folder.rglob('*')):
                if not p.is_file():continue
                item=tarfile.TarInfo('recal-predictions-v1/'+p.relative_to(folder).as_posix())
                item.size=p.stat().st_size;item.mode=0o644;item.mtime=0;item.uid=item.gid=0;item.uname=item.gname=''
                with p.open('rb') as f:tar.addfile(item,f)
    report={'archive':archive_path.name,'archive_bytes':archive_path.stat().st_size,'archive_sha256':sha(archive_path),
       'uncompressed_release_bytes':sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()),
       'configurations':27,'prediction_data_files':108,'original_annotation_fields_included':False,
       'private_hash_mapping_included':False,'source_results_modified':False,**totals}
    (output/(archive_path.name+'.sha256')).write_text(report['archive_sha256']+'  '+archive_path.name+'\n')
    (output/'README.md').write_text(format_readme())
    (output/'PACKAGE_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PREDICTION_RELEASE_PACKAGED',json.dumps(report),flush=True)
    return report


def unpack_and_verify(archive,output):
    output=Path(output);assert not output.exists(),'EXISTING_UNPACKED_COPY_PRESERVED';output.mkdir(parents=True)
    with tarfile.open(archive,'r:gz') as tar:
        for item in tar:
            assert item.isfile() and not item.issym() and not item.islnk()
            name=Path(item.name);assert not name.is_absolute() and '..' not in name.parts
            assert item.uname==item.gname=='' and item.uid==item.gid==0
            dest=output/name;dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as f:
                stream=tar.extractfile(item)
                for block in iter(lambda:stream.read(1024*1024),b''):f.write(block)
    root=output/'recal-predictions-v1';checked=0
    for line in (root/'SHA256SUMS').read_text().splitlines():
        h,name=line.split('  ',1);assert sha(root/name)==h;checked+=1
    counts={'configurations':0,'prediction_data_files':0,'images':0,'predictions':0}
    for complete in sorted(root.glob('runs/*/attempt01/predictions01/EXPORT_COMPLETE.json')):
        export=json.loads(complete.read_text());counts['configurations']+=1
        for name,h in export['files'].items():
            p=complete.parent/name;assert sha(p)==h
            raw=json.loads(p.read_text());assert set(raw)==set(TOP_KEYS)|{'images'}
            assert all(set(im)==set(IMAGE_KEYS) for im in raw['images'])
            assert sanitize(raw)==raw
            counts['prediction_data_files']+=1;counts['images']+=len(raw['images'])
            counts['predictions']+=sum(len(im['predictions']) for im in raw['images'])
        if counts['configurations']%3==0:print('UNPACK_VERIFY_CONFIGS',counts['configurations'],flush=True)
    assert counts['configurations']==27 and counts['prediction_data_files']==108
    return {'status':'PASSED','release_file_checksums_verified':checked,**counts,
            'schema_whitelist_verified':True,'tar_usernames_and_accounts_stripped':True,'private_hash_mapping_included':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build');b.add_argument('--source-archive',type=Path,required=True);b.add_argument('--source-manifest',type=Path,required=True)
    b.add_argument('--output',type=Path,required=True);b.add_argument('--private-record',type=Path,required=True)
    v=sub.add_parser('verify');v.add_argument('--archive',type=Path,required=True);v.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='build':build(a.source_archive,a.source_manifest,a.output,a.private_record)
    else:print(json.dumps(unpack_and_verify(a.archive,a.output),indent=2))
