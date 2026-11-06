"""Aggregate additive D005--D007 results and build the Chinese report."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'analysis'))
from common import csv_write,sha
from aggregate import cross_seed
from posthoc_20261104.archive import dump_new

KEY=('rotation','model','dataset','scheme','nozzles','K')
METRICS=('mean_actual_weed_image_miss_rate','mean_actual_WCR','mean_spray_area_fraction',
         'mean_crop_exposure','full_spray_fallback_count','weed_image_miss_rate_le_alpha_fraction',
         'WCR_ge_target_fraction','K_prime_mean','valid_metrics_repeats','empty_test_repeats',
         'test_images_mean','test_images_min','test_images_max')


def rows(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))


def typed(row):
    out={}
    for k,v in row.items():
        if v=='':out[k]=None
        elif k in ('rotation','model','dataset','method','scheme'):out[k]=v
        else:
            try:out[k]=float(v) if any(c in v.lower() for c in '.e') else int(v)
            except (TypeError,ValueError):out[k]=v
    return out


def identity(row,include_scheme=True):
    return tuple(str(row[k]) for k in KEY if include_scheme or k!='scheme')


def table(head,body):
    return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |',
                       *['| '+' | '.join(map(str,r))+' |' for r in body]])


def pct(x,digits=2):return 'NA' if x is None else f'{100*float(x):.{digits}f}'
def rng(row,metric,digits=2):
    if row.get(metric+'_seed_mean') is None:return 'NA'
    return f"{pct(row[metric+'_seed_mean'],digits)} [{pct(row[metric+'_seed_min'],digits)}, {pct(row[metric+'_seed_max'],digits)}]"


def aggregate_all(output):
    files=list(output.glob('per_configuration/*/*/POSTHOC_COMPLETE.json'))
    assert len(files)==81,('INCOMPLETE_TARGETS',len(files))
    a=[];b=[];audits=[];checks=[]
    for marker in sorted(files):
        complete=json.loads(marker.read_text())
        for name,digest in complete['files'].items():assert sha(marker.parent/name)==digest
        a.extend(typed(r) for r in rows(marker.parent/'D005_PER_SEED.csv'))
        b.extend(typed(r) for r in rows(marker.parent/'D006_PER_SEED.csv'))
        job=marker.parts[-3];ds=marker.parts[-2]
        audits.extend({'job':job,'dataset':ds,**typed(r)} for r in rows(marker.parent/'D006_DRAW_INDEX_AUDIT.csv'))
        checks.append({'job':job,'dataset':ds,**json.loads((marker.parent/'VALIDATION.json').read_text())['checks']})
    assert len(a)==6480 and len(b)==1620
    csv_write(output/'D005_PER_SEED_ALL_RESULTS.csv',a);csv_write(output/'D006_PER_SEED_ALL_RESULTS.csv',b)
    ac=cross_seed(a,('rotation','model','dataset','method','scheme','nozzles','K'))
    bc=cross_seed(b,('rotation','model','dataset','method','scheme','nozzles','K'))
    assert len(ac)==2160 and len(bc)==540
    csv_write(output/'D005_CROSS_SEED_ALL_RESULTS.csv',ac);csv_write(output/'D006_CROSS_SEED_ALL_RESULTS.csv',bc)
    csv_write(output/'RECALCULATION_VALIDATION_COUNTS.csv',checks)
    # Compare both same-loss policies on every exactly paired test set.
    paired={}
    for r in ac:paired.setdefault(identity(r),{})[r['method']]=r
    comparison=[]
    for key,methods in sorted(paired.items()):
        cr,nm=methods['crc'],methods['naive_m'];r={k:cr[k] for k in KEY}
        for metric in METRICS:
            for suffix in ('seed_mean','seed_min','seed_max'):
                r['crc_'+metric+'_'+suffix]=cr.get(metric+'_'+suffix)
                r['naive_m_'+metric+'_'+suffix]=nm.get(metric+'_'+suffix)
            av,bv=cr.get(metric+'_seed_mean'),nm.get(metric+'_seed_mean')
            r['naive_m_minus_crc_'+metric]=bv-av if av is not None and bv is not None else None
        r.update(n_seeds=cr['n_seeds'],crc_M_le_alpha=cr['mean_actual_weed_image_miss_rate_seed_mean']<=.1,
                 naive_m_M_le_alpha=nm['mean_actual_weed_image_miss_rate_seed_mean']<=.1)
        comparison.append(r)
    csv_write(output/'D005_CRC_VS_NAIVE_M_CROSS_SEED.csv',comparison)
    main5=[r for r in comparison if r['rotation'] in ('bbro','near30') and r['nozzles']==4 and r['scheme']=='grouped' and r['K'] in (20,50)]
    csv_write(output/'D005_MAIN_N4_GROUPED_K20_K50.csv',main5)
    counts=[]
    for n in (1,2,3,4):
        for scheme in ('ideal','grouped'):
            for K in (5,10,20,50,100):
                for scope,rotations in [('main_18',('bbro','near30')),('supplement_9',('walledgarden',)),('all_27',('bbro','near30','walledgarden'))]:
                    for method in ('crc','naive_m'):
                        selected=[r for r in ac if r['rotation'] in rotations and r['nozzles']==n and r['K']==K and r['scheme']==scheme and r['method']==method]
                        values=[r['mean_actual_weed_image_miss_rate_seed_mean'] for r in selected if r.get('mean_actual_weed_image_miss_rate_seed_mean') is not None]
                        counts.append({'scope':scope,'nozzles':n,'scheme':scheme,'K':K,'method':method,'cells':len(selected),
                            'defined_cells':len(values),'M_le_alpha_cells':sum(x<=.1 for x in values),
                            'M_gt_alpha_cells':sum(x>.1 for x in values)})
    csv_write(output/'D005_ATTAINMENT_CELL_COUNTS.csv',counts)
    original=[typed(r) for r in rows(ROOT/'results/CROSS_SEED_ALL_RESULTS.csv') if r['method']=='crc' and r['scheme']=='grouped']
    crop_original={identity(typed(r)):typed(r) for r in rows(ROOT/'results/CROP_EXPOSURE_SUMMARY_CROSS_SEED.csv')
                   if r['method']=='crc' and r['scheme']=='grouped'}
    for r in original:
        if r['dataset']=='PhenoBench':
            cr=crop_original[identity(r)]
            for suffix in ('seed_mean','seed_min','seed_max'):
                r['mean_crop_exposure_'+suffix]=cr['mean_crop_exposure_'+suffix]
    oi={identity(r,False):r for r in original};comparison6=[]
    for r in bc:
        old=oi[identity(r,False)];item={k:r[k] for k in KEY};item['original_scheme']='grouped'
        for metric in METRICS:
            for suffix in ('seed_mean','seed_min','seed_max'):
                item['nested_'+metric+'_'+suffix]=r.get(metric+'_'+suffix)
                item['original_'+metric+'_'+suffix]=old.get(metric+'_'+suffix)
            av,bv=old.get(metric+'_seed_mean'),r.get(metric+'_seed_mean')
            item['nested_minus_original_'+metric]=bv-av if av is not None and bv is not None else None
        comparison6.append(item)
    csv_write(output/'D006_VS_ORIGINAL_CROSS_SEED.csv',comparison6)
    csv_write(output/'D006_MAIN_N4_K20_K50_K100.csv',[r for r in comparison6 if r['rotation'] in ('bbro','near30') and r['nozzles']==4 and r['K'] in (20,50,100)])
    count6=[]
    for scope,rotations in [('main_18',('bbro','near30')),('supplement_9',('walledgarden',)),('all_27',('bbro','near30','walledgarden'))]:
        for n in (1,2,3,4):
            for K in (5,10,20,50,100):
                for method,source in [('original_grouped',original),('nested_fixed_test',bc)]:
                    selected=[r for r in source if r['rotation'] in rotations and r['nozzles']==n and r['K']==K]
                    values=[r['mean_actual_weed_image_miss_rate_seed_mean'] for r in selected if r.get('mean_actual_weed_image_miss_rate_seed_mean') is not None]
                    count6.append({'scope':scope,'nozzles':n,'K':K,'resampling':method,'cells':len(selected),'defined_cells':len(values),
                        'M_gt_alpha_cells':sum(v>.1 for v in values),'M_le_alpha_cells':sum(v<=.1 for v in values)})
    csv_write(output/'D006_EXCEEDANCE_COUNTS.csv',count6)
    ti=[]
    for r in bc:
        ti.append({**{k:r[k] for k in KEY},**{k:v for k,v in r.items() if k.startswith(('test_images','empty_test'))}})
    csv_write(output/'D006_TEST_IMAGE_COUNTS.csv',ti)
    audit_groups=defaultdict(list)
    for r in audits:audit_groups[(r['dataset'],r['K'])].append(r)
    audit_summary=[]
    for (ds,K),rs in sorted(audit_groups.items()):
        # Shared draws are counted once per target, rather than per detector/seed/rotation.
        byrep={}
        for r in rs:
            rep=r['repeat_one_based']
            key=tuple(r[k] for k in ('new_calibration_sha256','new_pool_sha256','new_test_sha256'))
            if rep in byrep:assert byrep[rep][0]==key,('D006_SAMPLES_NOT_SHARED',ds,K,rep)
            else:byrep[rep]=(key,r)
        unique=[x[1] for x in byrep.values()];assert len(unique)==200
        audit_summary.append({'dataset':ds,'K':K,'unique_repeats':200,'new_overlap_repeats':sum(r['new_calibration_pool_image_overlap']>0 for r in unique),
            'superseded_overlap_repeats':sum(r['superseded_definition_calibration_pool_image_overlap']>0 for r in unique),
            'new_test_images_mean':statistics.fmean(r['new_test_images'] for r in unique),
            'new_test_images_min':min(r['new_test_images'] for r in unique),'new_test_images_max':max(r['new_test_images'] for r in unique)})
    csv_write(output/'D006_NESTED_SAMPLING_AUDIT.csv',audit_summary)
    mc=[typed(r) for r in rows(output/'D007_MC_PER_SEED.csv')]
    mcgroups=defaultdict(list)
    for r in mc:mcgroups[tuple(r[k] for k in ('rotation','model','dataset','scheme','K'))].append(r)
    mcc=[]
    for key,rs in sorted(mcgroups.items()):
        flags=[r['mean_within_alpha_plus_minus_2SE']=='True' or r['mean_within_alpha_plus_minus_2SE'] is True for r in rs]
        mcc.append({**dict(zip(('rotation','model','dataset','scheme','K'),key)),
            'n_seeds':len(rs),'seed_mean_M':statistics.fmean(r['mean_actual_weed_image_miss_rate'] for r in rs),
            'M_seed_min':min(r['mean_actual_weed_image_miss_rate'] for r in rs),'M_seed_max':max(r['mean_actual_weed_image_miss_rate'] for r in rs),
            'MC_SE_seed_mean':statistics.fmean(r['M_MC_standard_error'] for r in rs),
            'MC_SE_seed_min':min(r['M_MC_standard_error'] for r in rs),'MC_SE_seed_max':max(r['M_MC_standard_error'] for r in rs),
            'n_seeds_with_boundary_flag':sum(flags),'seeds_with_boundary_flag':','.join(str(r['seed']) for r,f in zip(rs,flags) if f)})
    csv_write(output/'D007_MC_CROSS_SEED_WITH_FLAGS.csv',mcc)
    # Additional formal metrics accompany the MC diagnostic without redefining
    # it: these are joined from immutable seed-level summaries, not new risks.
    formal={tuple(str(r[k]) for k in ('rotation','model','seed','dataset','scheme','nozzles','K')):typed(r)
            for r in rows(ROOT/'results/PER_SEED_ALL_RESULTS.csv') if r['method']=='crc'}
    crops={tuple(str(r[k]) for k in ('rotation','model','seed','dataset','scheme','nozzles','K')):typed(r)
           for r in rows(ROOT/'results/CROP_EXPOSURE_SUMMARY_PER_SEED.csv') if r['method']=='crc'}
    context=[]
    for r in mc:
        key=tuple(str(r[k]) for k in ('rotation','model','seed','dataset','scheme','nozzles','K'))
        old=formal[key]
        context.append({**r,**{k:old.get(k) for k in ('mean_actual_WCR','WCR_ge_target_fraction',
            'mean_spray_area_fraction','full_spray_fallback_count','weed_image_miss_rate_le_alpha_fraction','K_prime_mean')},
            'mean_crop_exposure':crops[key]['mean_crop_exposure'] if key in crops else None})
    csv_write(output/'D007_CONTEXT_METRICS_PER_SEED.csv',context)
    csv_write(output/'D007_CONTEXT_METRICS_CROSS_SEED.csv',cross_seed(context,('rotation','model','dataset','method','scheme','nozzles','K')))
    return ac,bc,comparison,counts,count6,mc,mcc,audit_summary


def make_report(output,data):
    ac,bc,comparison,count5,count6,mc,mcc,audits=data
    hashes=json.loads((output/'POSTHOC_PLAN_HASH_HISTORY.json').read_text())
    inputs=json.loads((output/'INPUT_VERIFICATION_20261104.json').read_text())
    lines=['# ReCal 三项事后分析报告（2026-11-04）','',
        '**全部为结果产生后的追加分析，登记为偏离 D005–D007；不替换任何预设分析。**',
        'D005使用正式抽样；D006为另行核定的嵌套组顺序与固定测试集的新重采样，仅作描述性敏感性分析；D007量化给定模型和抽样方案下的蒙特卡洛误差。',
        '',f"原计划 SHA256：`{hashes['original_SHA256']}`。",f"追加修订1后 SHA256：`{hashes['revision_1_SHA256']}`；修订时间 `{hashes['revision_1_timestamp']}`，早于D006风险计算。",'',
        '## 输入与口径','',
        '27个正式配置、81个目标评估；全部4种喷头数、5档K、200次抽样、3个训练种子分别计算。D005含ideal和grouped；D006仅为grouped的新敏感性抽样。跨种子均值与[min,max]为等权汇总，不是置信区间。',
        'M先在每次测试集中对含草图平均，再跨抽样平均；WCR和A先在每次测试集中分别按棵/面积合计求比值；作物暴露仅PhenoBench，主估计为mean_crop_exposure（每次测试内像素比再跨抽样平均）。空测试/无含草测试的M记NA。',
        '',table(['数据','图像','杂草框','甜菜框'],[[d,v['images'],v['weed_boxes'],v['crop_boxes']] for d,v in inputs['dataset_counts'].items()]),
        '', 'OD正式风险标签仅含杂草，表中甜菜框0仅指这套风险输入，并非OD原数据没有作物。PhenoBench另有772张semantics。LB/OD新下载的5,366个标签与2份原清单逐文件匹配正式输入SHA256；既有PhenoBench标签和掩码也逐文件匹配。108份预测文件的哈希、图像ID、尺寸、分组及三份固定划分逐一通过核验。',
        '',f"下载时间：`{inputs['download_started_utc']}` 至 `{inputs['download_completed_utc']}`。来源原清单：",
        *['- `'+x+'`' for x in inputs['source_manifest_paths']],
        '', '完整来源路径、逐文件SHA256和本地私有位置见INPUT_VERIFICATION_20261104.json指向的TRUTH_PRIVATE_PROVENANCE.json（仓库外，不公开真值）。远端仅只读流式传输，无图像/权重，无远端写入、训练或推理；连接已结束。',
        '', '## D005：与CRC同损失、同全喷规则的naive-M','',
        'naive-M只移除CRC有限样本修正，以含草校准图的平均损失≤0.1选择最大阈值，无含草图或无可行阈值时全喷。两方法的校准与测试索引完全相同。',
        '下表达标格数按正文18格的跨种子平均M≤10%计算，和“200次抽样中M≤10%的比例”不同。', '']
    rr=[r for r in count5 if r['scope']=='main_18' and r['nozzles']==4]
    lines.append(table(['划分','K','CRC达标格/18','naive-M达标格/18'],[
        [scheme,K,next(r['M_le_alpha_cells'] for r in rr if r['scheme']==scheme and r['K']==K and r['method']=='crc'),
         next(r['M_le_alpha_cells'] for r in rr if r['scheme']==scheme and r['K']==K and r['method']=='naive_m')]
        for scheme in ('grouped','ideal') for K in (5,10,20,50,100)]))
    lines+=['','正文18格逐格配对（n=4，grouped，K=50；M、A单位%，“均值[min,max]”；退全喷为每200次的跨种子均数）：','']
    ar={(r['rotation'],r['model'],r['dataset'],r['method']):r for r in ac if r['rotation'] in ('bbro','near30') and r['nozzles']==4 and r['scheme']=='grouped' and r['K']==50}
    body=[]
    for rot in ('bbro','near30'):
        for model in ('asset_detr','rtdetr_r18','yolo26m'):
            for ds in (rot,'PhenoBench','OD_SugarBeets'):
                cr,nm=ar[(rot,model,ds,'crc')],ar[(rot,model,ds,'naive_m')]
                body.append([rot,model,ds,rng(cr,'mean_actual_weed_image_miss_rate'),rng(nm,'mean_actual_weed_image_miss_rate'),
                    rng(cr,'mean_spray_area_fraction'),rng(nm,'mean_spray_area_fraction'),
                    f"{cr['full_spray_fallback_count_seed_mean']:.1f}/{nm['full_spray_fallback_count_seed_mean']:.1f}"])
    lines.append(table(['轮换','模型','目标','CRC M','naive-M M','CRC A','naive-M A','CRC/naive-M退全喷'],body))
    lines+=['','全部逐格差值及K=20配对见D005_CRC_VS_NAIVE_M_CROSS_SEED.csv和D005_MAIN_N4_GROUPED_K20_K50.csv。作物暴露、WCR、抽样达标比例及种子范围均在D005_CROSS_SEED_ALL_RESULTS.csv；逐种子见D005_PER_SEED_ALL_RESULTS.csv。',
        '', '## D006：嵌套组顺序＋固定K100池外测试集','',
        '每次重复仅shuffle一次组列表，全部K共用；各K沿前缀入池并抽K张，测试为K100池外全部组。随机种子由20261013、D006、1起算的重复号确定。已断言每个K整个校准池与测试集在图像及组层面零交叉；同目标的索引跨检测器、训练种子和n共用。该新抽样同时改变了校准样本，不能把与正式结果的差异单独归因于测试集构成。',
        '',table(['K','原分组超标格/18','嵌套固定测试超标格/18'],[
            [K,next(r['M_gt_alpha_cells'] for r in count6 if r['scope']=='main_18' and r['nozzles']==4 and r['K']==K and r['resampling']=='original_grouped'),
             next(r['M_gt_alpha_cells'] for r in count6 if r['scope']=='main_18' and r['nozzles']==4 and r['K']==K and r['resampling']=='nested_fixed_test')]
            for K in (20,50,100)]),
        '', '测试图数随重复改变，但同一重复在全部K固定。下表每个目标200次的测试图数均值[min,max]；相同目标跨配置相同：','',
        table(['目标','测试图数均值[min,max]','新方案交叉次数/200','被弃原定义的K20池交叉次数/200'],[
            [r['dataset'],f"{r['new_test_images_mean']:.2f} [{r['new_test_images_min']}, {r['new_test_images_max']}]",r['new_overlap_repeats'],r['superseded_overlap_repeats']]
            for r in audits if r['K']==20]),
        '', 'n=4、K=50 的正文18格对照（M、A为%及跨种子均值[min,max]，退全喷为每200次的跨种子均数）：','']
    new50={(r['rotation'],r['model'],r['dataset']):r for r in bc if r['rotation'] in ('bbro','near30') and r['nozzles']==4 and r['K']==50}
    original50={(r['rotation'],r['model'],r['dataset']):r for r in ac if r['rotation'] in ('bbro','near30') and r['method']=='crc' and r['scheme']=='grouped' and r['nozzles']==4 and r['K']==50}
    body=[]
    for key,nr in sorted(new50.items()):
        old=original50[key]
        body.append([*key,rng(old,'mean_actual_weed_image_miss_rate'),rng(nr,'mean_actual_weed_image_miss_rate'),
            rng(old,'mean_spray_area_fraction'),rng(nr,'mean_spray_area_fraction'),
            f"{old['full_spray_fallback_count_seed_mean']:.1f}/{nr['full_spray_fallback_count_seed_mean']:.1f}"])
    lines.append(table(['轮换','模型','目标','原M','新M','原A','新A','原/新退全喷'],body))
    empty_total=sum(r.get('empty_test_repeats_seed_mean',0) or 0 for r in bc if r['nozzles']==4)
    lines +=['',f'空测试集计数：n=4跨种子均值合计为{empty_total:g}次（同一目标抽样在各检测器/轮换/K重复出现，不是去重次数）；逐格空集次数全部保留。',
        '逐格M、A、退全喷次数和种子范围见D006_MAIN_N4_K20_K50_K100.csv；全部n/K、WCR、作物暴露和达标比例见D006_CROSS_SEED_ALL_RESULTS.csv。测试图数每格见D006_TEST_IMAGE_COUNTS.csv；完整逐次索引哈希和交叉审计在per_configuration下。空测试抽样保留，不重抽；对应指标NA，空集次数保留。',
        '', '## D007：蒙特卡洛标准误','',
        '每个格子每个训练种子的SE=样本SD/√200；标记|平均M−0.1|≤2SE。这里的SE只描述固定检测器、固定数据及所定抽样机制下的MC误差，不包含跨训练种子的变异，也不是期望风险保证。', '']
    mcbody=[]
    for scheme,K in [('grouped',20),('grouped',50),('ideal',50)]:
        selected=[r for r in mc if r['rotation'] in ('bbro','near30') and r['scheme']==scheme and r['K']==K]
        flagged=[r for r in selected if r['mean_within_alpha_plus_minus_2SE']=='True' or r['mean_within_alpha_plus_minus_2SE'] is True]
        mcbody.append([scheme,K,len(selected),len(flagged),f"{100*min(r['M_MC_standard_error'] for r in selected):.3f}–{100*max(r['M_MC_standard_error'] for r in selected):.3f}"])
    lines.append(table(['划分','K','正文种子级格数','10%±2SE边界格数','SE范围(pp)'],mcbody))
    lines+=['','边界格子完整列示（含walledgarden补充）：','']
    flagged=[r for r in mc if r['mean_within_alpha_plus_minus_2SE']=='True' or r['mean_within_alpha_plus_minus_2SE'] is True]
    lines.append(table(['轮换','模型','目标','种子','划分','K','M(%)','SE(pp)','M±2SE(%)'],[
        [r['rotation'],r['model'],r['dataset'],r['seed'],r['scheme'],r['K'],pct(r['mean_actual_weed_image_miss_rate']),
         pct(r['M_MC_standard_error'],3),f"[{pct(r['M_minus_2SE'])}, {pct(r['M_plus_2SE'])}]"] for r in flagged]))
    lines+=['','全部243个种子级格及81个跨种子格均报告，详见D007_MC_PER_SEED.csv与D007_MC_CROSS_SEED_WITH_FLAGS.csv；WCR、A、作物暴露等同口径伴随指标见D007_CONTEXT_METRICS_PER_SEED.csv及其跨种子版。原始均值和SE共3,240个单元的核验均在1e-12绝对/相对容差内一致。',
        '', '## 异常、工程记录与限制','',
        '- 原请求D006存在K特异抽样造成的交叉问题；已按计算前核定的修订1实施新抽样，原被弃定义仅做交叉审计，未作为风险结果报告。',
        '- 正式网格使用原JSON数值0.001调用原threshold_grid，保留浮点表示（如0.11800000000000001），未用“看似相同”的另一浮点网格替换；新增数组查询接口已适配并单测。',
        '- 必要标签取证共3次连接尝试：第一次非交互shell无python命令；第二次OD路径白名单与清单实际目录不一致，守卫停止；第三次使用已有解释器绝对路径及已核实正式浮点标签目录完成。均无远端写入。',
        '- 新入口初次导入缺少仓库根路径，已在新增入口内补齐；本地原测试初次因缺matplotlib失败，使用既有本地依赖路径后通过；不修改既有代码。所有失败记录原样保留。',
        '- 新方法风险表中的校准损失由同一图级损失计算；未引用历史全喷记录中calibration_loss_mean=0占位来解释可行性。',
        '- D005和D006不能把按抽样达标比例当作CRC的期望风险保证；分组迁移仍缺乏图像交换性保证。本报告只给出这批固定数据与预测上的事后比较。',
        '', '## 文件与复核','',
        '本目录CSV为新文件；小数保留计算精度，不为显示舍入而改原结果。analysis/posthoc_20261104为独立模块。新测试、原70项测试、原83个公开文件SHA256复核和完整输出清单见最终验收文件。新增偏离登记见DEVIATIONS_D005_D007.json，历史plan/DEVIATIONS.json及冻结计划未改。',
        '不推送GitHub、不修改Zenodo。']
    p=output/'POSTHOC_REPORT_20261104.md';assert not p.exists();p.write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/posthoc_20261104')
    a=p.parse_args();data=aggregate_all(a.output);make_report(a.output,data)
    print('POSTHOC_SUMMARY_AND_REPORT_COMPLETE',str(a.output),flush=True)


if __name__=='__main__':main()
