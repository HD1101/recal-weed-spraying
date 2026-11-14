"""Build manuscript tables, verification report, and figures from existing CSVs."""
import csv
from decimal import Decimal, ROUND_HALF_EVEN
from hashlib import file_digest
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent
DATA = OUT.parent
KEYS = ('rotation', 'dataset', 'model')
FIELDS = {'M':'M_seed', 'A':'A_seed', 'fallback_count':'full_spray_fallback_count_seed'}
MODELS = {'asset_detr':'ASSET-DETR', 'rtdetr_r18':'RT-DETR-R18', 'yolo26m':'YOLO26m'}
D = Decimal
MAIN = ('bbro', 'near30')
SUFFIXES = ('mean', 'min', 'max')
SETS = {
    'D008': ('D008_KEY_GROUPED_K50.csv', 'nozzles'),
    'D009': ('D009_KEY_GROUPED_K20_K50_K100.csv', 'alpha'),
    'D010': ('D010_KEY_GROUPED_K20_K50_K100.csv', 'coverage'),
}


def sha(path):
    with path.open('rb') as f:
        return file_digest(f, 'sha256').hexdigest()


def read(path):
    with path.open(newline='') as f:
        return [dict(r, line=i) for i,r in enumerate(csv.DictReader(f), 2)]


def write_csv(path, rows):
    fields = list(dict.fromkeys(c for row in rows for c in row))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def ident(row):
    return tuple(row[k] for k in KEYS)


def format_value(raw, scale=1, digits=1):
    value = (D(raw)*scale).quantize(D(1).scaleb(-digits), rounding=ROUND_HALF_EVEN)
    return str(abs(value) if value == 0 else value)


def md_table(headers, rows):
    return '\n'.join('|'+'|'.join(map(str,row))+'|' for row in
                     [headers, ['---']*len(headers), *rows])


def metric_block(row, fields, prefix=''):
    out = {}
    for metric in fields:
        for s in SUFFIXES:
            col = FIELDS[metric] + '_' + s
            field = prefix + metric + '_' + s
            out.update({field:row[col], field+'_source_column':col,
                        field+'_display':format_value(row[col],1 if metric=='fallback_count' else 100)})
    return out


def to_supplement(name, rows, refs):
    filename, param = SETS[name]
    out = []
    for r in sorted(rows, key=lambda x: (*ident(x),int(x['K']),x[param])):
        item = {k:r[k] for k in ('rotation','dataset','model','scheme','K','nozzles','alpha','coverage')}
        item.update(sources_filename=filename,sources_line=r['line'])
        item.update(metric_block(r,FIELDS))
        if name=='D008':
            ref = refs[(*ident(r),r['nozzles'])]
            item.update(reference_source_filename=filename,reference_source_line=ref['line'])
            item.update(metric_block(ref,['A'],'reference_'))
        out.append(item)
    write_csv(OUT/f'SUPPLEMENT_{name}.csv', out)


def stats(rows, alpha):
    passed = {ident(r) for r in rows if D(r['M_seed_mean'])<=alpha}
    full = {ident(r) for r in rows if D(r['full_spray_fallback_count_seed_mean'])>100}
    effective = {ident(r) for r in rows
                 if ident(r) in passed and D(r['full_spray_fallback_count_seed_mean'])<100}
    return passed,full,effective


def lb_detr(rows):
    return [r for r in rows if r['rotation']==r['dataset'] and r['model']!='yolo26m']


def evidence(row, col, scale=100, digits=1):
    raw = row[col]
    return f'{format_value(raw,scale,digits)} ({raw}, L{row["line"]} · {col})'


def span(rows, col='A_seed_mean'):
    low,high = (func(rows,key=lambda r:D(r[col])) for func in (min,max))
    return ' 至 '.join(evidence(r,col) for r in (low,high))


def count_blocks(rows, alpha):
    passed,full,effective = stats(rows,alpha)
    return [len(rows), len(passed), len(full), len(effective),
            ', '.join(str(r['line']) for r in rows)]


def missing_summary(rows, focus):
    result=[]
    for label, subset in (('全部轮换',[r for r in rows if r['nozzles']=='16']),
                          ('正文轮换',[r for r in rows if r['nozzles']=='16' and r['rotation'] in MAIN])):
        results = [r for r in subset if r['valid_repeats']=='0'
                   and all(r[c]=='' for c in ('M','A','WCR','mean_selected_threshold'))]
        assert len(results)==len(subset)
        configs = {(r['rotation'],r['model'],r['seed']) for r in results}
        focus_rows = [r for r in focus if r['nozzles']=='16' and r['method']=='source' and
                      (label=='全部轮换' or r['rotation'] in MAIN)]
        assert len(focus_rows)==18
        result.append([label, len(configs),len(results),len(configs)*3,
                       ', '.join(str(r['line']) for r in focus_rows)])
        if label=='全部轮换':
            # Only the full table includes walledgarden; the D008 key contains the two main rotations.
            result[-1][-1] = 'D008_FULL.csv 的 n=16、source、per_seed 行'
    return result


def figure_a(rows, refs):
    fig, ax = plt.subplots(figsize=(6.26,3.43),layout='constrained')
    x = [5,10,20]
    colors = {('bbro','asset_detr'):'#0072B2', ('bbro','rtdetr_r18'):'#009E73',
              ('near30','asset_detr'):'#D55E00', ('near30','rtdetr_r18'):'#78538B'}
    markers = ['o','s','^','D']
    ledger, legend_handles = [], []
    for idx,((rotation,model),color) in enumerate(colors.items()):
        series=sorted([r for r in rows if r['rotation']==rotation and r['model']==model],
                      key=lambda r:D(r['alpha']))
        assert [100*D(r['alpha']) for r in series]==x
        ys=[[100*float(r[f'A_seed_{s}']) for r in series] for s in SUFFIXES]
        average,low,high=ys
        ax.errorbar(x,average,yerr=[[a-l for a,l in zip(average,low)],[h-a for h,a in zip(high,average)]],
                    color=color,linestyle=['-','--','-.',':'][idx],linewidth=1.45,
                    capsize=3,elinewidth=.85,label=f'{rotation} · {MODELS[model]}')
        legend_handles.append(Line2D([], [], color=color, marker=markers[idx],
                                     ls=['-','--','-.',':'][idx],
                                     label=f'{rotation} · {MODELS[model]}', markersize=5))
        for r,xx,yy in zip(series,x,average):
            mostly=D(r['full_spray_fallback_count_seed_mean'])>100
            ax.scatter(xx,yy,marker='X' if mostly else markers[idx],s=42,zorder=4,
                       facecolor=color,edgecolor='white',linewidth=.65)
            ledger.append({**{k:r[k] for k in KEYS},'alpha':r['alpha'],'mostly_full_spray':mostly,
                           'source_filename':SETS['D009'][0],'source_line':r['line'],
                           **metric_block(r,['A','fallback_count'])})
    ends=[op(refs,key=lambda r:D(r['A_seed_mean'])) for op in (min,max)]
    ax.axhspan(*(100*float(r['A_seed_mean']) for r in ends),color='#B1C7D7',alpha=.45,label='GT-box trigger range')
    for edge,r in zip(('lower','upper'),ends):
        ledger.append({**{k:r[k] for k in KEYS},'reference_edge':edge,'source_filename':SETS['D008'][0],
                       'source_line':r['line'],**metric_block(r,['A'])})
    assert len(ledger)==14
    ax.text(11.9,7.8,f'GT-box triggers ({format_value(ends[0]["A_seed_mean"],100)}–{format_value(ends[1]["A_seed_mean"],100)}%)',
            va='center',ha='center',color='#274558',fontsize=8)
    ax.set(xticks=x,xticklabels=['0.05','0.10','0.20'],xlim=(3.1,22),ylim=(0,100),
           xlabel='Risk target α',ylabel='Simulated sprayed area A (%)')
    ax.set_axisbelow(True)
    ax.grid(axis='y',color='0.88',linewidth=.55)
    ax.spines[['top','right']].set_visible(False)
    if any(r.get('mostly_full_spray') for r in ledger):
        legend_handles.append(Line2D([],[],marker='X',ls='',color='0.3',label='Mostly full spray'))
    ax.legend(handles=legend_handles,loc='upper right',frameon=False,fontsize=8,ncols=2)
    save(fig,'FIG_A_RISK_TARGET_SPRAY_AREA')
    write_csv(OUT/'FIG_A_SOURCE_VALUES.csv',ledger)


def figure_b():
    fig, ax = plt.subplots(figsize=(6.26,2.6))
    fig.subplots_adjust(left=.012,right=.988,bottom=.04,top=.98)
    ax.set(xlim=(0,1),ylim=(0,1))
    ax.axis('off')
    dark, blue, fill = '#263746','#477B9A','#DBE9F2'
    xs=[.006,.266,.526,.786]
    width=.208
    titles=['Detector output','Confidence\nthreshold','Nozzle-strip\ntriggers','Evaluation']
    for x,label in zip(xs,titles):
        ax.add_patch(Rectangle((x,.29),width,.58,lw=.8,ec='#7C8C96',fc='#F8F7F2'))
        ax.text(x+width/2,.79,label,va='center',ha='center',fontsize=8.0,weight='bold',color=dark)
    for x,next_x in zip(xs,xs[1:]):
        ax.add_patch(FancyArrowPatch((x+width+.006,.573),(next_x-.005,.573),
                                     arrowstyle='-|>',mutation_scale=9,lw=.9,color=dark))
    # Synthetic objects and dashed prediction rectangles.
    x0=xs[0]
    for px,py,size in ((.064,.60,.014),(.151,.475,.012)):
        px+=x0
        ax.plot(px,py,marker='*',color='#A8821D',ms=8)
        ax.add_patch(Rectangle((px-size,py-.060),2*size,.120,fill=False,ls='--',lw=.85,ec=dark))
    ax.text(x0+width/2,.345,'Scored weed boxes',ha='center',va='center',fontsize=8,color=dark)
    ax.text(xs[1]+width/2,.617,'Source threshold',ha='center',va='center',fontsize=8,color=dark)
    ax.text(xs[1]+width/2,.529,'or',ha='center',va='center',fontsize=8,color=dark)
    ax.text(xs[1]+width/2,.434,'CRC with K labeled\ntarget-field images',ha='center',va='center',fontsize=8,color=dark)
    x=xs[2]
    left, bottom, w, h = x+.027,.401,.154,.305
    ax.add_patch(Rectangle((left,bottom),w,h,fill=False,lw=.8,ec=dark))
    # Scale the diagram's horizontal and vertical coordinates to equal physical lengths.
    prediction_width = .027
    center = left + w / 2
    spray_width = max(prediction_width, h / 4 * (2.6 * .94) / (6.26 * .976))
    ax.add_patch(Rectangle((center-spray_width/2,bottom+h/4),spray_width,h/2,lw=.6,ec=blue,fc=fill))
    for strip in (1,2,3):
        y=bottom+strip*h/4
        ax.plot([left,left+w],[y,y],ls='--',lw=.65,color='#687984')
    ax.add_patch(Rectangle((center-prediction_width/2,bottom+h/2-.041),prediction_width,.102,
                           fill=False,ls='--',lw=.85,ec=dark))
    ax.text(x+width/2,.345,'n strips',ha='center',va='center',fontsize=7.6,color=dark)
    x=xs[3]+width/2
    ax.text(x,.61,'Miss rate M',ha='center',va='center',fontsize=8,color=dark)
    ax.text(x,.514,'Simulated\nsprayed area A',ha='center',va='center',fontsize=7.6,color=dark)
    ax.text(x,.383,'Full-spray\nfallback count',ha='center',va='center',fontsize=7.8,color=dark)
    a,b=xs[1]+width/2,x
    ax.plot([a,a,.452],[.29,.14,.14],lw=.85,color=blue)
    ax.text(.64,.14,'No feasible CRC threshold → Full spray',ha='center',va='center',
            fontsize=8,color=dark,bbox=dict(boxstyle='round,pad=.33',facecolor=fill,edgecolor=blue,linewidth=.65))
    ax.plot([.823,b],[.14,.14],lw=.85,color=blue)
    ax.add_patch(FancyArrowPatch((b,.14),(b,.287),arrowstyle='-|>',mutation_scale=9,lw=.85,color=blue))
    save(fig,'FIG_B_WORKFLOW')


def save(fig, name):
    fig.savefig(OUT/(name+'.pdf'),dpi=600,facecolor='white')
    fig.savefig(OUT/(name+'.png'),dpi=600,facecolor='white')
    plt.close(fig)


def build():
    sources={name:read(DATA/spec[0]) for name,spec in SETS.items()}
    crc={name:[r for r in rows if r['rotation'] in MAIN and r['method']=='crc']
         for name,rows in sources.items()}
    assert [len(crc[name]) for name in SETS]==[54,162,108]
    refs = {(*ident(r),r['nozzles']):r for r in sources['D008']
            if r['method']=='perfect_detection_reference'}
    for name,rows in crc.items():
        to_supplement(name,rows,refs)
    s8=[]
    for n in ('4','8','16'):
        group=[r for r in crc['D008'] if r['nozzles']==n]
        lb=lb_detr(group)
        ref=[r for r in refs.values() if r['rotation']==r['dataset'] and r['nozzles']==n]
        s8.append([n,*count_blocks(group,D('.1')),span(lb),span(ref)])
    lb8 = sorted(lb_detr(crc['D008']),key=lambda r:(int(r['nozzles']),r['rotation'],r['model']))
    lb8_table=[[r['nozzles'],r['rotation'],MODELS[r['model']],evidence(r,'full_spray_fallback_count_seed_mean',1)]
               for r in lb8]
    pheno=[[r['nozzles'],MODELS[r['model']],evidence(r,'A_seed_mean'),evidence(r,'M_seed_mean'),
            evidence(r,'full_spray_fallback_count_seed_mean',1)]
           for r in crc['D008'] if r['rotation']=='bbro' and r['dataset']=='PhenoBench' and
           r['model']!='yolo26m' and r['nozzles'] in ('4','8')]
    unavailable = missing_summary(
        [r for r in read(DATA/'D008_FULL.csv') if r['aggregation']=='per_seed' and r['method']=='source'],
        sources['D008'])
    s9,s10,changed = [],[],[]
    for k in ('20','50','100'):
        for alpha in ('0.05','0.1','0.2'):
            group=[r for r in crc['D009'] if r['K']==k and r['alpha']==alpha]
            s9.append([k,alpha,*count_blocks(group,D(alpha)),span(lb_detr(group)),span(lb_detr(group),'M_seed_mean')])
        pairs=[]
        for coverage in ('complete','0.9'):
            group=[r for r in crc['D010'] if r['K']==k and r['coverage']==coverage]
            s10.append([k,coverage,*count_blocks(group,D('.1')),span(lb_detr(group))])
            pairs.append(stats(group,D('.1'))[0])
        changed.append([k,len(pairs[0]&pairs[1]),len(pairs[0]-pairs[1]),len(pairs[1]-pairs[0]),
                        ', '.join('/'.join(fields) for fields in sorted(pairs[0]))])
    alpha_20=[r for r in crc['D009'] if r['K']=='50' and r['alpha']=='0.2']
    misses=[r for r in alpha_20 if D(r['M_seed_mean'])>D('.2')]
    assert len(misses)==5
    per_seed=read(DATA/'D009_FULL.csv')
    mc=[]
    for r in misses:
        group=[s for s in per_seed if s['aggregation']=='per_seed' and ident(s)==ident(r)
               and all(s[k]==r[k] for k in ('K','alpha','scheme','nozzles','method'))]
        group.sort(key=lambda s:s['seed'])
        assert [s['seed'] for s in group]==['0','1','2']
        close=abs(D(r['M_seed_mean'])-D('.2')) <= 2*D(r['M_MC_SE_seed_mean'])
        mc.append([*ident(r),evidence(r,'M_seed_mean'),evidence(r,'M_MC_SE_seed_mean',100,3),
                   '是' if close else '否',
                   '<br>'.join(f'{s["seed"]}: {evidence(s,"M",100,3)} / {evidence(s,"M_MC_SE",100,3)}'
                                + ('（在 2SE 内）' if abs(D(s['M'])-D('.2')) <= 2*D(s['M_MC_SE']) else '（在 2SE 外）')
                               for s in group)])
    figure_a(lb_detr([r for r in crc['D009'] if r['K']=='50']),
             [r for r in refs.values() if r['rotation']==r['dataset'] and r['nozzles']=='4'])
    figure_b()
    parts=[
        '# D008–D010 写作用核对报告',
        '范围为 bbro 与 near30 两个轮换、分组划分、18 个正文评价格。用跨种子均值的 CSV 原值判断达标、主要靠全喷（平均退回严格大于 100 次）和有效点喷（达标且平均退回严格小于 100 次）。'
        ' M 与 A 在表内以百分数显示，括号是原比例，回退次数保留一位小数。**以下按十进制 round half to even 舍入。**'
        ' 每段的 L 指该段来源 CSV 的物理行号，表头为 L1。格数来源为所列行的 `M_seed_mean`、`full_spray_fallback_count_seed_mean`、`alpha`。',
        '## D008 · 喷头数',
        '附件所列 D008 格数、范围、四个 DETR 回退序列和 PhenoBench 两组对照全部一致。来源为 `D008_KEY_GROUPED_K50.csv`。',
        md_table(['n','格数','达标','主要靠全喷','有效点喷','格数来源行','DETR 类 LB A','真值框参照 LB A'],s8),
        md_table(['n','轮换','检测器','平均回退次数'],lb8_table),
        md_table(['n','bbro → PhenoBench','A (%)','M (%)','平均回退次数'],pheno),
        'n=16 时，源留出集的阈值搜索对每个配置均返回不可用。源方法只接受存在按棵 WCR ≥ 0.90 的网格阈值；若可行集为空，M、A、WCR 留空且 `valid_repeats=0`。'
        ' `D008_FULL.csv` 中按 `aggregation=per_seed, method=source, nozzles=16` 筛得 810 行，每个训练配置对应 3 个目标、2 种划分、5 档 K。'
        ' 正文 18 个逐种子训练配置占 540 行。跨种子评价格分别为 27 个（全范围）和 18 个（正文）。'
        ' 阈值规则见工作区 `analysis/posthoc_20261114/run.py` 第 103、174、250 行。源阈值采用源留出集统一选取。',
        md_table(['范围','训练配置','全量逐种子行数','逐种子配置 × 目标','重点表行'],unavailable),
        '## D009 · 风险目标',
        '附件所列全部格数、面积和 M 范围均一致。来源为 `D009_KEY_GROUPED_K20_K50_K100.csv`。',
        md_table(['K','α','格数','达标','主要靠全喷','有效点喷','格数来源行','DETR 类 LB A','DETR 类 LB M'],s9),
        'K=50、α=0.20 时，五格按未舍入 M 均超标。表中平均 SE 取已有 `M_MC_SE_seed_mean`（三种子各 200 次抽样的 SE 等权平均），'
        '以该字段为描述性尺度检查 |M−0.20| ≤ 2 × 平均 SE。每个种子的 M 与 SE 列于末列，来自 `D009_FULL.csv`。'
        ' 该尺度仅描述种子内的重复抽样波动。三个训练种子共用抽样，跨种子均值的正式蒙特卡洛 SE 还需要同次抽样的种子间协方差，当前汇总表无法给出。',
        md_table(['轮换','目标','模型','M (%)','平均种子 SE (pp)','距目标 ≤ 2 平均 SE','种子号及 M (%) / SE (pp)'],mc),
        '## D010 · 覆盖判据',
        '附件所列格数和区间全部一致。来源为 `D010_KEY_GROUPED_K20_K50_K100.csv`。'
        ' 每个 K 的两种判据逐格比较，达标集合完全一致。格子键依次为轮换／目标／模型。',
        md_table(['K','判据','格数','达标','主要靠全喷','有效点喷','格数来源行','DETR 类 LB A'],s10),
        md_table(['K','共同达标','仅完全包含达标','仅 ≥90% 达标','共同达标格子键'],changed),
        '## 图件与补充表',
        '图 A 的 12 个点及参考带端点可回溯至 `FIG_A_SOURCE_VALUES.csv`。其 12 个点的平均回退均未超过 100 次，图注说明标记规则和实际情况。'
        ' 图 B 参照当前 IPA 稿 Fig. 1 的虚线检测框、蓝色喷带和浅色背景，使用合成示意元素。'
        ' 三张补充表的 M、A 保留比例原值，`_display` 列为百分数（回退为次数），并附种子极值及来源行列。',
    ]
    (OUT/'DERIVED_NUMBERS_AUDIT.md').write_text('\n\n'.join(parts)+'\n')
    (OUT/'FIGURE_CAPTIONS_EN.md').write_text(
        '# Draft figure captions\n\n'
        '**Figure A. Risk target and simulated sprayed area (post hoc analysis D009).** '
        'Points show the mean across three training seeds for each detector on each held-out Lincoln Beet field '
        'with grouped calibration, K = 50, and n = 4. Error bars show the minimum and maximum across seeds. '
        'An X would denote a mean of more than 100 full-spray fallbacks per 200 draws. None of the plotted cells met that criterion. '
        'The shaded band shows the range of simulated sprayed area for the ground-truth-box trigger reference on the two held-out fields.\n\n'
        '**Figure B. From detector predictions to spot-spraying decisions and evaluation.** '
        'Weed predictions at or above the confidence threshold trigger a spray zone in each strip intersected by the predicted box. '
        'The threshold can be set on source-field data or recalibrated by conformal risk control (CRC) using K labeled target-field images. '
        'If CRC finds no feasible threshold, the simulated policy switches to full spray. '
        'Evaluation summarizes the mean missed-weed fraction in weed-positive images (M), simulated sprayed area (A), '
        'and the number of full-spray fallbacks across calibration draws. The graphic is schematic.\n'
    )


if __name__=='__main__':
    matplotlib.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.linewidth':.75})
    build()
