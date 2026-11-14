"""Render the registered sensitivity tables with source row/column provenance."""

from collections import defaultdict
import csv
from decimal import Decimal, ROUND_HALF_EVEN
from run import OUT, PLAN_SHA, FROZEN_SHA, REPO, digest, read

FORMAL = REPO / 'results/CROSS_SEED_ALL_RESULTS.csv'
KEY = ('rotation', 'model', 'dataset', 'method', 'scheme', 'nozzles', 'K')
M = 'M_seed_mean'
A = 'A_seed_mean'
F = 'full_spray_fallback_count_seed_mean'
BASE_COLUMNS = {
    'M': 'mean_actual_weed_image_miss_rate', 'A': 'mean_spray_area_fraction',
    'full_spray_fallback_count': 'full_spray_fallback_count',
    'WCR': 'mean_actual_WCR', 'K_prime_mean': 'K_prime_mean',
}
BASE_METRICS = {name + suffix: formal + suffix for name, formal in BASE_COLUMNS.items()
                for suffix in ('_seed_mean', '_seed_min', '_seed_max')}


def table_input(path, select):
    with path.open(newline='') as f:
        return [dict(row, source_file=path.name, source_row=line)
                for line, row in enumerate(csv.DictReader(f), 2) if select(row)]


def key(row):
    return tuple(row[k] for k in KEY)


def base_row(row):
    required = {k:row[v] for k,v in BASE_METRICS.items()}
    return {**row, **required, 'alpha':'0.1', 'coverage':'complete'}


def decimal(text):
    return Decimal(text) if text not in ('', None) else None


def rounded(value, scale=100, digits=1):
    value = decimal(value)
    if value is None:
        return 'NA'
    result = (value * scale).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_EVEN)
    return str(result if result else result.copy_abs())


def collect(row, baseline=False):
    result = {k:row[k] for k in KEY}
    result.update(alpha=row['alpha'], coverage=row['coverage'],
                  source_file=row['source_file'], source_row=row['source_row'])
    for metric in BASE_METRICS:
        if row['method'] == 'perfect_detection_reference' and not metric.startswith('A_'):
            continue
        if not metric.startswith('A_') and not (baseline or metric in row):
            continue
        if metric not in row:
            continue
        raw = row[metric]
        proportion = metric.startswith(('M_', 'A_', 'WCR_'))
        result.update({metric:raw, metric+'_rounded':rounded(raw, 100 if proportion else 1,
                       2 if metric.startswith('K_prime_mean') else 1),
                       metric+'_source_column':BASE_METRICS[metric] if baseline else metric})
    for metric in ('M', 'A'):
        for suffix in ('_seed_mean', '_seed_min', '_seed_max'):
            column = metric + '_MC_SE' + suffix
            original = 'weed_image_miss_rate_MC_standard_error' + suffix
            if baseline and metric == 'M' and row['method'] != 'perfect_detection_reference':
                result.update({column:row[original], column+'_source_column':original})
            elif not baseline and column in row:
                result.update({column:row[column], column+'_source_column':column})
    for suffix in ('_seed_mean', '_seed_min', '_seed_max'):
        column = 'mean_selected_threshold' + suffix
        if not baseline and row.get(column) not in (None, ''):
            result.update({column:row[column], column+'_source_column':column})
    return result


def main_scope(row):
    return row['rotation'] in ('bbro','near30')


def in_grouped(row, ks):
    return row['scheme']=='grouped' and row['K'] in ks


def select_full(path, ks):
    return table_input(path, lambda r:r['aggregation']=='cross_seed' and in_grouped(r,ks))


def to_markdown(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                     '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                     *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]])


def shown(row, metric, scale=100):
    if decimal(row.get(metric)) is None:
        return 'NA'
    half_even = rounded(row[metric], scale)
    return f'{half_even} ({row[metric]})'


def source(row, metric):
    if metric.endswith('_change') or metric == 'A_minus_reference':
        return f'{row["key_file"]}:{row["key_row"]} · {metric}'
    columns = row.get(metric + '_source_column',metric)
    return f'{row["source_file"]}:{row["source_row"]} · {columns}'


def span(rows, metric, scale=100, digits=1):
    valid = [row for row in rows if decimal(row.get(metric)) is not None]
    if not valid:
        return 'NA'
    ends = [pick(valid, key=lambda r:decimal(r[metric])) for pick in (min,max)]
    return ' 至 '.join(f'{rounded(r[metric],scale,digits)}（{r[metric]}，{source(r,metric)}）'
                       for r in ends)


def count_rows(rows, alpha):
    valid = [r for r in rows if decimal(r[M]) is not None]
    return (sum(decimal(r[M]) <= Decimal(alpha) for r in valid), len(valid),
            sum(decimal(r[F]) > 100 for r in valid if decimal(r[F]) is not None))


def group(rows, fields):
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row[k] for k in fields)].append(row)
    return buckets


def add_delta(nrows, official, metrics):
    for row in nrows:
        base_key = tuple('4' if k=='nozzles' else row[k] for k in KEY)
        old = official[base_key]
        row.update(baseline_source_file=old['source_file'], baseline_source_row=old['source_row'])
        for metric in metrics:
            before, after = decimal(old[metric]), decimal(row[metric])
            delta = after-before if before is not None and after is not None else None
            row.update({metric+'_baseline':before, metric+'_baseline_rounded':rounded(before),
                        metric+'_baseline_column':BASE_METRICS[metric],
                        metric+'_change':delta, metric+'_change_rounded_pp':rounded(delta)})


def build(refresh=False):
    regression, compute = (read(OUT / name) for name in ('REGRESSION_STATUS.json', 'COMPUTATION_STATUS.json'))
    assert regression['status']=='PASSED' and compute['status']=='COMPLETED'
    assert regression['ended_at'] <= compute['started_at']
    official = {key(r):collect(base_row(r),True) for r in table_input(FORMAL,
                lambda r:r['nozzles']=='4' and in_grouped(r,('20','50','100')))}
    k50 = ('50',)
    d008new = [collect(r) for r in select_full(OUT/'D008_FULL.csv',k50) if main_scope(r)]
    d008base = [r for r in official.values() if main_scope(r) and r['K']=='50' and
                r['method'] in ('source','crc','perfect_detection_reference')]
    add_delta([r for r in d008new if r['method']=='crc'], official, (M,A))
    d008 = sorted(d008base+d008new,key=lambda r:(r['rotation'],r['dataset'],r['model'],r['method'],int(r['nozzles'])))
    assert len(d008)==162
    bucket = group(d008, ('rotation','dataset','model','nozzles'))
    for rows in bucket.values():
        cells = {r['method']:r for r in rows}
        crc, ref = (cells[m] for m in ('crc','perfect_detection_reference'))
        diff = decimal(crc[A]) - decimal(ref[A])
        crc.update(A_minus_reference=diff,A_minus_reference_rounded_pp=rounded(diff),
                   reference_source_file=ref['source_file'],reference_source_row=ref['source_row'],
                   reference_source_column=ref[A+'_source_column'])
    ks = ('20','50','100')
    d009 = sorted([collect(r) for r in select_full(OUT/'D009_FULL.csv',ks)] +
                  [r for r in official.values() if r['method']=='crc'],key=lambda r:(int(r['K']),r['alpha'],r['rotation'],r['dataset'],r['model']))
    assert len(d009)==243
    d010new = [collect(r) for r in select_full(OUT/'D010_FULL.csv',ks)]
    add_delta(d010new,official,(M,A))
    d010 = sorted([r for r in official.values() if r['method'] in ('source','crc')] + d010new,
                  key=lambda r:(int(r['K']),r['rotation'],r['dataset'],r['model'],r['method'],r['coverage']))
    assert len(d010)==324
    keys = ['D008_KEY_GROUPED_K50.csv','D009_KEY_GROUPED_K20_K50_K100.csv','D010_KEY_GROUPED_K20_K50_K100.csv']
    key_rows = [d008,d009,d010]
    for name, rows in zip(keys,key_rows):
        for line,row in enumerate(rows,2):
            row.update(key_file=name,key_row=line)
        with (OUT/name).open('w' if refresh else 'x', newline='') as file:
            fields = list(dict.fromkeys(col for row in rows for col in row))
            writer = csv.DictWriter(file, fields)
            writer.writeheader()
            writer.writerows(rows)
    tab8 = []
    for n in ('4','8','16'):
        crc = [r for r in d008 if r['method']=='crc' and r['nozzles']==n]
        lb = [r for r in crc if r['rotation']==r['dataset'] and r['model'] in ('asset_detr','rtdetr_r18')]
        ref = [r for r in d008 if r['method']=='perfect_detection_reference' and r['nozzles']==n]
        lb_ref = [r for r in ref if r['dataset']==r['rotation']]
        tab8.append([n,span(lb,A),span(ref,A),span(lb_ref,A),span(crc,'A_minus_reference'),
                     span(lb,'A_minus_reference')])
    detail = []
    models = {'asset_detr':'ASSET-DETR','rtdetr_r18':'RT-DETR-R18','yolo26m':'YOLO26m'}
    for (rotation,dataset,model), rows in sorted(group([r for r in d008 if r['method']=='crc'],
                                                      ('rotation','dataset','model')).items()):
        cols = [rotation,dataset,models[model]]
        for row in sorted(rows,key=lambda r:int(r['nozzles'])):
            cols.append(f'{shown(row,M)} / {shown(row,A)} / {shown(row,F,1)} · L{row["key_row"]}')
        detail.append(cols)
    tab9 = []
    tab10 = []
    for k in ks:
        for alpha in ('0.05','0.1','0.2'):
            rr = [r for r in d009 if r['K']==k and r['alpha']==alpha]
            for label, subset in (('正文', [r for r in rr if main_scope(r)]), ('全范围',rr)):
                ok, valid, fallback = count_rows(subset,alpha)
                tab9.append([k,alpha,label,f'{ok}/{valid}',span(subset,A),f'{fallback}/{valid}',
                             ', '.join(str(r['key_row']) for r in subset)])
        for label in ('正文','全范围'):
            for method in ('crc','source'):
                rr = [r for r in d010 if r['method']==method and r['K']==k and
                       (label=='全范围' or main_scope(r))]
                old=[r for r in rr if r['coverage']=='complete']
                new=[r for r in rr if r['coverage']=='0.9']
                oc, ov, _ = count_rows(old,'0.1')
                nc, nv, _ = count_rows(new,'0.1')
                tab10.append([k,label,method,f'{oc}/{ov} → {nc}/{nv}',span(new,M+'_change'),
                              span(new,A+'_change'),', '.join(str(r['key_row']) for r in rr)])
    # All draws at K=5 and K=10 must fail the 5% CRC correction.
    mandatory = table_input(OUT/'D009_FULL.csv',lambda r:r['aggregation']=='per_seed'
                             and r['alpha']=='0.05' and r['K'] in ('5','10'))
    assert len(mandatory)==324 and all(r['valid_repeats']=='200' and r['full_spray_fallback_count']=='200' for r in mandatory)

    contents = [
        '# ReCal 事后分析 D008–D010',
        '三项分析均在正式结果和 D005–D007 已知后登记。本报告只作描述。',
        '## 计划与回归',
        f'- 登记时间 **{regression["registered_at"]}**。修订 2 SHA256 为 `{PLAN_SHA}`。',
        f'- 冻结计划 SHA256 复核为 `{FROZEN_SHA}`。',
        f'- 回归从 {regression["started_at"]} 到 {regression["ended_at"]}。'
        f' 逐种子全量表、跨种子全量表和主表合计比较 {regression["rows_checked"]} 行'
        f'（1,620 + 540 + 72），超出容差 {regression["bad_rows"]} 行，最大绝对差 '
        f'`{regression["maximum_absolute_difference"]}`。容差为 1e-9。'
        ' 逐格核对见 `REGRESSION_COMPARISON.csv`，其中记录原表行号、列名、原值与复算值。',
        f'- 重新核对 {regression["labels_verified"]:,} 个本地真值文件的历史 SHA256，并核验正式三种轮换划分、'
        '导出预测文件、原抽样索引和 n=4 几何缓存。源阈值用原定义在源留出集重选。主 CRC 从原抽样校准集重选。',
        '- 第一次回归启动在加载本地模块时遇到导入路径错误，未进行任何回归比较。'
        '只修正新增入口的模块搜索路径后，回归完整运行一次并通过。日志保留为 `REGRESSION.log` 和 `REGRESSION_RUN.log`。',
        '- 新增几何测试通过 2 项，覆盖 75 组喷带与置信度曲线对照和四个 90% 边界例。'
        '直接对照测试按阈值独立构建喷区并集，以有理数求交核验命中数。',
        f'- D008–D010 计算从 **{compute["started_at"]}** 到 **{compute["ended_at"]}**。'
        f' 计算阶段累计 CPU 时间 {compute["cpu_seconds"]:.2f} 秒，经过 {compute["wall_seconds"]:.2f} 秒。'
        f' 回归累计 CPU 时间 {regression["cpu_seconds"]:.2f} 秒，经过 {regression["wall_seconds"]:.2f} 秒。'
        ' 原始计时及逐配置用时见 `COMPUTATION_STATUS.json` 和 `REGRESSION_STATUS.json`。',
        '每个全量 CSV 同时收录逐种子与跨种子结果，由 `aggregation` 区分。跨种子字段给出均值、种子最小值和最大值。'
        ' M、A、WCR、回退比例和 K′ 的逐种子蒙特卡洛标准误按抽样标准差（ddof=1）除以 √200，'
        '有定义的抽样数少于 200 时留空。跨种子 SE 字段只是逐种子 SE 的汇总。'
        '正式基线的重点表附原发布结果已有的 M 标准误。',
        '下表百分数和百分点采用十进制 round half to even。括号内值为 CSV 未舍入原值。'
        ' L 表示对应重点 CSV 自表头起算的行号，重点 CSV 给出每个值的原始文件、行、列及舍入值。'
        ' 达标按跨种子平均 M 判定。回退超过一半按跨种子平均回退次数严格大于 100 判定。',
        '## D008 喷头数',
        '以下为分组、K=50、18 个正文评价格。每组数值依次为 M（%）、A（%）和平均回退次数（每 200 次抽样）。'
        '原值列保留比例单位，来源行见 `D008_KEY_GROUPED_K50.csv`。',
        to_markdown(['轮换','目标','检测器','n=4','n=8','n=16'],detail),
        '以下 A 范围用百分数表示，面积差用百分点表示。DETR 类留出田范围包含两种模型和两个正文轮换。'
        '外部图像的喷带宽度是图像坐标中的模拟值，无法对应到具体机型。',
        to_markdown(['n','DETR 类留出田 A','真值框参照 A（全部正文格）','真值框参照 A（LB）',
                     'CRC 减参照的 A（全部正文格）','CRC 减参照的 A（DETR 类 LB）'],tab8),
        '## D009 风险目标',
        '式（8）要求 1/(K′+1) ≤ α，故 α=0.05 时 K′ 至少为 19，α=0.20 时至少为 4。'
        'α=0.05、K 为 5 或 10 时全部退回全喷。逐种子全量表的 324 行（两种划分）均为 200/200 次回退。',
        '达标数和回退数由 `D009_KEY_GROUPED_K20_K50_K100.csv` 所列行的'
        ' `M_seed_mean` 和 `full_spray_fallback_count_seed_mean` 计数。面积范围显示百分数。',
        to_markdown(['K','α','范围','达标格数','A 范围','回退超过一半','重点 CSV 行'],tab9),
        '## D010 覆盖判据',
        '0.9 判据用各喷带的合并区间与框求交，跨喷带求和。阈值在对应判据下重新选取后，才在原测试集上评估。'
        '达到 90% 的边界计入覆盖。数值非常接近边界时，对输入坐标做有理数比较。'
        '下表变化为 0.9 判据减去完全包含判据，单位为百分点。',
        '计数与逐格原值见 `D010_KEY_GROUPED_K20_K50_K100.csv` 所列行。达标使用 `M_seed_mean`，'
        '变化用 `M_seed_mean_change`、`A_seed_mean_change`。若源阈值不存在，测试指标留空，并按有定义的格数给出分母。',
        to_markdown(['K','范围','方法','达标格数','M 变化','A 变化','重点 CSV 行'],tab10),
        '## 文件与核验',
        '所有新增代码和数值输出分别位于工作区根目录下 `analysis/posthoc_20261114/` 与 '
        '`results/posthoc_20261114/`，登记文件位于 `plan/`。公开仓库 v1.1.0 为只读输入。',
        to_markdown(['分析','全量逐种子行数','全量跨种子行数','重点表行数'],
                    [[name, counts['per_seed'],counts['cross_seed'],len(rows)]
                     for (name,counts),rows in zip(compute['rows'].items(),key_rows)]),
    ]
    with (OUT/'POSTHOC_REPORT_20261114.md').open('w' if refresh else 'x') as f:
        f.write('\n\n'.join(contents)+'\n')
    with (OUT/'RESULT_SHA256SUMS.txt').open('w' if refresh else 'x') as f:
        for path in sorted(OUT.iterdir()):
            if path.is_file() and path.suffix in ('.csv','.md','.json'):
                f.write(f'{digest(path)}  {path.name}\n')


if __name__=='__main__':
    build()
