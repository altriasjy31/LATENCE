请继续协助LATENCE第二阶段NBS研究。先阅读我附上的latence_nbs_v085.zip、README_v085.md、ANALYSIS_v085.md及新的实验结果，再作判断。不要仅依据历史摘要推断新指标结果。当前目标是核查评估、验证有限参数/采样改动，再决定是否迁回原完整版图模型；不是继续无边界堆版本或直接长训。v085 ZIP是增量包，依赖已安装v084；若基础源码不可访问，需要读取已有v084相关包或请我提供当前相关源码，不能凭摘要编造实现。

## 研究目标与不可丢失的约束

- 最终图预测G必须超过Expert/external probability E的Fmax和AUPRC，同时比较第一阶段Modelout M以及Backbone B。E/M/B必须分开，不能把略超过B等同最终目标达成。
- weak–GO和weak–core–GO都是weak-to-strong的核心路径，不能只保留后者。
- E/M只用于评估参考，不允许直接混入图前向或作为dense软目标。weak的二值pseudo来自Stage1 Modelout原始>0.5的CSR成员关系，保留这种离散监督；不要再次引入v082双塔/classification/dense teacher策略。
- 长期希望把有效机制迁回原完整异质图模型，但当前v084/v085简化模型用于归因。不要仅凭模型更大/epoch更长宣称会更好。
- 实验时间紧张，先复用预测缓存，保持少量可归因对照；避免新增大批辅助脚本、额外审批或询问已经授权的常规动作。

## 当前数据和训练方式

BP独立测试1800蛋白、21312个GO列；训练registry549722蛋白，其中weak489222、core60500。Stage2 core holdout1024，Stage1已见过这些core蛋白；因此该holdout只作monitor，不能承担独立泛化checkpoint选择。

v084：从真实全量来源图的候选池在线采样，两层关系SAGE，full-GO matcher。5类关系为ppi、similar_to、weak_to_core、core_to_weak、cosine；firsthop32、secondhop4、pool64。fixed/dynamic共享loss。默认每卡weak64/core16，两卡每步128/32。loss按角色归一化，weak/core总权重50/50，不是80/20。2000步约0.523个weak遍历，一轮需3823步。

v084只有前25步warmup，之后LR恒定3e-4。旧v071是OneCycle，不能把不同版本step/epoch或耗时直接横比。v084训练种子25%cosine-only第一跳，外部种子和当前core holdout第一跳全部cosine；后续邻居仍有原生图。Dynamic确实采到约28%位于fixed前缀外的邻居，但该数不等于跨epoch新增覆盖率。

## 已确认的评估问题

实际helper.py SHA256：70135074954cf10a084c75e215908f3fa106595d6bbf4c24e4cc3f2e62b7f4ef，与原结果一致。

`evalperf_torch`的Fmax是对整个N×GO矩阵汇总的micro-Fmax，扫描101阈值，并非逐蛋白Fmax；同文件fmax_torch/fmax_score才是逐蛋白口径。其AUPRC是101阈值PR点排序后trapz，空预测precision=0，存在网格/端点敏感性，不是精确AP或按唯一分数计算的精确PR-AUC。

原函数反例：y=[1,0,1,0]，p=[.9,.8,.7,.6]，旧Fmax80/AUPRC50；单调变为.01*p+.795后旧Fmax66.667/AUPRC25，exact AP始终83.333。该反例不代表真实数据偏差大小。

不能据旧主表AUPRC低于B就断言全部排序变差，也不能将更换指标当成模型提升。必须用原概率矩阵同时重算E/M/B/G。没有真实概率矩阵时，不得编造新逐蛋白Fmax或精确PR-AUC。

## v084实测结果（历史口径，百分数）

方法/step：历史micro-Fmax；历史网格AUPRC；exact micro-AP
B：55.6317；32.0087；49.4508
E：69.7811；56.4960；67.5547
M：69.7716；59.2118；68.7907
fixed600：56.7943；30.2616；51.9240
fixed2000：56.5506；30.3789；51.6110
dynamic600：56.9094；30.2339；52.0686
dynamic2000：56.5023；30.3290；51.5325

600→2000，历史AUPRC略升，但exact AP和Top-k下降，所以不能全归因指标实现。训练loss持续下降、残差幅度增加，不能再说loss完全没优化或梯度未启动。

step2000 full减去对应推理干预（Fmax / exact AP，百分点）：
fixed：weak_off −0.136/−0.434；core_off +0.763/+1.921；pp_off +0.325/+0.527；graph_off +0.857/+1.846；go_shuffle +0.441/+0.477。
dynamic：weak_off −0.139/−0.328；core_off +0.691/+1.935；pp_off +0.296/+0.375；graph_off +0.900/+1.784；go_shuffle +0.300/+0.292。

图已有正贡献，尤其core证据和PP传播。weak_off同时关闭种子candidate、邻居candidate和邻居pseudo，不能据此否定weak–GO或定位为pseudo单独有害。这些是推理输入依赖，不是去掉模块重训的因果贡献，不能相加。

core holdout的B是micro-Fmax98.0975/histAP99.8291；2000步fixed97.5142/99.6246、dynamic97.5122/99.6197。full−core_off的holdout Fmax为负（约−.358/−.333），独立测试却为正，故该holdout不适合作图路线选择。

难gold字段旧版约1.289是每rank16个core的批次总数，不是每蛋白；约.081个/蛋白。旧难goldcoverage是批比例平均，不能当全局覆盖率。图输入32邻居，但固定PU保护仍为cosine top8，可能有模型证据与目标压力不一致，尚未证实未标注位置是真正例。

## v085已交付代码与接口

这是新增训练/评估工具包，版本0.8.5，复用模型/数据实现0.8.4；原v084和helper.py均不修改。安装到已含v084的latence-project：
`python ../latence_nbs_v085/install_v085.py --project-root .`
配置stage.name仍为nbs_v084_*是复用数据协议，release_version/runner为0.8.5。

1. 缓存重算，不加载checkpoint/模型：
`COMPARISON_PATH=outputs/latence_nbs_experiments/v084/fixed/eval_step2000/full/metrics/nbs_w2s_comparison.json bash scripts/nbs/run_nbs_v085.sh recompute`
输出原metrics目录下metric_audit_v085/metric_audit_v085.json/tsv；保留legacy primary，新增methods.*.standardized和E/M/B差值。重算600/2000两组full，再重算已有图消融。需服务器上的概率、metadata、manifest和ID文件；仅摘要JSON不足。

2. 指标：standard_protein_fmax（0.001严格阈值网格，同时0.01网格）；standard_micro_ap；standard_micro_pr_auc；standard_micro_fmax_exact。AP和PR面积分开。逐蛋白Fmax排除空gold蛋白；全部GO列保留；不自动GO概率传播或校准，不宣称完整CAFA协议。新实验预先关注逐蛋白Fmax+exact PR-AUC，AP和legacy并列报告。

3. 预置constant/schedule/lowlr/aligned/dynamic。schedule固定4000步horizon，25步warmup，3e-4→3e-5 cosine；lowlr为1e-4→1e-5；aligned只相对lowlr把native_dropout设1；dynamic只相对aligned改变sampler。`STEPS`仅停止预算，不重定horizon。先运行2000，再可同实验续到4000。不能用RESUME改LR、loss、采样或开发集身份，不能v084续到v085。
示例：`STEPS=2000 bash scripts/nbs/run_nbs_v085.sh train schedule`；smoke默认20独立目录；默认2卡。

4. 可选开发集：prepare_nbs_development_v085.py绑定既有prepared inputs、E/M refs、二值gold和明确protein/GO IDs，要求完整Stage1训练ID与最终test ID排除，运行时额外排除整个Stage2 registry。DEV_MANIFEST传给shell；标签/E/M不入forward。脚本不生成独立标注或自动切test。Stage1 ID列表完整性和同源/时间隔离仍需研究者保证。
仅在开发G proteinFmax>=B且exact PR-AUC>B时，按PR-AUC保存best_development.pt；不是自动迁移成功或超过Expert的证明。无dev仍可运行预设预算，仅有core monitor。

5. 新日志含固定scheduler、core/weak遍历、难gold累积分子分母。只有可信开发集才用于参数/checkpoint选择；独立test仅按预先约定报告。

本地验证是94项CPU测试，包括真实小图训练导出/缓存重算、source绑定、日程精确2+2恢复和开发集隔离；无用户全量BP/A100/DDP运行结论，不保证性能提高。

## 下一步任务

请先阅读本次附件中的最新结果，区分legacy与standardized字段。优先完成缓存指标审计，再比较schedule/lowlr及相应第一跳对齐实验。不要默认五组都长训或把independent test当超参选择集。确认重复种子与图特异性收益后，才做原完整图骨干等预算迁移对照；若仍无改善，优先审查weak三类输入融合、可信难gold来源与PU冲突，再考虑SeHGNN式预聚合、关系采样或ego-net Transformer。请给具体、可测试、成本可控的代码修改，不保证尚未验证的性能提升。
