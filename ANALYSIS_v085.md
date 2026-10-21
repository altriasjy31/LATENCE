# v085：评估口径核查与下一轮实验决策

本报告基于 v084 fixed/dynamic 的 step 600/2000 对照、训练记录，以及本次新增的两份 validation_history、两份 nbs_graph_effects 和 helper.py。未访问用户服务器，未重新计算真实 BP 预测矩阵。

## 1. 已确认的评估口径问题

上传 helper.py 的 SHA256 为 `70135074954cf10a084c75e215908f3fa106595d6bbf4c24e4cc3f2e62b7f4ef`，与评估 JSON 记录完全一致。

`evalperf_torch` 对整个 N×GO 矩阵汇总 TP、预测阳性数和真阳性数，得到 micro precision/recall，扫描 0、0.01、…、1 的 101 个阈值。主表 Fmax 是 micro-Fmax，而不是逐蛋白平均 precision/recall 后计算的 Fmax。同文件中的 `fmax_torch` / `fmax_score` 才采用逐蛋白口径。

其 AUPRC 对上述 101 个 PR 点按 recall 排序后做梯形积分；空预测点 precision=0，没有标准 PR 曲线的空集 precision=1 端点，重复 recall 的排序也可能影响梯形连接。这是粗阈值网格的面积，不是 exact AP，也不是按所有唯一分数计算的 exact PR-AUC。排序不变但概率压缩到一个小区间时，该数值可大幅变化。

隔离执行原函数的反例：标签 `[1,0,1,0]`，分数 `[.9,.8,.7,.6]`，原主表 Fmax=80、AUPRC=50；严格单调变换为 `.01*p+.795` 后，主表 Fmax=66.667、AUPRC=25，而两者 exact AP 都为83.333。该反例证明网格敏感性，不是对真实数据偏差大小的估计。

v085保留旧 `primary`，另行报告逐蛋白 Fmax、exact micro AP、exact micro PR-AUC和 exact micro-Fmax，所有 E/M/B/G 用相同实现。逐蛋白 Fmax仍是显式0.001阈值网格近似，同时输出0.01网格结果；不宣称执行了完整CAFA协议或额外GO传播。AP与梯形PR-AUC分别命名。

必须先重算已保存矩阵，不能据本报告推算新的真实逐蛋白Fmax或PR-AUC。Expert、Modelout和Backbone同样需要重算。更换指标不等于模型性能提高。

## 2. 原有结果仍支持什么

| 方法 | Step | 历史micro-Fmax | 历史网格AUPRC | exact micro AP |
|---|---:|---:|---:|---:|
| B | — |55.632|32.009|49.451|
| E | — |69.781|56.496|67.555|
| M | — |69.772|59.212|68.791|
| Fixed |600|56.794|30.262|51.924|
| Fixed |2000|56.551|30.379|51.611|
| Dynamic |600|56.909|30.234|52.069|
| Dynamic |2000|56.502|30.329|51.533|

主表AUPRC略回升，而exact AP及Top-k从600到2000下降，因此退化并非纯粹的指标近似问题。另一方面，以历史主表AUPRC直接断言所有图排序都变差，也不成立。G相对B的exact AP仍有约2个百分点收益，离E/M仍有约16–17个百分点。

## 3. 图消融提供的直接证据

下表均为 step2000 的 full 减去关闭某通道后的结果，单位百分点。这是同一个已训练模型的推理干预，不是去掉通道后重新训练的因果比较。

| 通道 | Fixed Δ历史Fmax | Fixed Δexact AP | Dynamic Δ历史Fmax | Dynamic Δexact AP |
|---|---:|---:|---:|---:|
| weak证据（full−weak_off）|−0.136|−0.434|−0.139|−0.328|
| core证据（full−core_off）|+0.763|+1.921|+0.691|+1.935|
| 蛋白关系传播（full−pp_off）|+0.325|+0.527|+0.296|+0.375|
| 全部图输入（full−graph_off）|+0.857|+1.846|+0.900|+1.784|
| 正确GO对应（full−go_shuffle）|+0.441|+0.477|+0.300|+0.292|

图并非完全无效；core证据和蛋白传播已有条件性收益。当前weak证据融合存在干扰，但weak_off同时关闭种子candidate、邻居candidate和邻居二值pseudo，不能据此否定weak–GO路径，更不能将weak监督整体删除。新版先修正训练日程和第一跳条件对齐，不叠加新的decoder或loss。

## 4. 当前holdout不能决定最终图路线

| 方法/Step | holdout micro-Fmax | holdout histogram AP |
|---|---:|---:|
| B/0 |98.0975|99.8291|
| Fixed/600 |97.6672|99.6767|
| Fixed/2000 |97.5142|99.6246|
| Dynamic/600 |97.6378|99.6767|
| Dynamic/2000 |97.5122|99.6197|

Stage1见过这些core训练蛋白，其B已接近饱和。更有辨识力的是：holdout中core证据的Fmax贡献为负（fixed约−0.358、dynamic约−0.333），独立测试中却为正（+0.763/+0.691）。因此不能拿该holdout否定图迁移路线，也不能将其best checkpoint当作超过Expert的证据。

可信开发集必须与Stage1参数训练、Stage2全部图蛋白和最终独立测试分离，标签仅用于评估。v085提供显式manifest接口，不从现有test中自动切分或把已见过的core改名为开发集。仅ID隔离不保证同源性或时间隔离，科学实验仍需按研究协议划分。

## 5. 训练与采样判断

v084前25步warmup后恒定3e−4；旧v071使用OneCycle。v084全局weak/core batch为128/32，2000步仅覆盖约0.523轮weak；一个weak遍历需3823步。不能用step数或耗时直接推断过拟合或模型容量。

Dynamic约28%的选中邻居位于同池fixed前缀之外，已确实生效，但没有稳定提升且训练耗时高约15%。这不是跨epoch新增覆盖率。

训练种子仅25%使用cosine-only第一跳；新蛋白推理第一跳全部cosine。`native_dropout=1`仅让监督种子的第一跳匹配推理，保留邻居背后的原生关系以及weak–GO。现有holdout第一跳也是cosine，不应误称其仍使用原生种子关系。

loss角色归一化使weak/core权重为50/50，并非按64:16分成80/20。难gold记录均值1.289是每rank的16个core蛋白的总数，约0.081个/蛋白；原难gold覆盖率是批次比例平均，不能当全局覆盖率。新版记录累计分子/分母和core遍历量。

## 6. 有限实验与迁移判据

先完成现有预测的指标审计；再以同一开发集、同一监督和数据顺序比较constant、schedule、lowlr、aligned，dynamic只作与aligned匹配的后续对照。日程horizon固定4000，先运行到2000作为停止预算，再按开发集决定是否完成既定4000步。不要把horizon随续训总步数重设。

进入原完整图模型的有限迁移实验，至少需要：逐蛋白Fmax与exact PR-AUC均优于B，重复种子方向一致；图相对训练对照有额外收益；在完整日程下收益没有消失。推理关闭实验只提供依赖证据，不能替代重训去图对照。满足后迁移成功的采样/监督机制，并以相同预算替换图骨干，不能直接恢复旧训练的监督丢失问题。

最终目标仍是图预测G超过Expert E，同时报告Modelout M。没有承诺学习率调整能填平当前差距。若日程和归纳条件对齐仍无改善，下一步优先处理可信难gold来源及图支持位置的PU冲突，再考虑预聚合或ego-net Transformer路线。
