# NBS v0.8.8：异质图 tunedGraphSAGE 完整分类训练

**性能状态：未验证。** 本包提供可安装实现，没有新的完整 BP 训练权重，也没有新测试集 Fmax。当前环境只有 CPU 与小型测试数据，无法代替用户服务器上的真实训练。v087 direct epoch5 的 Fmax 为57.732654，较 B=57.723149 仅+0.009505，**未达标**。

本轮验收同时要求 Fmax≥60、相对基线提升≥3个百分点，因此有效门槛为 **60.72314936447243**。固定使用已有 `methods.NBS_final.standardized.standard_protein_fmax`。旧报告的 `primary.fmax` 是另一项指标，不能混用。

## 本次选择的依据

新诊断中，direct 的蛋白内 AP 从58.316降至56.613；同GO跨蛋白AP从37.368降至36.271，低于B的38.706。该证据说明判别质量存在退化，不能将全部问题归为共同分数漂移。旧实现另有确定的低分负例梯度截断、窄共享解码器和训练/外部查询构图不一致。因此选择一次替换编码器、完整分类头、监督目标及查询接入方式。现有证据尚不能证明它们是唯一瓶颈，也不能预先保证+3。

## 唯一主配置

配置：`nbs_models/nbs_protein_go/configs/bp_full_task_v0.8.8_hetero_tuned.json`。

- **独立绝对分类模型**：最终 `Linear(320, 21312)` 输出每个原始GO列。移除旧GO-query路由解码器、残差B输出、逐对candidate特征和vote读出。输出不融合B、expert或modelout。
- **异质图 tunedGraphSAGE**：3个真正逐层采样的blocks；按关系进行加权均值后求和；每节点只加一次self变换。第2、3层加上一层BN前的raw residual，再执行按节点类型的BN→ReLU→dropout。hidden=320，input dropout=.1，hidden dropout=.25。
- **GO证据的位置**：共享Box/ontology编码器生成GO状态；候选、core gold与weak二元成员关系分别聚合后注入初始蛋白状态。三层blocks处理带类型的蛋白关系；没有声称GO节点也在这些blocks中双向动态更新。稀疏B候选证据仍在输入中，因此本版不宣称脱离B生成的数据。
- **完整轴BCE**：所有GO位置共用分母21312，取消旧ASL截断、难例挖掘、ranking和anchor目标。core、weak的全局角色均值按1:.25归一；只在训练中排除已知正例的未标注等价GO列，评测保留全部列。未标注项作为二元分类的负向监督近似，不能解释为已证实的生物学负例。弱监督仍仅用已有`modelout>0.5`二元成员关系，未加入概率蒸馏或输出融合。
- **训练先验初始化**：只从实际training core和weak CSR计算与角色权重及alias有效样本一致的输出偏置；不读取holdout或测试集标签频率。
- **统一外部查询视图**：训练和预测均建立无标签的虚拟查询节点，只用cosine接入支持图。整个DDP宏批次中的原蛋白ID从所有层支持图中排除，避免多跳回流。

这是 tunedGNN 的异质图任务移植，不是官方ogbn-proteins配置的逐项复现。保留其SAGE、自身变换、raw residual、post-BN/ReLU/dropout、完整分类头和逐层blocks；深度、关系采样、标签数、监督权重、学习率与调度按本任务明确配置。不能把v087旧模型的失败当作官方方案已经失败的证据。

固定参考源码：[models.py](https://github.com/LUOyk1999/tunedGNN/blob/23f9604e8b13a9a6d3faa2f691cd844006979153/large_graph/models.py)、[protein.py](https://github.com/LUOyk1999/tunedGNN/blob/23f9604e8b13a9a6d3faa2f691cd844006979153/large_graph/protein.py)。本版用原生PyTorch实现SAGE均值算子，无新增DGL/PyG依赖。

**后续限制**：标签缺失、伪标签噪声和冻结的序列特征仍可能限制Fmax。完整线性头优先解决当前固定GO任务；它不具备新GO的zero-shot输出能力，本版不把该能力当作已完成。

## 安装

将压缩包解压至现有项目外的目录，安装到已经运行v087的同一项目。安装器先检查依赖和包内SHA256；已有不同内容的新文件会报错，不覆盖。v080–v087代码、配置与实验目录保持不变。

```bash
unzip latence_nbs_v088.zip
python latence_nbs_v088/install_v088.py --project-root /path/to/latence-project --check-only
python latence_nbs_v088/install_v088.py --project-root /path/to/latence-project
cd /path/to/latence-project
```

只需现有v087训练数据、图候选池和相同外部BP评测输入。运行脚本会校验/复用原池；未准备时执行必要准备。不要求导出expert/modelout评测矩阵。

## 实验流程

默认两张GPU；若训练服务器的路径不同，仅设置已有输入与元数据路径：

```bash
export NUM_GPUS=2
export INPUT_DIR=outputs/latence_nbs_eval/bp_shared_full_task_top512_v071_v2
export METADATA_FILE=/home/dataset-assist-0/datafile/latence-dataset/unidata_with_exp_train_pseudo.pkl
```

**1. 一次必要的运行检查。** 独立目录运行20个optimizer updates；检查非有限值、显存和真实分布式入口，不用于验收。它不会自动运行外部测试或暂停的消融。

```bash
bash scripts/nbs/run_nbs_v088.sh smoke
```

**2. 从头训练到epoch1。** 禁止接着v087权重训练；5-epoch调度的总长度在启动时固定，`STOP_EPOCH`只控制本次执行终点。

```bash
STOP_EPOCH=1 bash scripts/nbs/run_nbs_v088.sh train
CHECKPOINT=outputs/latence_nbs_experiments/v088/hetero_tuned/nbs_epoch1.pt \
  bash scripts/nbs/run_nbs_v088.sh evaluate
python scripts/nbs/check_nbs_v088_acceptance.py \
  --baseline-comparison BASELINE_CONTRACT_v088.json \
  --comparison outputs/latence_nbs_experiments/v088/hetero_tuned/eval_step3823/full/metrics/nbs_w2s_comparison.json \
  --output outputs/latence_nbs_experiments/v088/hetero_tuned/acceptance_epoch1.json
```

默认人口与2卡条件下epoch1=3823 updates。若数据人口或卡数不同，实际step从日志/`resolved_config.json`读取，修改上面的`eval_step...`路径；任务、评测蛋白和GO列不能更换。

验收脚本返回码：达标=0、未达标=1、未验证=2。它会先写JSON，非零退出码不是需要绕过的错误。评测只计算原蛋白Fmax及其必要计数，保留原0.001阈值网格、严格`>`比较、空gold处理和全部21,312列，不计算新的校准、融合、ontology后处理或任务子集。

**3. 固定终点为epoch5。** epoch1是中期测量，epoch5是预先指定的完整训练终点。若第一阶段运行正常，以下命令继续同一模型、优化器、随机状态和既定调度；不会生成其他配置或自动堆补丁。

```bash
RESUME=outputs/latence_nbs_experiments/v088/hetero_tuned/latest.pt \
  STOP_EPOCH=5 bash scripts/nbs/run_nbs_v088.sh train
CHECKPOINT=outputs/latence_nbs_experiments/v088/hetero_tuned/nbs_epoch5.pt \
  bash scripts/nbs/run_nbs_v088.sh evaluate
python scripts/nbs/check_nbs_v088_acceptance.py \
  --baseline-comparison BASELINE_CONTRACT_v088.json \
  --comparison outputs/latence_nbs_experiments/v088/hetero_tuned/eval_step19115/full/metrics/nbs_w2s_comparison.json \
  --output outputs/latence_nbs_experiments/v088/hetero_tuned/acceptance_epoch5.json
```

不从反复查看测试结果中挑选“最佳epoch”冒充预定终点成绩。现有1800蛋白集已经被多轮观察，仍按相同条件做当前工程验收；这不增加新的独立泛化证据。若已有合格外部开发集，可从新训练开始绑定`DEV_MANIFEST`，只按其Fmax选checkpoint；core holdout仅是训练监控。

**4. 判断。** epoch5的`new_Fmax`、`delta_Fmax`与`status`为本轮结论。若低于60.723149，明确“未达标”；下一步首先检查此次“完整轴分类训练与查询视图能改善判别”的假设，不自动加深同一结构或堆同类正则。回传epoch1/5的`nbs_w2s_comparison.json`、两个`acceptance_*.json`及`training_history.json`即可，不要求三项已暂停消融。

## 固定评测与暂停范围

验收锁定原BP的1800个蛋白、21312个GO列、63688个已标注正例、原始输入/蛋白ID/GO注册表/元数据/B矩阵哈希、同一Fmax实现哈希及阈值规则。任何身份不一致会拒绝给出达标结论。`BASELINE_CONTRACT_v088.json`为原epoch1 comparison的未修改副本，文件SHA也已固定。

`core`、`query candidate`、`pp`三项消融不运行；脚本仅支持`full`，没有effects或series入口。本模型正常前向中的图关系不属于额外消融，测量完整模型Fmax也不依赖那些消融。

## 资源与验证边界

- 每卡宏批仍为64 weak＋16 core；分为4个16＋4微批，DDP全局角色计数作为唯一分母，不再额外除4。默认2卡每epoch覆盖489222 weak一次，3823 updates；5epoch为19115 updates。core按独立无放回序列循环。
- 从外到内fanout按关系为`[1,1,1,1,1]`、`[2,2,2,2,2]`、`[0,0,0,0,32]`，关系顺序为ppi/similar_to/weak_to_core/core_to_weak/cosine。每个20查询微批外层节点未去重上界43560；超预算会报错，不悄悄裁掉图。
- 新分类头含6,841,152个参数，FP32权重约27.4MB；含梯度及Adam两组状态约109.5MB/卡，**这不是整模型峰值显存**。图更深更宽，但移除了旧逐GO路由解码器；净训练时间、总显存和推理延迟目前未验证，不能预先声称更快。
- 每个update包含4次前后向，GO编码也随微批重算。`training_history.json`记录seconds_per_step和rank0峰值allocated显存；稳定区间平均`t`秒/update时，单epoch训练约`3823*t/3600`小时，另加准备、验证与导出时间。初次缓存准备不能计作稳定推理延迟。
- 默认AdamW lr=.003、weight decay=.0001；warmup=.1epoch，hold至3epoch，余下至5epoch余弦降至.0003。它们是此单一重构配方的预设值，没有声称已由性能搜索验证。epoch完整遍历保留；不把官方1000个小epoch机械等同本任务1000轮全量weak遍历。
- BN使用卡内微批统计，每个update前同步rank0的running buffers，导出使用同一份running统计；没有将它称为全局SyncBN。
- CPU测试用于必要正确性，不能证明真实GPU/NCCL训练可用，也不能证明Fmax≥60.723。具体测试记录见`VALIDATION_v088.md`。
