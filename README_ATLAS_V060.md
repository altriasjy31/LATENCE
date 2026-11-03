# LATENCE Functional Observability Atlas evaluation v0.6.0

## 1. 本次交付与评价主线

本包按照已确认的四组数据实现评价，保留包含关系与来源重叠，将其视为 Atlas 的证据扩张结构，不拆成互斥测试集。

| evidence_layer | metadata split | 角色 |
|---|---|---|
| future_union | dark_weak_test_direct | 严格未来新增功能总验证 |
| future_exp | dark_weak_test_exp_direct | 未来实验来源支持 |
| future_ipr | dark_weak_test_ipr2go_direct | 未来 InterPro2GO 计算来源支持 |
| history | dark_weak_test_2022_ipr2go_direct | 2022 历史计算证据底图 |

每组都支持 BP、MF、CC。固定 GO 输出词表中的 canonical 蛋白–GO 直接关系是主评价对象。历史与未来的证据标签、模型评分与模型是否命中分别记录；不会把历史计算注释计作新的功能发现。

程序仅读取已有 metadata、冻结概率及明确的蛋白/GO ID。它不运行模型、不训练、不融合概率、不改变已有文件。输出目录必须是新目录。

## 2. 核心计算

令 H 为历史 canonical 直接关系，E/I 为未来实验/IPR2GO 关系，F 为 future_union。首先验证 F=E∪I，并保留所有关系的四种布尔来源标记。默认检查 F∩H 为空；若不为空会失败，避免把历史已有关系算为严格新增。`--allow-history-overlap` 仅用于明确的诊断运行，报告仍会记录重叠。

输出两层扩张：

1. **观测证据扩张**：H、F、E、I、H∪F、F\H 的关系/蛋白/GO数量及来源交集；不要求模型命中，不代表这些关系全被模型发现。
2. **模型恢复扩张**：对每个模型、来源、频次组，评价完整 canonical GO 词表中的排名；再统计 H∪模型已恢复的F，以及 NBS 相对 Stage-1 的 gained/lost/shared。

排序先在完整 canonical 非根 GO 词表进行，再按正例所属集合和历史频次分层。绝不先过滤成已知正确功能的子词表。别名预测分数取原始输出列均值，直接标签按 canonical 关系去重；根 GO 不计入直接功能恢复。

程序同时报告两种口径：

- `stable`：与此前结果连续，按原始 canonical 列序打破同分；仅作为点估计与历史对照。
- **`conservative`：score>0 且同分块最差排名≤K。** 用作主文的稳健命中口径，避免 Expert 零分大并列造成伪命中。

提供 micro/macro Recall@K、MRR、排名上下界、正例零分/同分比例、命中关系数、受益蛋白与 GO 数。K 默认10/50/100/512/1024。

NBS 的新增恢复以三个 Stage-1 模型各自 Top-K 命中集合的并集为参照：

`target_hits − (backbone_hits ∪ expert_hits ∪ modelout_hits)`。

这个集合参照比单一 Stage-1 更强，最多可涉及3×K个候选，不是同预算部署集成，也不是概率融合。每个单模型仍有独立成绩。若 NBS 输入存在，默认要求三种 Stage-1 对照都存在；只有 Stage-1 时可以先运行，增量表为空并保留正确表头。

bootstrap 使用同一来源内相同正例对的配对差异，按可用的序列 SHA256 聚类；无哈希时回退蛋白 ID。置信区间与同分位置上下界分别报告。四个重叠来源不被当作独立重复实验。主文可优先看 future_union，再读来源分层；不能重复累计 Union 和两个来源的正例。

## 3. 安装与本地验证

在现有 LATENCE 项目根目录执行：

```bash
unzip -n LATENCE_atlas_eval_v0.6.0.zip -d .
python -m unittest discover -s tests -p '*v060.py' -v
```

本包附带未修改的 v0.5.3/v0.5.4 输入与指标工具；仍需要现有项目的 `scripts/eval_dark_weak_predictions.py`、`utils/util_functions.py`、`scripts/dark_weak/native_v085_metrics.py`、Stage-1 diagnostics 源码和 v085 native metric 源码。

本地16项合成测试全部通过，覆盖三namespace端到端、来源重叠/包含、别名/根去重、完整词表排名、零分/同分、增量恢复、跨namespace独立蛋白计数、概率ID覆盖、哈希及原文件保护。未来分类指标的非Torch路径已测试。原始Torch helper未在当前无PyTorch环境逐值运行；v0.5.4保留了可在训练环境执行的原函数对照测试。本次没有用户服务器的真实概率，未重算真实结果。

## 4. 第一步：只核验四组数据规模

不需要任何预测文件：

```bash
python scripts/dark_weak/eval_atlas_v060.py \
  --metadata data/unidata_with_dark_weak_eval.pkl \
  --unified-root data/dark_weak_test/unified \
  --inventory-only \
  --out outputs/dark_weak_eval/atlas_v060_inventory
```

对照 `configs/selected_dataset_counts.json` 中用户提供的12组统计，查看输出 `dataset_inventory.tsv`。该表同时列原始direct/ancestor计数及canonical非根关系数；后者应允许因别名和根过滤发生变化。输入metadata必须遵守v0.3.1字段语义：annotations=direct，prop_annotations=完整祖先闭包。

`atlas_manifest.json` 的 `unique_counts` 按namespace合并蛋白ID，避免将 BP/MF/CC 的蛋白数直接相加。不同GO分支的canonical关系可以合计，仍需保留namespace来源。提供预测config时，inventory-only还会列出预测覆盖及缺失人数，但不运行模型/排序。

## 5. 第二步：覆盖四组数据的无标签推理目标

旧future预测只覆盖176个BP等小集合时，不能直接扩大评分范围。先使用现有工具选择四组蛋白并集，生成新的推理metadata和ID目录：

```bash
python scripts/dark_weak/prepare_inference_split.py \
  --metadata data/unidata_with_dark_weak_eval.pkl \
  --unified-root data/dark_weak_test/unified \
  --source-splits \
    dark_weak_test_direct \
    dark_weak_test_exp_direct \
    dark_weak_test_ipr2go_direct \
    dark_weak_test_2022_ipr2go_direct \
  --split atlas_inference_v060 \
  --output data/unidata_with_atlas_inference_v060.pkl \
  --id-dir data/dark_weak_test/atlas_inference_ids_v060
```

这里故意列出四个来源，前3组的包含关系不会使ID重复；不使用存在旧schema差异的 `--full-cohort` 开关。新split标签为空，`dark_weak_test*`带标签split从推理pickle中删除；原评价metadata保持独立。

然后沿用已经成功运行的 Stage-1 导出和 native v094 导出流程，将推理metadata、mode、目标ID和MSA索引切换到这个新版本。每个namespace各自绑定对应的checkpoint、历史MSA与Expert概率，不能将BP导出文件用于MF/CC。MSA索引可由现有 `prepare_msa_index_view.py` 引用历史shards，无需复制MSA；缺失不能静默丢弃。

本包不会自动重跑GPU推理。已有概率只要ID覆盖四组并集且GO顺序一致，就可直接复用；目录名字含full并不能保证覆盖人数。historical全组包含未来未匹配/序列修订者，其预测应使用历史2022序列；它是历史底图，不是与未来逐蛋白配对的盲测。

## 6. 第三步：填写预测文件配置

复制 `configs/atlas_predictions.example.json`，将所有 `REPLACE_WITH...` 占位项替换为实际导出路径及身份。结构如下：

```json
{
  "bp": {
    "backbone": {
      "probability": "/actual/bp/backbone_base.prob.f32.npy",
      "protein_ids": "/actual/bp/protein_ids.txt",
      "go_ids": "/actual/bp/go_ids.txt",
      "branch": "backbone_base"
    }
  }
}
```

对每个需要评价的namespace填写backbone/expert/modelout/nbs，各模型可以使用不同ID文件和行顺序，程序按ID对齐。只评Stage-1时删除nbs项。`branch`只是显式来源标记，不能自动证明分支身份；实际来源需要导出manifest确认。可额外给各项提供 probability_sha256/protein_ids_sha256/go_ids_sha256，程序会核验。所有输入的实际哈希都会保存。

config中的相对路径以config文件所在目录解析。NBS `native_adapter` 可作来源记录，但本程序不替代原v094 adapter的checkpoint/输入/时间来源验证；应先通过既有native导出审计再填入冻结概率。

也可以用多个 `--complete-manifest bp=... --complete-manifest mf=...` 读取已有complete manifest中的预测源；它不把旧manifest的future cases当作本次完整覆盖，仍然严格检查四组所需ID。该方式与prediction-config互斥。

## 7. 正式评价

先用较少bootstrap确认数据/覆盖，再正式使用500次。历史分组上的CPU bootstrap和每行全GO排序会比原小future集合慢。

```bash
python scripts/dark_weak/eval_atlas_v060.py \
  --metadata data/unidata_with_dark_weak_eval.pkl \
  --unified-root data/dark_weak_test/unified \
  --prediction-config configs/atlas_predictions.json \
  --train-metadata data/unidata_with_exp_train_pseudo.pkl \
  --train-label-field prop_annotations \
  --namespaces bp mf cc \
  --ks 10 50 100 512 1024 \
  --bootstrap 500 \
  --classification future \
  --out outputs/dark_weak_eval/atlas_v060_frozen_checkpoint
```

只有BP已导出时用 `--namespaces bp`；不要填入虚构MF/CC输入。`--train-metadata`可选：提供时按历史canonical训练蛋白频次输出all/rare/rare≤5/medium/common/zero_train等分层；没有时只输出all。频次不使用未来标签。

`--classification future`为默认：前三个未来集合附带完整Stage-1 histogram、原始helper和native v085指标。原始helper需要PyTorch；无Torch环境显式加 `--skip-stage1-official`。只做证据排名可用 `--classification none`。`--classification all`还计算历史大组的分类指标，适合补充材料；其精确AP会排序大量protein×GO对，超过默认80,000,000 cells时会拒绝。确认RAM后才用 `--allow-large-metrics`。正例排名本身逐蛋白计算，不受这个分类矩阵限额影响。

分类指标使用原始输出列和prop_annotations，对未知位置按代理负例处理；它们不参与独立发现计数。native/Stage-1字段及阈值约定延续v0.5.4。历史GO层次与预测均不在这里重新传播。

## 8. 输出及回传

| 输出 | 用途 |
|---|---|
| dataset_inventory.tsv | 12组人数、原始direct/ancestor数、canonical非根数 |
| prediction_coverage.tsv | 各模型实际ID覆盖，发现是否仍为旧小集合 |
| evidence_expansion.tsv | 历史→未来观测证据扩张、来源交集、无历史直接证据的未来蛋白数 |
| atlas_observed_relations.tsv | 每条canonical关系的四组来源标记，支持未来Atlas导入 |
| atlas_pair_ranks.tsv | 每个模型对每条已观测关系的分数/排名/同分界限/命中/频次 |
| atlas_metrics.tsv | 4组×3namespace×模型×频次组的直接功能排名成绩 |
| stage2_incremental_recovery.tsv | NBS相对Stage-1命中并集的gained/lost/shared及受益蛋白/GO数 |
| stage2_incremental_pairs.tsv | 新增/丢失关系的具体身份、模型排名与全部证据来源 |
| paired_rank_bootstrap.tsv | 配对排名差异的序列聚类置信区间 |
| classification_metrics.tsv | 可选完整Stage-1/native分类指标 |
| atlas_manifest.json | 输入/源码协议、canonical与集合审计、真实独立蛋白数、参数来源 |

首先回传 `dataset_inventory.tsv`、`evidence_expansion.tsv`、`atlas_metrics.tsv`、`stage2_incremental_recovery.tsv`、`atlas_manifest.json`；筛选案例时再读具体关系与排名表。

`atlas_observed_relations.tsv`是已观测证据的导出表，并不包含历史母集上全部待验证预测。完整Atlas后续还应在不依赖2025是否获得注释的历史蛋白母集上导出预测候选，保留“待验证”状态；不能将本评价表当成整个蛋白宇宙或真实负例集合。

## 9. unseen GO下一步

详见 `docs/UNSEEN_GO_EVALUATION_PLAN.md`。当前交付完成的是已有GO下的蛋白–功能关联证据扩张。现有v094共享scorer、canonical holdout mask和requested-GO导出接口为Stage-2 unseen评价提供了基础，但本包没有宣称已完成新的unseen训练实验。

优先顺序为：真实seen mask/暴露审计→受控Stage-2 GO holdout→显式GO development/test划分→真正时间新增GO的身份核验与锚定几何生成。自然零core正例、主动heldout、旧ontology内词表外GO、历史ontology不存在的新节点应分别记录。不能将Stage-1已见过的目标称为全流程zero-shot。
