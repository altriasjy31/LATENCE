# LATENCE unseen GO 评估：下一阶段实施方案

本文是经现有 v094 源码核对的设计方案，不代表已经完成 unseen GO 数据集、重新训练或新 GO 几何生成。v0.6.0 的主要交付仍是当前四组数据上的 Atlas 证据扩张评价。

## 1. 先区分三种扩张对象

| 对象 | 准确含义 | 当前四组数据是否能直接证明 |
|---|---|---|
| 新的蛋白–GO 关联 | GO 本身在历史词表中，但该蛋白的这项功能未被历史证据支持；未来新增注释支持其关联 | 可以做回溯验证；这是当前 Atlas 的关系扩张 |
| Stage-2 未监督 GO | 某 canonical GO 不参与 Stage-2 的正例、负例、伪标签、软教师及蛋白–GO 消息监督，但可能已被 Stage-1 见过 | 需要读取真实训练 mask 并审计；仅凭 core frequency 为零不成立 |
| 真正新增的本体 GO | canonical GO 实体在历史本体快照中不存在，随后才出现；不是旧 GO 的 alt_id、迁移名称或替换编号 | 当前冻结输出词表不能直接评价，需要新 GO registry、标签及几何资源 |

建议先完成 Stage-2 未监督 GO 的受控验证，再进入时间新增本体节点。端到端“全流程未见 GO”应单独作为更严格协议，不与 Stage-2 未监督混称。

## 2. 当前代码已经具备什么

### 2.1 v094 已有共享 seen / unseen 评分器

`nbs_models/nbs_protein_go/nbs_pg/full_task_model_v094.py` 的 `FullTaskGraphModelV094` 使用 `output_strategy=shared_direct`。seen 与 unseen ontology row 共用同一个蛋白–GO scorer，没有按 GO 单独训练的输出分类行。Backbone / Modelout 在训练中提供监督，但不作为这个 readout 的概率或 logit 跳接输入。

这使“评分一个训练时未监督的 ontology row”在接口上可行，但评分能力本身不等于已获得 zero-shot 泛化证据。

### 2.2 已有任意 registry GO 的导出入口

`scripts/nbs/predict_nbs_go_v094.py` 支持：

- `--go-ids-file`：明确请求的 GO 身份及顺序。
- 基于当前 registry 的 `resolve_targets`：输出 canonical 身份、ontology row、原 task columns 和 Stage-2 seen 状态。
- `--ontology-file`：可选的、已经生成好的 anchored ontology bundle。
- 通过 `target_ontology_ids` 调用共享 scorer，输出请求 GO 的概率矩阵。
- 原始源码明确输出 `stage1_exposure_audited=false`、`strict_pipeline_zero_shot_claimed=false`、`new_GO_geometry_created=false`。

因此，历史本体里存在、但不在闭集 task 输出列中的 GO，可能已经可以通过 requested-GO 路径评分；在任何旧本体资源里都不存在的新 GO，则还缺少新节点资源。

### 2.3 已有 Stage-2 标签 holdout 骨架

`scripts/nbs/run_nbs_v094.py`、`scripts/nbs/calibrate_nbs_backbone_v094.py` 和 `scripts/nbs/build_nbs_dataset_v094.py` 已有 `--label-holdout-fraction` 与 `--seed` 路径。

核心逻辑在：

- `nbs_pg/unified_weak_dataset_v093.py::make_seen_mask`：按 canonical ontology row 整组划分，别名列不能一部分 seen、一部分 unseen；训练 core 中自然无正例的 canonical 组也被排除。
- `nbs_pg/weak_dataset_v094.py::build_dataset`：保存 `seen_task_mask`、`heldout_ontology_rows`，构建受 mask 限制的 weak hard targets 和候选缓存；校准文件必须绑定相同 mask。
- `nbs_pg/full_task_data_v094.py`：过滤 gold / pseudo / anchor 蛋白–GO 边、候选及 hard / soft targets。
- `nbs_pg/full_task_loss_v094.py`：hard BCE 和 Modelout KL 都只作用于 seen 列；unseen 列不能被当作负例训练。

不过，其 manifest 明确说明 inherited Stage-1 features / teacher / proposals 可能见过 heldout GO。现有逻辑提供的是 Stage-2 监督隔离骨架，尚不自动证明端到端 zero-shot。

## 3. 第一优先级：Stage-2 GO-label holdout

这是最小、最接近现有 v094 的可实施实验。

### 3.1 冻结划分

1. 以历史本体 canonical GO 为单位选择 heldout GO；整组冻结 alt_id 对应的原始列。
2. 选择规则只依赖训练频次与历史本体，不根据未来测试效果选择 GO。
3. 输出 GO train / development / test 划分。development GO 与 test GO 应互斥，避免根据 test GO 选阈值或 checkpoint。
4. 保持蛋白划分与 GO 划分两个轴独立记录：GO 未监督不等于蛋白未参与图训练。

现有随机 holdout 参数可以作为 Stage-2 隔离的首轮检查，但它没有内建 GO development/test 双划分及本体蕴含强度控制。正式实验需要显式 GO 列表，而不是只记录一个随机比例。

### 3.2 对所有蛋白消除 heldout GO 的监督路径

不仅删除 core 的直接正例，还应确认：

- heldout canonical 列及全部别名从 core gold、传播标签、weak hard targets、pseudo CSR 中排除。
- heldout 列不参与 BCE 负例，也不参与 Modelout soft KL。
- 蛋白–GO gold / pseudo / anchor / candidate 边和反向或倒排索引不能保留这些目标关系。
- 重新生成受 mask 约束的候选和校准缓存，不能复用 all-seen 缓存后仅改一个标签文件。
- threshold / early stopping / checkpoint selection 只使用 seen 或独立 development GO。
- 按该隔离协议重新训练 Stage-2；不能把已被全部 GO 监督过的 Stage-2 checkpoint 继续微调后称为“从未监督”。

Stage-1 可以保持冻结以隔离第二阶段能力，但实验名称应明确为 **Stage-2-unseen GO with frozen pretrained Stage-1**。如果要进一步声称全流程未见，需重建 Stage-1、教师和所有导出资源，使目标 GO 没有任何监督暴露。

### 3.3 本体先验与祖先泄漏要分别处理

GO 的名称、定义、层级关系及纯 GO 公理可作为允许的语义先验；应记录使用哪个快照。它们不应混入蛋白功能标签。

若 heldout GO 是某个 seen descendant 的祖先，seen descendant 的正例会通过本体直接蕴含 heldout 正例。这属于结构化推断线索，不能忽略。建议：

- 先报告一般 canonical-label holdout，明确允许历史 GO 本体。
- 再用历史本体中的叶节点，或对一整片子树同时隔离，建立较少“已知 descendant 直接蕴含”的确认子集。
- 原始 heldout gold 在评价文件中保留；训练传播或消息中不能因后续传播再次补回 heldout 列。

这一步是划清“没有直接训练该 GO 输出”与“训练证据已能机械推导目标 GO”两种情形，不需要额外引入 rare 配额或特殊候选预算。

## 4. 第二优先级：历史本体没有的新 GO

### 4.1 用两个本体快照建立身份清单

不能只用字符串集合差。应为每个未来 GO 记录：

- 两个快照的 canonical 身份、namespace、active / obsolete 状态。
- alt_id、replaced_by、consider、合并或迁移关系。
- 它是否只是旧实体的身份调整，是否真的在历史 registry / geometry 中不存在。
- 允许的未来 GO 定义与拓扑资料版本，以及未来蛋白注释首次观测窗口。

主 unseen-new-term 集合只纳入真正新增的 canonical 节点；身份迁移另作描述，不计为新的功能类别。所有当前四组历史词表内的蛋白–GO 新关联继续作为独立的关系扩张评价。

### 4.2 现有新节点接口还缺什么

`FullTaskGraphModelV094.replace_ontology_for_inference` 复用 v093 的 anchored replacement 校验。它要求：

- 新 bundle 显式提供 `center`、`offset`、`edges`、`task_to_ontology`。
- `old_ontology_rows` 将每个旧 ontology row 注入映射到新资源；所有旧 center 和 log-offset 坐标必须精确保持。
- 旧 GO 身份、别名与 task mapping 保持一致。
- 关系类型等于训练时已有类型；不能临时加入未经学习的新关系类型。
- registry records 明确绑定新旧身份，并重算 ontology 编码缓存。

**当前代码不会自动生成新 GO 的 BoxSquaredEL geometry。** 下一项实现工作应是“新 GO 几何/registry 资源生成器与锚定验证”，再接 requested-GO exporter。不能用随机新向量或复制父节点几何冒充已完成的新 GO 泛化实验。

若采用未来 GO 定义与结构为新节点生成几何，应声明为“允许未来本体语义、禁止未来蛋白标签”的 ontology-informed 扩展。应将几何拟合限制在本体公理，禁止使用未来蛋白–GO 注释训练新节点。若重新计算了旧节点几何，则不满足现有 exact-anchor 接口，需要另行设计并明确模型变更范围。

新增节点及其关系还可能改变旧节点的图编码，原源码不保证旧预测不变。因此需要同一冻结模型在扩展前后的 seen-GO 简要保持性检查。

## 5. 评价空间必须同时有两种

| 评价空间 | 回答的问题 |
|---|---|
| unseen-only GO | 模型在未知功能集合内部能否区分正确目标 |
| seen + unseen 共同 GO 词表 | 未知功能能否与已有功能竞争，进入实际 Atlas 候选范围 |

共同词表成绩应是主结果之一，不能在先过滤为真实 unknown GO 后将较高 Recall@K 称为全词表恢复。

建议输出：

- canonical 直接正例的 micro / macro Recall@K、MRR、同分保守界限。
- 有正例可评的 GO 数、蛋白数、蛋白–GO 对数；按 GO 宏平均作为补充，防止少数 GO 决定总分。
- NBS 相对可比较基线新增恢复的目标关系及受益蛋白数。
- 使用正例与未知位置计算的 AP / Fmax 作为代理补充，阈值来源和未知负例约定写入 manifest。

新 term 不在 Stage-1 固定输出词表中时，不能给 Backbone / Expert / Modelout 添加人为零列，再以此得到所谓完整性能优势。应标记“该基线不支持该目标输出”；只有具有明确跨 GO 语义 readout 的基线才能参与相同候选空间的数值对比。

GO 祖先传播后的标签用于层级一致性或辅助评价。unseen GO 恢复主计数使用直接目标，避免旧祖先的正确预测被计为新节点恢复。

## 6. 下一步需要的最小文件集合

| 文件 | 内容 |
|---|---|
| `go_split_manifest.json` | canonical train / development / test GO，ontology 行与别名列，选择规则和 seed |
| `go_exposure_audit.tsv` | 每个目标在 core gold、propagated gold、weak hard、soft teacher、candidate、protein–GO 边、Stage-1/teacher 来源中的暴露状态 |
| `ontology_identity_transition.tsv` | 历史与未来 GO 的 canonical 身份、别名/替换/新增分类 |
| `unseen_direct_gold.tsv` | 蛋白、直接 GO、namespace、证据类型与时间，独立于模型输入 |
| `unseen_protein_ids.txt` / `candidate_go_ids.txt` | 明确、有序的蛋白轴与共同候选 GO 轴 |
| `unseen_geometry_bundle.pt` | 仅真正新节点协议需要：anchored boxes、关系、registry 和 old row mapping |
| 新训练与导出 manifest | checkpoint、数据、mask、候选缓存、特征/教师版本与输入输出哈希 |

`go_exposure_audit.tsv` 中“未知”必须保留为未知，不能默认写作未暴露；`core_count=0` 也不能替代其他列的审计。

## 7. 建议实施顺序

1. 完成当前四组、三 namespace 的 Atlas 关系扩张评价，并记录真实训练 seen mask。
2. 读取现有 v094 dataset manifest，将自然零 core 正例、主动 heldout、词表外但 ontology 内、历史 ontology 外四类 GO 分开统计，确认各类实际有多少可评直接正例。
3. 通过现有 mask / requested-GO 路径完成一个明确标记为 Stage-2-unseen 的受控训练与评价；先确保监督隔离和候选轴正确。
4. 实现显式 GO development/test 划分及 exposure audit，再升级正式 heldout 协议。
5. 生成时间新增 GO 身份清单和 anchored 几何，最后开展真正新增 ontology node 的推断。

Atlas 中可将第 1 步表示为“已有 GO 下的关联证据扩张”，将第 3–5 步分别标明监督未见与节点新增范围；两条扩张轴可以共同展示，但必须保留各自的证据来源与评价定义。
