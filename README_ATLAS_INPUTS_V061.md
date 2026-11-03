# Atlas v0.6.1：免手填配置与四组预测覆盖说明

## 1. 这次报错的含义

**无需再手写config。** `--complete-manifest`已正确读取三模型概率。当前问题是已有概率仅覆盖未来176个BP蛋白，四组并集需要6672个，缺6496个。

| 项目 | BP蛋白数 |
|---|---:|
| 未来严格新增并集 | 176 |
| 2022历史InterPro2GO | 6504 |
| 两组重叠蛋白 | 8 |
| 需要的蛋白并集 | 6672 |
| 现有预测覆盖 | 176 |
| 尚缺 | 6496 |

重叠8个指蛋白身份，不表示历史与未来GO关系重叠。inventory-only的Completed只表示统计/覆盖检查完成；它允许报告缺失，不代表预测完整。请读已有目录中的prediction_coverage.tsv。

更改config、改metadata名字、填零或重复行都不能补齐这些预测。程序继续保留完整覆盖检查。

## 2. 本包新增什么

- `eval_atlas_v061.py --stage1-export-dir bp=DIR`：从Stage-1原生prediction_manifest.json、protein_ids.txt、go_ids.txt及三个通道自动生成输入连接文件，无需手填模型配置。也继续接受原来的--complete-manifest。
- inventory-only后直接显示各模型的覆盖/缺失数。
- `align_atlas_expert_v061.py`：从覆盖更大范围的历史Expert概率按ID切片重排；默认float32，避免旧对齐工具再次量化到float16。
- 原eval_atlas_v060.py、指标、标签、模型和NBS训练代码均不修改。

新入口只是便利工具，不能生成缺少的模型分数。原生export目录必须真实存在，并通过任务、轴维度、ID和GO哈希、通道输出哈希检查。自动生成的连接文件还保留来源manifest路径及哈希。

```bash
unzip -n LATENCE_atlas_input_helper_v0.6.1.zip -d .
python -m unittest discover -s tests -p '*v061.py' -v
```

依赖已经安装的v0.6.0。新入口直接复用该版本的评分逻辑。

## 3. 先看能否复用更大范围的已有导出

如果已有全量weak/core Stage-1导出，并且同一checkpoint、Modelout分支、GO顺序及ID来源都能确认，可复用其中覆盖所需6672个蛋白的概率，不必再次GPU推理。

若更大的导出本来就是原Stage-1 exporter产生、包含prediction_manifest.json，可直接检查：

```bash
python scripts/dark_weak/eval_atlas_v061.py \
  --stage1-export-dir bp=/实际路径/完整Stage1导出目录 \
  --namespaces bp \
  --inventory-only \
  --out outputs/dark_weak_eval/atlas_v061_full_export_coverage
```

若现有更大范围文件采用其他manifest格式，保留原完整manifest/ID关系，不要伪造prediction_manifest.json；仍可通过原complete-manifest入口使用合规的已有文件。

## 4. 如果确实只有176人的预测：准备四组推理并集

在项目根目录执行已有工具：

```bash
python scripts/dark_weak/prepare_inference_split.py \
  --metadata data/unidata_with_dark_weak_eval.pkl \
  --unified-root data/dark_weak_test/unified \
  --source-splits \
    dark_weak_test_direct \
    dark_weak_test_exp_direct \
    dark_weak_test_ipr2go_direct \
    dark_weak_test_2022_ipr2go_direct \
  --split atlas_inference_v061 \
  --output data/unidata_with_atlas_inference_v061.pkl \
  --id-dir data/dark_weak_test/atlas_inference_ids_v061
```

输出BP目标ID应有6672行；MF/CC以实际输出为准。推理split标签为空，带标签的dark_weak_test* split从推理副本移除。评分仍读取原评价metadata。

## 5. MSA和Expert输入必须覆盖新目标

两个资源无法从旧complete manifest中自动恢复：

1. 覆盖新目标的原始历史MSA source index，不能只提供176人的小view。
2. 覆盖新目标的历史Expert全量概率及其权威行顺序，不能使用176人的已切片Expert输出。

沿用此前的源文件即可，不需要设计新的模型config。下面变量填写源文件实际路径。

### MSA view

```bash
python scripts/dark_weak/prepare_msa_index_view.py \
  --target-ids data/dark_weak_test/atlas_inference_ids_v061/bp_protein_ids.txt \
  --source-index "$HISTORICAL_WEAK_MSA_INDEX" \
  --source-index "$HISTORICAL_CORE_MSA_INDEX" \
  --out-index data/dark_weak_test/atlas_msa_bp_v061/index.pkl
```

若一个source index已经完整覆盖，只传一个--source-index即可。重复ID会要求检查来源优先级；不要未经核实就覆盖。工具引用原shards，不重算MSA。MSA数据库的历史时间来源仍需沿用已审计版本。

### Expert概率按ID对齐

如果原始概率的行顺序就是源metadata中的exp_train：

```bash
python scripts/dark_weak/align_atlas_expert_v061.py \
  --source-prob "$HISTORICAL_FULL_EXPERT_PROB" \
  --source-metadata data/unidata_with_exp_train_pseudo.pkl \
  --source-split exp_train \
  --target-ids data/dark_weak_test/atlas_inference_ids_v061/bp_protein_ids.txt \
  --task bp \
  --output-dtype float32 \
  --out-prob outputs/dark_weak_eval/atlas_expert_bp_v061/expert.prob.f32.npy \
  --out-ids outputs/dark_weak_eval/atlas_expert_bp_v061/protein_ids.txt
```

仅当行顺序确实对应exp_train时使用此命令；如果有独立源ID文件，改用 `--source-ids "$HISTORICAL_FULL_EXPERT_IDS"`，去掉source-metadata和source-split。所用概率必须是原训练GO列顺序；列宽相同本身不能证明GO身份相同。ID对齐也不自动证明时间独立。

缺少目标ID会在输出前报错；不填零。默认float32保留原float32的小概率信号，但无法恢复上游已经丢失的精度。

## 6. 用已验证的Stage-1 exporter导出新队列

用户现有0.5.1 task-alias修复继续使用，本包不覆盖exporter。复用此前成功的checkpoint与训练args配置，仅换新队列metadata、MSA index、目标ID和已对齐Expert：

```bash
python scripts/dark_weak/export_stage1_predictions.py \
  --checkpoint "$STAGE1_BP_CHECKPOINT" \
  --task bp \
  --num_classes 21312 \
  --file_address data/unidata_with_atlas_inference_v061.pkl \
  --working_address data/dark_weak_test/atlas_msa_bp_v061/index.pkl \
  --mode atlas_inference_v061 \
  --expected-protein-ids data/dark_weak_test/atlas_inference_ids_v061/bp_protein_ids.txt \
  --go-ids data/dark_weak_test/unified/labels/bp/go_ids.txt \
  --external_prob_path outputs/dark_weak_eval/atlas_expert_bp_v061/expert.prob.f32.npy \
  --external-protein-ids outputs/dark_weak_eval/atlas_expert_bp_v061/protein_ids.txt \
  --export-expert \
  --export-modelout \
  --decoder-prob-source expert \
  --modelout-contract v088_weak_supervision \
  --query_decoder_logit_base_mode mix_expert_base_anchor \
  --output-dtype float32 \
  --output_dir outputs/dark_weak_eval/stage1_atlas_bp_epoch100_v061
```

模型架构、解码器和args.json必须与原checkpoint实际训练一致；若原训练模式不同，应保留原分支并使用其真实协议，不能为了运行而强改模式。上述命令对应此前已使用的mix_expert_base_anchor/expert分支。保留之前成功命令中其他必要架构/loader参数，旧args文件移动时用原支持的--train_args_json路径。

旧导出目录旁prediction_manifest.json保存checkpoint路径及哈希；旧complete评估manifest只保存概率和ID引用，并没有完整GPU推理配置。路径失效时应指向原同一checkpoint并核对身份。

## 7. 直接从新导出目录评价，不手填config

先核验覆盖，再正式评价，使用不同输出目录：

```bash
python scripts/dark_weak/eval_atlas_v061.py \
  --stage1-export-dir bp=outputs/dark_weak_eval/stage1_atlas_bp_epoch100_v061 \
  --namespaces bp \
  --inventory-only \
  --out outputs/dark_weak_eval/atlas_v061_bp_full_coverage

python scripts/dark_weak/eval_atlas_v061.py \
  --stage1-export-dir bp=outputs/dark_weak_eval/stage1_atlas_bp_epoch100_v061 \
  --namespaces bp \
  --train-metadata data/unidata_with_exp_train_pseudo.pkl \
  --train-label-field prop_annotations \
  --classification future \
  --out outputs/dark_weak_eval/atlas_v061_stage1_bp
```

旧的--complete-manifest入口仍可使用，但必须指向实际覆盖6672人的预测。新入口自动读取三个通道；未提供NBS时增量恢复表为空，其他Stage-1四组结果正常输出。NBS仍在训练时不影响先完成这些对照。

如果最初的176人目录再次传入新入口，覆盖检查仍会显示缺6496人，这是正确行为。新入口不降低数据要求。
