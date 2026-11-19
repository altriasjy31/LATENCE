# 使用原 DeepGOZero 附件的 JAR 生成 go.norm

这次附件包含原 DeepGOZero 项目及 `jars/` 中的 59 个 JAR，其中有 JCEL 0.24.1、OWLAPI 4.5.19 和 OBO 格式解析器。因此此前“只有 OBO 不能生成 go.norm”的表述需要修正。**这里可以直接读取现有 `go-2025_06_01.obo`，并处理其中的 `is_a`、`relationship`、`intersection_of` 等逻辑内容。**普通图边导出与 OBO 文件经 OWLAPI 转成逻辑公理不是同一过程。

但这并不意味着 OBO 和同日期 GO-Plus OWL 一定包含相同公理。结果需要记录具体文件及其哈希，不能将 OBO 结果称为完整 GO-Plus 复现。无逻辑约束、未被规范化公理覆盖的新 GO 仍会报告缺失，不会人为给它补边。

## 为什么没有直接照搬原 Normalizer.groovy

核查与实际微型本体测试发现两个独立问题。

1. 原脚本的 `@Grapes` 请求 OWLAPI 4.2.5，与附件 JAR 的 4.5.19 不同；仅增加 classpath 仍可能触发联网下载或混用版本。
2. 更关键的是，原脚本给规范化器另建了一个 `IntegerOntologyObjectFactory`，与翻译现有 GO 时的整数 ID 空间分离。规范化交集和存在量词时，新辅助类的 ID 可能撞到原 GO。把 factory 改成共用以后，原来的反向渲染器又没有新辅助类的 IRI 映射，会输出 `Ignoring` 并丢失公理。

这不只是依赖安装问题。我们用一个真实运行的微型 OBO 验证了原逻辑 `C3 ≡ C1 ∩ ∃part_of.C2`。原脚本对应的 Java 调用过程错误地生成 `C1 ∩ C2 ⊑ C3`。纠正后引入独立的 `ZC_0000000`，生成以下五条模型可消费的公理。

```text
GO_0000001 and ZC_0000000 SubClassOf GO_0000003
GO_0000002 SubClassOf GO_0000001
GO_0000003 SubClassOf GO_0000001
GO_0000003 SubClassOf part_of some GO_0000002
part_of some GO_0000002 SubClassOf ZC_0000000
```

`ZC_*` 是新的逻辑辅助类，既不覆盖原 GO，也不进入评估候选。`symbol_mapping.tsv` 保存整数 ID、符号、原 IRI 和辅助类标记。非 GO 类同样显式映射到长度兼容的符号，因为原 Python NF1 解析器要求两侧类名各长 10 个字符。

本包默认使用 **Java 11+ source-file 模式的纠正版**，共享翻译器与规范化器的 factory，并按 JCEL 的 GCI0–GCI3 类型直接输出 NF1–NF4。仍使用附件 JAR，不下载新依赖，不修改原仓库。默认路径不需要 Groovy。此修正应在论文实现说明中记录，不能称为字节级复现原 normalization 脚本。

## 运行

在 LATENCE 项目根目录运行，确保 `scripts/nbs/DeepGOZeroNormalizer.java` 与 Python 包装器一起复制。

```bash
DGZ_REPO=/absolute/path/to/deepgozero
GO_NEW="$LATENCE_DATA/go-2025_06_01.obo"
DGZ_NORM=outputs/latence_nbs_eval/external_unseen_bp/deepgozero_go20250601_norm

# 检查输入、JAR 内容和运行命令；不创建输出目录。
python scripts/nbs/normalize_deepgozero_ontology.py \
  --repo-dir "$DGZ_REPO" --ontology-file "$GO_NEW" \
  --candidate-go-ids "$EXT/new_go_ids.txt" \
  --output-dir "$DGZ_NORM" --dry-run

# 实际规范化；默认 corrected / Java 11+。
python scripts/nbs/normalize_deepgozero_ontology.py \
  --repo-dir "$DGZ_REPO" --ontology-file "$GO_NEW" \
  --candidate-go-ids "$EXT/new_go_ids.txt" \
  --output-dir "$DGZ_NORM"
```

随后将 `$DGZ_NORM/go.norm` 传给原 DeepGOZero 重训适配器的 `--normal-forms`。先看 `candidate_coverage.tsv` 和 `normalization_manifest.json` 的 NF1–NF4 数量与候选覆盖；这里尚未拿到用户服务器上的真实 2025 OBO，不能预先保证所有 265 个 GO 均被覆盖。

需要 Java 11 或更高版本，并包含 `jdk.compiler` 模块以支持源文件启动。可用 `--java-bin /absolute/path/to/java` 指定。仅缺少独立 `javac` 命令不一定有问题，本次环境没有 `javac` 命令，但 Java 21 的 source-file 启动已实际完成 OBO→go.norm。这里没有声称附件 JAR 已在所有 Java 版本上验证。

本体如声明 imports，先将所需依赖固定版本并合并为本地文档。默认输入检查会拒绝 import 声明，Java 后端也将 import IRI 映射到禁止载入的本地文件，避免规范化过程悄悄读取最新远程本体。

## 输出与边界

| 输出 | 含义 |
|---|---|
| `go.norm` | 纠正版的 NF1–NF4 类公理，按文本排序后冻结；训练和预测必须复用同一文件、同一顺序 |
| `normalization_manifest.json` | 输入、源脚本、Java 后端、JAR 和结果哈希，Java 版本、运行命令、模式、候选覆盖 |
| `symbol_mapping.tsv` | 原类/角色与辅助符号的映射 |
| `excluded_normal_forms.tsv` | 模型不消费的其他规范化公理类型与其整数表示 |
| `normalizer_stats.tsv` | 输入公理数量、规范化数量和被排除类型数量 |
| `untranslated_input_axioms.tsv` | 不能翻译或被译为空集合的逻辑公理；非空时直接失败，避免以不完整结果继续训练 |
| `candidate_coverage.tsv` | 每个候选 GO 是否真正进入原 Python 解析器的类集合 |
| `normalizer.log` | 后端完整日志 |

原 DeepGOZero 模型仅使用 NF1–NF4。角色链、角色包含等其他类型在这里单列，不进入模型损失；例如 `part_of` 的传递性不能因为生成了本体文件就声称被该模型显式学习。OWL 顶/底类及非 GO 类也会记录映射；`special_builtin` 列及 manifest 单独列出 `owl:Thing`、`owl:Nothing`，模型不会自动强制它们取全集/空集语义。排序冻结的是本次产物，不保证不同 JCEL/JVM 运行的辅助类分配必然完全一致；后续推理必须保留本次文件和哈希。

当前已用附件实际 JAR 和 Java 21 完成包含交集与存在量词的 OBO 测试，正确产生四种 NF，并验证候选覆盖和完整 Python 解析。独立的两元素有限模型枚举验证了该例的保守辅助类扩展。它不等同于完成全量 GO/GO-Plus 每条公理的语义证明。非 EL 文档如果后端不能翻译会失败，不应擅自删掉报错公理继续宣称完整复现。

## 原 Groovy 仅作诊断

若需要复查原行为，运行 `--mode native --groovy-bin /path/to/groovy`。包装器仅在输出目录生成去掉 `@Grapes` 的临时脚本副本，使用本地 JAR，原文件不变。Groovy 自身和 `CliBuilder` / commons-cli 不在附件 JAR 中，必须来自可用的 Groovy 安装。

该模式只生成 `native_diagnostic.norm`，manifest 标为 `complete=false`、`status=native_diagnostic_unsafe`，不输出供新训练使用的 `go.norm`。已有作者 checkpoint 仍必须使用它原来的配套规范化文件；不能换上本轮纠正版后继续解释旧 embedding 行号。

## 测试

```bash
DEEPGOZERO_SOURCE="$DGZ_REPO" python tests/test_deepgozero_normalizer.py
```

测试包含本地 JAR 检查、依赖注解隔离、import 拒绝、原解析器不支持的行检测、无运行时 preflight，以及真实 Java/OBO/NF/候选覆盖的端到端过程。有限模型语义检查脚本也随包提供。
