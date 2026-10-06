# v0.8.8 必要正确性验证

性能状态：**未验证**。本轮没有真实BP训练权重或BP测试预测，不能将以下测试替代Fmax结果。CPU生成的小型测试权重不随包作为实验模型交付。

已在Python 3.12、PyTorch 2.4.1+cpu上通过178项测试：模型26、采样23、loss/先验35、真实数据loader34、训练/续训9、完整导出与评测1、评测/shell/验收34、安装器16。真实CPU集成使用可复现小型CSR/mmap数据；完整导出交接覆盖2903列CC测试夹具。它不是1800×21312的真实BP性能测试。

验证范围直接对应本轮修改与验收：

- 与单关系普通GraphSAGE公式一致，自身变换只加一次，关系间求和；按节点映射添加BN前raw residual。
- 真实三层依赖与关系身份保留；整个DDP宏批查询原ID从各层支持图排除，标签不能经多跳返回；推理分批结果一致。
- 完整GO统一分母BCE；低分负项有梯度；四micro累积及模拟双rank角色分割与完整macro的loss/梯度相符。没有再次除4。
- 二元成员监督、alias未知项遮罩与训练先验来源正确，holdout不进入先验。
- 实际v088 stage通过真实loader，不伪装版本或修改旧loader；错误来源与合同被拒绝。
- Dropout开启时，中断/续训与不中断训练的模型、优化器、调度、RNG、stream和验证结果一致；短smoke不运行完整验证。
- CPU训练→检查点→独立查询图→全列预测→真实评测子进程；导出等于模型直接输出，无E/M输出依赖。
- Fmax-only与未修改v085原内核逐字段一致，包括并列分数、空gold与无正例边界。未运行AP排序、TopK或bootstrap。原指标文件SHA256为`d056ecb578f59ace9cf06c9aeb7e287d0b0657d77fd1961ab25c396d1dc0d314`，与历史BP报告相同。
- 验收拒绝换输入、换标签、换GO列、改口径、融合与暂停消融；Fmax=60被判未达标，使用其他Fmax字段不能绕过60.723149门槛。
- 安装前完整校验、重复安装、冲突拒绝、写入失败回滚；早期版本代码和数据不覆盖。

运行已安装项目的新版本测试：

```bash
PYTHONPATH=.:nbs_models/nbs_protein_go:nbs_models/nbs_protein_go/tests \
  OMP_NUM_THREADS=1 python -m pytest -q \
  nbs_models/nbs_protein_go/tests/test_*v088.py tests/test_*v088*.py
```

安装器测试在解压目录运行：

```bash
python -m pytest -q latence_nbs_v088/test_install_v088.py
```

未验证边界：真实BP完整数据、GPU显存与吞吐、CUDA/NCCL多卡执行、实际Fmax增益。环境中没有生产图/Box数据与元数据文件，`torch.cuda.is_available()`为False。用户服务器上的独立20-update smoke是必要的实际运行检查；完成训练后仍须执行固定Fmax验收。

最终安装实测：在完整旧项目的隔离副本中，58项依赖哈希校验通过，安装21个新文件，原379个文件逐字节不变；第二次安装新增0文件。随后在该安装目标中运行162项新模型/运行链测试，全部通过（13.80秒）；解压包内安装器测试16项全部通过（0.28秒）。检查器用真实原BP基线、缺少新结果运行，返回码2且状态为“未验证”，新Fmax与提升量均为null。
