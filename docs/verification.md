# 初始骨架验证记录

验证日期：2026-11-06。环境：Windows，Python 3.12.10，CPU。

已实际运行：

- `python -m ruff check .`：通过。
- `python -m unittest discover -s tests -v`：17 项全部通过，包括 Gymnasium contract 检查与 16 步 PPO smoke test。
- `python -m build`：成功生成 sdist 和 wheel。
- `demo`、`backtest`、`paper`：全部完成；同一输入的 demo 与 backtest 指标一致；paper 与 backtest 一致性另有测试覆盖。
- `scripts/train_ppo.py --data artifacts/demo/synthetic.csv --config configs/default.toml --timesteps 128 --output artifacts/ppo`：完成训练、保存模型和留出测试输出。
- `python -m pip check`：无依赖冲突。

本地已验证的可选依赖：Gymnasium 1.4.0、Stable-Baselines3 2.9.2、NumPy 2.5.4、PyTorch 2.15.0。依赖声明保留兼容版本区间，并非跨平台锁文件。

种子 42、默认配置、120 条合成行情的基线回测产生 114 个执行步骤、35 笔成交，期末权益为 75040.0030620667，初始现金为 100000。该结果仅用于复现工程路径，不能解释为真实证券表现。原始运行产物保存在本地 `artifacts/`，默认不纳入 Git；可按 README 重建。

仓库另配置 Linux / Python 3.11、3.12、3.13 基础 CI 及 Python 3.12 的可选 RL CI。本记录的本地通过结果不等同于这些远程任务已通过。
