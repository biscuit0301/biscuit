# biscuit · A 股量化研究骨架

面向 A 股深度强化学习研究的可执行 Python 项目起点。默认流程只依赖 Python 标准库，使用固定种子的**合成行情**完成数据、特征、策略、回测和离线模拟交易；可选安装 Gymnasium / Stable-Baselines3 后训练 PPO。

目前是**单资产、日频、只做多的离线研究原型**。尚未接入真实行情、券商或实盘交易，示例结果不代表任何真实证券收益，也不构成已经验证的 A 股策略。

## 快速运行

要求 Python 3.11+。以下命令均从仓库根目录运行，适用于 PowerShell、bash 等终端；直接使用虚拟环境内的 Python 可省略激活步骤。

```sh
git clone https://github.com/biscuit0301/biscuit.git
cd biscuit
python -m venv .venv
```

激活虚拟环境：PowerShell 使用 `.venv\Scripts\Activate.ps1`；Linux/macOS 使用 `source .venv/bin/activate`。

```sh
python -m pip install -e .
python -m biscuit_quant demo --config configs/default.toml --output artifacts/demo
python -m unittest discover -s tests -v
```

`demo` 生成 120 条合成工作日行情并运行动量基线。输出包括：

| 文件 | 用途 |
| --- | --- |
| `synthetic.csv` | 固定种子生成的合成 OHLCV，非真实市场数据 |
| `manifest.json` | 输入 SHA-256、解析后的配置、版本和种子 |
| `metrics.json` | 总收益、最大回撤、费用、成交次数和期末持仓 |
| `equity.csv` | 每个估值日的权益、现金和持仓 |
| `trades.csv` | 信号日期、成交日期、价格、数量和费用 |

每次运行必须使用新的输出目录，防止覆盖已有研究证据。命令也可写为 `biscuit demo ...`。

## 目录

```text
configs/                 TOML 实验配置
data/
  raw/                   原始数据（默认不提交）
  processed/             中间数据（默认不提交）
  sample/                数据契约与示例说明
src/biscuit_quant/
  data/                  CSV 校验与合成行情生成
  features/              仅使用历史信息的收益率、动量特征
  envs/                  成交与账户环境、可选 Gymnasium 适配器
  strategies/            策略协议、动量与持有基线
  backtest/              回测与绩效指标
  paper/                 离线模拟交易 JSONL 事件账本
  config.py              配置解析和参数校验
  cli.py                 命令行与复现清单
scripts/                 PPO 训练入口
tests/                   数据、账户、时序、CLI、可选 PPO 测试
docs/                    架构、数据契约、研究流程与限制
.github/workflows/       基础 CI 与可选依赖 CI
```

## 外部 CSV 回测与模拟交易

```sh
python -m biscuit_quant backtest --data data/raw/example.csv --config configs/default.toml --output artifacts/backtest-001
python -m biscuit_quant paper --data artifacts/demo/synthetic.csv --config configs/default.toml --output artifacts/paper-001
```

CSV 必须依次包含 `date,open,high,low,close,volume,can_buy,can_sell`。每个文件仅包含一只证券，日期唯一、严格递增，交易标记为 `0` 或 `1`。详细约定见 [数据契约](docs/data-contract.md)。`paper` 是历史事件回放，输出 `ledger.jsonl` 和清单；没有实时行情轮询、断点恢复或券商连接。

## 可选 PPO

```sh
python -m pip install -e ".[rl]"
python scripts/train_ppo.py --data artifacts/demo/synthetic.csv --config configs/default.toml --timesteps 1024 --seed 42 --output artifacts/ppo-001
```

训练脚本按时间顺序划分 80% 训练、20% 留出测试，测试段仅借用训练尾部的历史数据预热特征。状态为日收益率、历史动量、仓位比例、现金比例；动作是空仓或满仓；奖励为扣费后账户的单日收益率。模型、留出指标和训练清单保存在输出目录。PPO 可能按 rollout 长度多采样，清单同时记录请求步数与实际步数。

合成数据只用于验证执行链路；单次种子和单段留出测试不能证明策略有效或收敛。真实研究还需要严格的样本外时间划分、多种子、基线对照、成本敏感性和数据偏差审查，见 [研究流程](docs/research-workflow.md)。

## 成交假设与边界

- 收盘 `t` 形成信号，开盘 `t+1` 成交，再按收盘 `t+1` 估值；特征不读取未来行情。
- 一天仅一次调仓，禁止做空和杠杆；当日买入最早在下一条行情卖出。此时序只适用于本原型的日频模型。
- 买卖数量按可配置整手单位取整；佣金、最低佣金、卖出税费和双向滑点进入账户。
- 零成交量或禁止该方向交易的标记阻止成交；涨跌停及其他限制由数据适配器提供方向标记，不通过统一固定比例推断。
- 期末持仓按收盘估值，**不强制平仓**，因此不包含假设期末卖出的额外费用。

默认配置中的数值仅为演示参数，并非对当前交易所规则或税率的确认。真实数据接入前须按市场板块、证券、交易日期核实交易单位、涨跌幅、停复牌、复权、公司行为和交易日历。模型没有成交量参与率、部分成交、排队、现金分红及拆合股记账，不适合直接迁移到科创板、创业板等不同规则市场。

## 开发与验证

```sh
python -m pip install -e ".[dev]"
python -m ruff check .
python -m unittest discover -s tests -v
python -m build
```

基础测试使用标准库 `unittest`；未安装 RL 依赖时仅跳过可选 Gymnasium/PPO 测试。CI 对 Python 3.11、3.12、3.13 运行基础测试、lint、构建及演示链路，并单独在 Python 3.12 检查可选 PPO 接口。

后续优先事项：真实行情适配及来源追踪 → 按日期/板块验证交易规则 → 点时数据与公司行为 → 多资产环境 → 可复现 PPO 实验与走步检验。架构扩展位置见 [架构说明](docs/architecture.md)。
