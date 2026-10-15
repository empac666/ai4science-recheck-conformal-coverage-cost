# R13：PC³ 物理边界投影感知保形区间完整预注册

版本：v1.2；修订登记日期：2026-10-14（北京时间）；前版：v1.1。本轮唯一有效第二审为 `grok_review2.md`（REVISE AND RESUBMIT）；其余第二审没有有效正文，不用作修订依据。选题仍为 PC³，候选排序与总分 98 不变。**本稿仅冻结未来复核协议，本次修订没有启动正式实验、训练烟测或生成新的性能结果；表内数字均是原文参考或预先规定的门槛。** 正式计时前须保存本稿、执行代码、判定函数、任务清单和环境锁定文件的 SHA-256。之后任何修订另开版本，不覆盖冻结稿。

## 1. 研究对象、来源与解释范围

论文：*An Uncertainty-Aware Decision-Support System for Composite-Property Prediction with Physically Admissible Conformal Intervals*，Information 17(10), 946，2026-09-24；[论文与表 8/9](https://www.mdpi.com/2078-2489/17/10/946)，[DOI](https://doi.org/10.3390/info17100946)。实测数据仅为 [UCI Concrete](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)。不扩展到其他实测材料，不跑整篇论文其他基线，不验证解释模块或决策认证案例。FRP 是合成机制诊断，不能充作第二个实测数据集。

源码为论文数据可用性声明所指公开仓库 `pc3-conformal`，固定提交 `4c5befc6305bdcdda11f4411958c6373f9028f56`。必要文件为 `robust_concrete.py`（表 8/9 入口）、`pc3.py`（拟合及有限样本秩）、`grouped_families.py`（配方构造参考）和 `frp_experiment.py`（生成器）。已有 v1.1 来源核验记录和本轮只读源码文本证据作为依据；正式执行须核对固定提交并记录逐文件 SHA-256。公开稿通过 DOI 的数据可用性声明定位仓库，不列含账户标识的仓库 URL；私有来源清单保留完整来源和下载核验信息，不能直接整体公开。

Concrete 的上界是训练强度分位数形成的经验边界，不能称为实验证实的严格物理定律。名义覆盖固定 `1-α=0.90`。限制在物理走廊内的区间总体覆盖上限为 `1-η`；`1-max(α,η)` 仅是本研究的目标/可行前沿参考，不是任意区间的普适上限。测试边界外比例仅作事后描述，不能用来选边界、删种子、删行或决定是否评价 P93。

## 2. 假设与确认性问题

**主点永久固定 P93**：每个划分取 `L=0`、`U=float(np.percentile(y_train,93,method="linear"))`。不随新测试集 η、覆盖或接近原表的程度换点；不可行时也保留输出。

| 假设 | 原文 P93 参考 | 本协议问题 |
|---|---:|---|
| H1：投影感知校准补回朴素裁剪的覆盖损失 | 85.3% → 89.9%，提高 4.6 个百分点 | A 对齐原数字；B 检验同底模正增益和接近 90% |
| H2：覆盖恢复付出宽度代价 | 18.6 → 23.3 MPa，增加 4.7 MPa，约 25.27% | A 对齐原入口宽度；B 检验正的绝对宽度代价 |
| H3：合并覆盖与宽度后的评分效率 | 表 8/9 无 区间评分 IS 参考值 | 独立评价 R-N，不虚构原文目标；空区间不能带来改善结论 |
| H4：未见配方改变覆盖与增益 | 无同协议参考值 | C 与 B 比较，只评价覆盖与增益，不替代 A/B |

原文同点测试边界外比例约 6.2%，仅供对齐诊断，不要求新划分达到该值。**只有 A 判“原数字复现”；B 判去重数据上的新划分稳定性，不能把 B 的均值套进原表的六个容差带。** 桥接臂 D、C、非单调消融、上界曲线、校准子样本、归因诊断和 FRP 均不得挽救或翻转 A/B。H3 始终独立并列，H1/H2 成功不能被写成综合评分效率优越。

## 3. 数据冻结、去重与可独立复现的划分

### 3.1 数据及配方键

使用原入口对应的同一冻结 CSV，保留原文件行序，读入后给原行号 `0..1029`。列顺序固定为 `cement, slag, ash, water, superplastic, coarseagg, fineagg, age`，目标 `strength`，MPa；9 列均以 float64 解析。记录来源、下载 UTC/北京时间、字节数和哈希；断言 1030 行、8 输入、1 标签、全部有限、无缺失、单位一致。现有镜像 CSV 的 SHA-256 为 `d7d8bd087f832935e902bcb2687667238cac3fe06799677a93ca2f46dce8db02`；既有核验表明其浮点行序及行多重集合与官方表一致。正式输入必须匹配此哈希，不能静默换数据。

A 保留全部 1030 行，包括完整重复。B/C/D 在划分前执行 `df.drop_duplicates(subset=features+["strength"],keep="first")`，不排序、不重置原行号，预期 1005 行。只删除 9 列数值完全相同的记录，不合并特征同而标签不同的记录，不按标签大小、误差或边界状态删行。保存完整的重复记录到首次记录映射。

配方键不读标签：从去重数据按固定顺序取除 `age` 外的 7 列，明确使用 `np.round(...,decimals=6)` 的最近偶数舍入及 float64 运算，不使用笼统的“四舍五入”、字符串排序或其他舍入函数。匿名 ID 采用 Python 原生元组字典序：

```python
comp = ["cement","slag","ash","water","superplastic","coarseagg","fineagg"]
a = np.round(df_unique[comp].to_numpy(dtype=np.float64), decimals=6)
keys = [tuple(float(v) for v in row) for row in a]
ordered = sorted(list(set(keys)))
lookup = {key: i for i, key in enumerate(ordered)}
group_id = np.asarray([lookup[key] for key in keys], dtype=np.int64)
assert len(ordered) == 427
```

映射保留原行序，同配方不同龄期同组。1030/1005/427 任一数目不符或输入哈希不符，均是数据审计硬停止：报告未交付，不凑行、不改舍入、不接受新组数。无插补、标准化、标签变换、异常值筛除或标签分层。

### 3.2 分支、种子及两次行划分

所有种子写死。每个分支内部 U/N/R 共用划分、底模预测和校准样本；单调与非单调共用样本划分。

| 分支 | 数据及种子 | 作用 |
|---|---|---|
| A | 原始 1030 行；`0,1,2,3,4,5` | 原数字复现，P93 与单调全上界曲线；保留非单调 U 原对照 |
| B | 去重 1005 行；`100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119` | 新随机行划分稳定性 |
| C | 与 B 相同去重数据、相同 20 种子 | 配方分组留出压力 |
| D | 与 B 相同去重数据；`0,1,2,3,4,5` | 单调 P93 桥接诊断，不入判定；不再加“原始数据＋新种子”臂 |

A/B/D 及 FRP 的行划分均采用以下两次独立初始化的调用，`ids` 为本分支按原行序排列的行 ID；不在分区内排序：

```python
train_ids, tmp_ids = train_test_split(
    ids, test_size=0.5, random_state=int(s), shuffle=True, stratify=None)
cal_ids, test_ids = train_test_split(
    tmp_ids, test_size=0.5, random_state=int(s), shuffle=True, stratify=None)
```

复现必须调用事前锁定的 scikit-learn 补丁版本的 `train_test_split`，不得用手写“打乱后前半训练”替代。第一个返回数组分别是 `train_ids` 与 `cal_ids`；第二个返回数组分别是 `tmp_ids` 与 `test_ids`。在论文披露的 1.8.0 实现中，每次 `ShuffleSplit` 独立生成置换 `p`，第二个返回数组取输入的 `p[:n_test]`（测试前缀），第一个返回数组取 `p[n_test:n_test+n_train]`（训练后缀）；第二次的输入保持第一次 `tmp_ids` 的返回顺序，故校准是第二次置换后缀、最终测试是第二次置换前缀。每次传整数 `s` 重新初始化随机状态，不共用 Generator/RandomState 的连续流。浮点 `test_size=0.5` 的测试数量为 `ceil(n*0.5)`，训练数量是剩余；第二次对 `n_tmp` 同样向上取整。因而 A 为 515/257/258，B/D 为 502/251/252，FRP 为 750/375/375（训练/校准/测试）。不得删行凑成其他表的 257 个测试样本。调用接口参见 [scikit-learn 1.8 划分文档](https://scikit-learn.org/1.8/modules/generated/sklearn.model_selection.train_test_split.html)。

### 3.3 C 的组划分与公平比较

```python
u = np.unique(group_id)                 # 升序的 int64 数组
rng = np.random.RandomState(int(s))
rng.shuffle(u)                          # 明确原地打乱
G = len(u)
n1, n2 = int(np.floor(0.5*G)), int(np.floor(0.75*G))
S_train, S_cal, S_test = set(u[:n1]), set(u[n1:n2]), set(u[n2:])
```

按这些集合回取去重表的原行序，组内所有行归属同一分区。427 组对应 213/107/107 组，目标比例 50/25/25；实际行数因组大小不同不相等，不拆组或按标签优化比例。拟合前断言行 ID 互斥且并集完整，C 还断言组 ID 互斥且并集完整。C 不宣称行级交换性保证；主指标按测试行计算再等权平均种子，等配方权重覆盖另作诊断。

公平性指分支内 U/N/R 的严格配对，及 B/C 相同冻结样本、目标比例、参数和种子标签下的设计配对；不声称 B/C 的测试行一一相同。D-A 在相同种子标签和划分算法下展示去重引起的变化（删除记录也会改变置换和训练边界）；B-D 在同一去重数据上展示新旧种子集合的差异。D/A 可按种子标签报告差值但不作同测试行推断；B 与 D 的种子集合不同，只有描述性均值差，不使用配对 bootstrap、不声称完整因果分解。原始/去重样本的差异绝不能归因成纯校准效应。

## 4. 底模、校准、区间及有限样本语义

### 4.1 明确拟合配置

每个划分、每种单调性拟合两个独立 `HistGradientBoostingRegressor`，分位分别为 0.05 和 0.95，都用 `random_state=0`，无中位模型。固定参数如下，输入为数值 float64 数组：

```python
params = dict(loss="quantile", max_iter=250, learning_rate=0.06,
              max_leaf_nodes=31, max_depth=None, min_samples_leaf=20,
              l2_regularization=1.0, max_features=1.0, max_bins=255,
              categorical_features=None, interaction_cst=None,
              warm_start=False, early_stopping=False, scoring="loss",
              validation_fraction=0.1, n_iter_no_change=10, tol=1e-7,
              verbose=0, random_state=0)
# 两个独立实例：quantile=0.05 / 0.95；monotonic_cst=本任务方向
```

Concrete 单调方向为 `[+1,0,0,-1,0,0,0,+1]`，非单调为全 0；FRP 为 `[+1,+1,+1,0,-1]`。不传验证集、样本权重，不调参。上游未显式关闭早停；在锁定 1.8.0 环境、本协议少于 10000 个训练样本且不传验证集的条件下，其 `auto` 不启用早停，本稿显式 `False` 冻结同一行为。该说明依据 [模型官方文档](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)，属于实现对齐推断，不是已完成的预测一致性实验。实际 `n_iter_` 与完整参数需记录；不得借改变默认值追数。

同划分模型仅拟合一次，上界、校准子样本、legacy 和全局边界诊断只重算分数/阈值/端点。A 拟合单调及非单调各一对（非单调只报 U）；B/C 两种底模均报 U/N/R；D 只拟合单调一对并报 U/N/R。

### 4.2 秩与方法定义

校准分数 `a_i=max(q_lo(x_i)-y_i, y_i-q_hi(x_i))`，允许负分，不事后截为 0。主方法固定 `legacy=False`，`k=(9*(n_cal+1)+9)//10`（Python 整数精确上取整，`n_cal` 为非负整数；禁止先乘浮点 0.90 再 ceil）；升序有限分数至少 k 个时取第 k 小，少于 k 个则 `Q=+∞`。数值 NaN、负无穷或非边界置换造成的非有限底分数是技术错误，不能过滤后继续；边界外点预先指定的正无穷是合法算法状态。

1. U（不投影）：原分数阈值 `Q_N`，区间 `[q_lo-Q_N,q_hi+Q_N]`。
2. N（朴素）：沿用 `Q_N`，端点 `lo=max(q_lo-Q_N,L)`，`hi=min(q_hi+Q_N,U)`。
3. R（投影感知）：校准标签在 `[L,U]` 外时置 `a_i=+∞`，其余不改，得到 `Q_R`；端点使用与 N 相同的求交式。`Q_R=+∞` 明确返回 `[L,U]` 并记录退回；`Q_N=+∞` 时 N 同样返回 `[L,U]`、U 返回扩展实数区间 `[-∞,+∞]`。

Concrete 的走廊宽度常数，不用归一化分数；FRP 也按此冻结的同底模未归一化 CQR 分数做机制诊断，不调用含中位模型或其他归一化选项的整套入口。A 的非单调 U 是原入口对照，不能将它与单调 R 的差全归因于校准；机制比较始终为同底模 R-N。

`lo>hi` 精确判空；保留求交端点，不换端点、不改中点、不扩走廊、不删测试行。覆盖使用精确闭区间比较，判空和主判据不加 epsilon。断言采用绝对公差 `ε=1e-8`（Concrete 为 MPa，FRP 为 GPa）、相对公差 0，仅用于实现等式/不等式和独立汇总核查；不能把微小逆序变为非空或扩宽接受带。端点完成求交后以 float64 原样存储，禁止先舍入、转换精度或再计算另一套端点。有限端点仅做一次 `endpoint_delta = np.float64(hi) - np.float64(lo)`，保存该差；`empty=(lo>hi)` 仍用存储端点精确比较，`near_zero_empty = empty and (-epsilon <= endpoint_delta < 0)`，其中 `epsilon=1e-8`，即数学上的 `0<lo-hi<=epsilon`。W_raw、W_set、非空主宽度及评分的宽度项共同复用 `endpoint_delta`，不再反向相减来判近零。合法无限端点另按第 5.1 节赋值，避免非有限相减。

逐行同时保存 `base_cross=(q_lo>q_hi)`、`preclip_empty=(q_lo-Q>q_hi+Q)`、`outside_corridor=(q_lo-Q>U or q_hi+Q<L)` 和最终 `empty`；根因标志允许并存，不强行互斥。交叉不必导致最终空区间，负阈值也可能造成投影前空区间。无穷分支先显式处理，避免 `∞-∞`。

手算例只核查语义，不对照原表：分数例 `q_lo=10,q_hi=20,y=15` 得 `a=-5`；非空区间 `[10,20]` 在 `y=25` 的 区间评分 IS 为 `10+20*5=110`。秩例 `n=10,α=0.1` 得 `k=(9*11+9)//10=10`；分数 `-1,0,1,2,3,4,5,6,7,8` 得主 Q=8，把最后一项置正无穷则 Q=正无穷；`n=8` 得 `k=(9*9+9)//10=9`，必退回。另加整数点核查：`n=9` 得 k=9，`n=19` 得 k=18。`n=20`、分数 `0..19` 得 `k=(9*21+9)//10=19`，主 Q=18，隔离 legacy 的 `np.quantile(scores,min(1,k/n),method="higher")` 得 19。

## 5. 指标、空区间与不确定性计算

### 5.1 共同测试行与宽度口径

各任务每种方法保留同一全部测试行，均从新逐样本输出独立汇总。覆盖 `C=mean(1{lo≤y≤hi})`，空区间恒未命中。逐行宽度和评分遵循同一保守惩罚政策：

| 输出 | 覆盖 | 主宽度 w | 区间评分 IS IS |
|---|---:|---:|---:|
| 有限非空区间 | 精确闭区间命中 | `hi-lo` | 下述公式 |
| 空区间 | 0 | `+∞` | `+∞` |
| 合法退回 `[L,U]` | 正常计算 | `U-L` | 正常计算 |
| U 的 `[-∞,+∞]` | 1（标签有限） | `+∞` | `+∞`，显式赋值 |

有限非空区间的 `IS=(hi-lo)+20*(lo-y)*1{y<lo}+20*(y-hi)*1{y>hi}`，单位与标签相同，越小越好。主均宽 `W=mean(w)` 与主 区间评分 IS `IS=mean(IS_i)` 均是扩展实数算术均值；一个正无穷即令该任务均值为正无穷。这样空区间在两指标上都受罚，不能借 0 宽度制造效率优势。空区间的正无穷是本协议保守评分约定，不称作标准公式的自然延拓。

同时保存 `W_raw=mean(hi-lo)`（原入口口径，可为负）、`W_set=mean(max(hi-lo,0))`（集合几何宽度）和非空子集 区间评分 IS。**A 对齐 18.6/23.3 及 4.7 MPa 的三项门槛只用 W_raw；B 宽度代价用主 W。** W_raw/W_set 是诊断及来源对齐字段，不能替换主效率指标，不能将空区间拉低的 W_raw 写成效率更好。A 数字复现成功若伴随空区间，必须并列披露惩罚主指标及 H3，不得声称完整效率改善。

空区间率 `E=mean(empty)`；`F_cal=退回任务数/任务数` 和 `F_row=退回任务测试行数/全部任务测试行数` 同报。有限阈值恰好裁满走廊另记 `full_corridor_rate`，不混作无穷退回。还报 `η_cal,η_test`、走廊均宽、U 端点违规率、底分位交叉率、裁剪激活率、各根因计数及 `Q_N,Q_R`。边界外比例用于有限测试集前沿说明，不用于修改任务。

### 5.2 种子汇总及 bootstrap

主汇总是**种子指标的等权算术均值**：A/D 为全部 6 个种子，B/C 为全部 20 个种子。不得替换为逐种子全通过或测试行池化。另报逐种子、全行池化、最小/最大值；标准差仅在全部有限时计算 `ddof=1`，否则记“未定义（合法无穷）”，同时给有限种子数、无穷种子数及有限子集摘要。

配对差每次在同划分计算 R-N，覆盖差用百分点，绝对宽度差及评分差用 MPa（FRP 为 GPa）。**确认性宽度代价是 ΔW=W_R-W_N 的绝对差；A 为 ΔW_raw。** 相对增幅 `100*(跨种子R均宽/跨种子N均宽-1)` 只描述，不是额外门槛，也不是逐种子比值均值。分母≤0、任一均值无穷时比例记不可定义，不运算。

B/C 的有限统计量采用 10000 次配对 percentile bootstrap：`rng=np.random.RandomState(900001)`，一次生成 `J=rng.randint(0,20,size=(10000,20))`，所有方法、分支及统计量共用 J。在每次重抽中先重算各方法的种子等权均值，再重算覆盖差、绝对宽度差、C-B 的两个差及可定义的描述比例；最后 `np.percentile(values,[2.5,97.5],method="linear")`。不对已得区间做减法，不以逐行或校准重复为重抽单位。A 六项向量按第 6.2 节直接由六种子均值重算，不用置信区间放宽容差。

**无穷先检查，bootstrap 后运行的做法禁止。** 对每个拟评价的指标比较，先检查所有参与种子和方法的完整均值状态。只要某参与均值为合法正无穷，跳过该比较的 bootstrap：不生成 `∞-∞`，不对含无穷数组调用 percentile；输出 `ci_status=skipped_legal_infinity`、种子索引及原因，并按第 6 节离散分支判定。其他全有限指标（如覆盖）仍可 bootstrap。有限输入重抽后出现溢出/NaN 是技术错误。不得删除无穷种子、截尾、温莎化或把分数记成 0。

序列化采用 `value=null,state=positive_infinity`，有限为 `state=finite`；技术缺失为 `state=missing/error`，并附原因。CI 跳过是已交付合法诊断，不是未交付；均值/表文件缺失、报错、超时才是未交付。区间只描述固定数据的随机划分稳定性，不代表 20 份独立实测数据或未来材料总体误差。

## 6. 单一判定函数：优先级、终态及不重叠规则

### 6.1 输入契约、对象门与统一入口

全部判断使用未四舍五入值。`judge(kind,record)` 是唯一科学判定入口，对任意输入均返回一个 `status∈{成功,失败,不确定}` 及原因码；未知 kind、类型错误、缺字段等也不能抛到三态之外。对象交付另列 `已交付/未交付`；整轮完整性另列 `完成/未完成`，两者不是第四种科学终态。

每个对象的输入范围固定：A 为六种子单调 P93 N/R 及对应主 U 诊断；B/H3 为 B 的二十种子单调 P93 U/N/R；C 为 C 和 B 的二十种子单调 P93 U/N/R 及两项 B/C 覆盖比较。A 的非单调对照、D、非主网格、有限校准子样本、归因列和 FRP 不属于这些对象的必需任务。按以下顺序执行：

1. **入口契约与对象交付**：只检查本对象任务、逐行结果、状态和所需汇总。缺失/崩溃/超时、短向量、缺 CI（且没有合法无穷短路）返回 `U_DELIVERY`；已提供但类型/单位错误、NaN、非法无穷、逆序 CI、状态矛盾返回 `U_AUDIT`。按下列入口代码的固定检查顺序返回首个原因类别，并在审计记录列出可复核的缺失/格式证据。对象内交付失败不得由诊断任务补齐。
2. **对象审计**：仅查第 9.1 节对象门 O1–O7；不通过返回 `U_AUDIT`，列编号和证据；不读取第 9.1 节轮次完整性 T1–T4。压力格尚未产生不是对象缓存错误。已有主格与同配置重复格在判定前均存在且哈希冲突时才由 O7 返回 `U_AUDIT`。
3. **对象环境门**：A/B/H3/C 均须实际匹配同一事前冻结补丁版本组，未匹配/未锁定返回 `U_ENV`。A 还须匹配论文披露精确组，否则 `U_ENV`。B/H3/C 使用已冻结但不同于论文的组时保留科学三态并附 `E_LOCKED_NONPAPER`；成功句必须明示该限制（第 10 节）。
4. **科学树**：通过上述门后按 A/B/H3/C 唯一分支树裁决，失败条件优先于边界重叠的不确定。对象门及三态记录原子锁定；预算超限、压力/FRP 缺失不会改写已锁定对象的三态。

合法无穷、CI 按规则跳过、双无穷“不可比较”、描述比例不可定义，均是**已交付的有效诊断**；不可比较编码 `U_INCOMPARABLE`。CI 缺失只有在对应参与种子合法无穷、`ci_status=skipped_legal_infinity`、来源审计和计数齐全时才能用 None 表示；不能只填无穷标志来规避交付。整轮完整性要求全部任务结果或事前允许的容量不可执行状态，以及预算等 T 类记录合规；资源中断不是合法不可执行。

以下为冻结规范伪代码，本次不执行。输入为解码后的普通 dict；数值先规范为 Python int/float（不接受 bool 冒充数值），缺字段与非法值由入口收口。`object_audit_pass` 等布尔值必须由 O 类证据和独立重算产生，不能来自人工自报。科学树只读规范后的本对象摘要；压力/FRP/预算数据不进入 `r`。

```python
import math

FIELDS = {
    "A": ("A_vector",),
    "B": ("ci_R_cover", "ci_gain", "ci_width",
          "any_N_R_width_infinite", "ci_width_status"),
    "H3": ("R_IS_infinite", "N_IS_infinite", "ci_IS_diff", "ci_IS_status"),
    "C": ("ci_group_cover", "ci_group_gain"),
}
UNITS = {
    "A": ["%", "%", "pp", "MPa", "MPa", "MPa"],
    "B": {"ci_R_cover": "%", "ci_gain": "pp", "ci_width": "MPa"},
    "H3": {"ci_IS_diff": "MPa"},
    "C": {"ci_group_cover": "pp", "ci_group_gain": "pp"},
}
COMMON = ("delivered", "object_audit_pass", "locked_environment_match",
          "exact_environment", "units")

def finite_number(x):
    return type(x) in (int, float) and math.isfinite(x)

def contract_gate(kind, r):
    if type(kind) is not str or kind not in FIELDS or type(r) is not dict:
        return "U_AUDIT"
    if any(f not in r or r[f] is None for f in COMMON):
        return "U_DELIVERY"
    if any(type(r[f]) is not bool for f in COMMON[:4]):
        return "U_AUDIT"
    if not r["delivered"]:
        return "U_DELIVERY"
    if any(f not in r for f in FIELDS[kind]):
        return "U_DELIVERY"
    if kind == "A":
        v = r["A_vector"]
        if v is None: return "U_DELIVERY"
        if type(v) not in (list, tuple): return "U_AUDIT"
        if len(v) != 6: return "U_DELIVERY"
        if any(x is None for x in v): return "U_DELIVERY"
        if not all(finite_number(x) for x in v): return "U_AUDIT"
        if not (0 <= v[0] <= 100 and 0 <= v[1] <= 100
                and -100 <= v[2] <= 100): return "U_AUDIT"
        ci_names, allow_none = (), False
    else:
        flags = {"B": ("any_N_R_width_infinite",),
                 "H3": ("R_IS_infinite", "N_IS_infinite"), "C": ()}[kind]
        if any(r[f] is None for f in flags): return "U_DELIVERY"
        if any(type(r[f]) is not bool for f in flags): return "U_AUDIT"
        allow_none = any(r[f] for f in flags)
        ci_names = {"B": ("ci_R_cover", "ci_gain", "ci_width"),
                    "H3": ("ci_IS_diff",),
                    "C": ("ci_group_cover", "ci_group_gain")}[kind]
        optional = {"B": "ci_width", "H3": "ci_IS_diff"}.get(kind)
        # None 只允许相应无穷比较，覆盖 CI 永远必需。
        if any(r[f] is None and not (f == optional and allow_none)
               for f in ci_names): return "U_DELIVERY"
        for f in ci_names:
            v = r[f]
            if v is None: continue
            if type(v) not in (list, tuple): return "U_AUDIT"
            if len(v) != 2: return "U_DELIVERY"
            if any(x is None for x in v): return "U_DELIVERY"
        for f in ci_names:
            v = r[f]
            if v is None: continue
            if not all(finite_number(x) for x in v) or v[0] > v[1]:
                return "U_AUDIT"
            if f == "ci_R_cover" and not (0 <= v[0] <= v[1] <= 100):
                return "U_AUDIT"
            if f in ("ci_gain", "ci_group_cover") and not (-100 <= v[0] <= v[1] <= 100):
                return "U_AUDIT"
            if f == "ci_group_gain" and not (-200 <= v[0] <= v[1] <= 200):
                return "U_AUDIT"
        if optional:
            status_field = "ci_width_status" if kind == "B" else "ci_IS_status"
            if r[status_field] is None: return "U_DELIVERY"
            expected = "skipped_legal_infinity" if allow_none else "finite"
            if r[status_field] != expected or ((r[optional] is None) != allow_none):
                return "U_AUDIT"
        if allow_none:
            if "infinity_detail" not in r or r["infinity_detail"] is None:
                return "U_DELIVERY"
            detail = r["infinity_detail"]
            if type(detail) is not dict: return "U_AUDIT"
            expected_seeds = list(range(100, 120))
            for method in ("N", "R"):
                if method not in detail or detail[method] is None: return "U_DELIVERY"
                z = detail[method]
                if type(z) is not dict: return "U_AUDIT"
                required = ("W_infinite_seed_ids", "W_infinite_seed_count",
                            "IS_infinite_seed_ids", "IS_infinite_seed_count", "rows")
                if any(f not in z or z[f] is None for f in required):
                    return "U_DELIVERY"
                for metric in ("W", "IS"):
                    ids, count = z[metric+"_infinite_seed_ids"], z[metric+"_infinite_seed_count"]
                    if type(ids) is not list or type(count) is not int: return "U_AUDIT"
                    if any(type(i) is not int or i not in expected_seeds for i in ids):
                        return "U_AUDIT"
                    if ids != sorted(set(ids)) or count != len(ids): return "U_AUDIT"
                rows = z["rows"]
                if type(rows) is not list: return "U_AUDIT"
                if len(rows) != 20: return "U_DELIVERY"
                for seed, row in zip(expected_seeds, rows):
                    if type(row) is not dict: return "U_AUDIT"
                    keys = ("seed", "test_n", "empty_rows", "near_zero_empty_rows")
                    if any(f not in row or row[f] is None for f in keys): return "U_DELIVERY"
                    if any(type(row[f]) is not int for f in keys): return "U_AUDIT"
                    if (row["seed"] != seed or row["test_n"] != 252 or
                        not (0 <= row["near_zero_empty_rows"] <= row["empty_rows"] <= 252)):
                        return "U_AUDIT"
                empty_ids = [row["seed"] for row in rows if row["empty_rows"] > 0]
                # N/R 的有限走廊下，无穷主宽度/评分只来自空区间。
                if (z["W_infinite_seed_ids"] != empty_ids or
                    z["IS_infinite_seed_ids"] != empty_ids): return "U_AUDIT"
            n_inf = bool(detail["N"]["IS_infinite_seed_count"])
            r_inf = bool(detail["R"]["IS_infinite_seed_count"])
            if kind == "B" and r["any_N_R_width_infinite"] != (n_inf or r_inf):
                return "U_AUDIT"
            if kind == "H3" and (r["N_IS_infinite"] != n_inf or r["R_IS_infinite"] != r_inf):
                return "U_AUDIT"
            # O3/O6 另对明细与逐行来源、摘要和原子表的一致性独立核验。
    if r["units"] != UNITS[kind]: return "U_AUDIT"
    return None

def scientific_tree(kind, r):
    if kind == "A":
        bands = [(83.3,87.3),(87.9,91.9),(2.6,6.6),
                 (16.74,20.46),(20.97,25.63),(3.2,6.2)]
        codes = []
        for j in range(6):  # 长度已在入口精确核查，禁止 zip 截断
            t, (a, b) = r["A_vector"][j], bands[j]
            if t < a: codes.append(f"F_LOW_A_{j+1}")
            elif t > b: codes.append(f"F_HIGH_A_{j+1}")
        return ("失败", codes) if codes else ("成功", ["S_A"])
    if kind == "B":
        c, g, w = r["ci_R_cover"], r["ci_gain"], r["ci_width"]
        if c[1] < 87: return "失败", ["F_LOW_B_C"]
        if c[0] > 93: return "失败", ["F_HIGH_B_C"]
        if g[1] <= 0: return "失败", ["F_NONPOS_B_GAIN"]
        if w is not None and w[1] <= 0:
            return "失败", ["F_NONPOS_B_WIDTH"]
        if r["any_N_R_width_infinite"]:
            return "不确定", ["U_EMPTY_WIDTH"]
        if c[0] >= 87 and c[1] <= 93 and g[0] > 0 and w[0] > 0:
            return "成功", ["S_B"]
        return "不确定", ["U_CI_BOUNDARY"]
    if kind == "H3":
        if r["R_IS_infinite"] and not r["N_IS_infinite"]:
            return "失败", ["F_IS_R_INFINITY"]
        if r["R_IS_infinite"] and r["N_IS_infinite"]:
            return "不确定", ["U_INCOMPARABLE"]
        if r["N_IS_infinite"]:
            return "不确定", ["U_EMPTY_N"]
        d = r["ci_IS_diff"]
        if d[1] < 0: return "成功", ["S_IS_IMPROVE"]
        if d[0] > 0: return "失败", ["F_IS_WORSE"]
        return "不确定", ["U_IS_ZERO"]
    if kind == "C":
        c, g = r["ci_group_cover"], r["ci_group_gain"]
        if c[1] < -2 or g[1] < -2:
            return "失败", ["F_GROUP_DEGRADE"]
        if c[0] >= -2 and g[0] >= -2:
            return "成功", ["S_GROUP_NO_LARGE_DROP"]
        return "不确定", ["U_GROUP_BOUNDARY"]
    return "不确定", ["U_AUDIT"]

def judge(kind, r):
    try:
        code = contract_gate(kind, r)
        if code: return "不确定", [code]
        if not r["object_audit_pass"]: return "不确定", ["U_AUDIT"]
        if not r["locked_environment_match"]:
            return "不确定", ["U_ENV"]
        if kind == "A" and not r["exact_environment"]:
            return "不确定", ["U_ENV"]
        status, codes = scientific_tree(kind, r)
        if kind in ("B", "H3", "C") and not r["exact_environment"]:
            codes = codes + ["E_LOCKED_NONPAPER"]
        return status, codes
    except Exception:
        # 仅封闭契约外输入解析异常，附不可覆盖的异常类型/字段证据；
        # 不重跑、不据此补值，不捕获资源终止来伪造已完成对象。
        return "不确定", ["U_AUDIT"]
```

入口的完整验证还包含 O1–O7 中按种子逐行重算出的向量/CI 与摘要一致性、无穷来源与计数一致性，不能以表面形状通过代替对象审计。入口以代码固定次序返回唯一 U_DELIVERY/U_AUDIT，随后才检查对象审计和环境，再运行科学树；禁止以缺失值或 NaN 越过任一门。组合只消费 A/B 已锁定状态及原因码（第 6.5 节）。

### 6.2 A：原数字复现，只读六种子等权均值

只取 A 单调 P93、完整校准、`legacy=False`、训练边界的 N/R。令 `bar_C_M=(1/6)*Σ_{s=0}^5 C_M,s`，`bar_Wraw_M=(1/6)*Σ W_raw_M,s`。判定六项向量固定为：

`T_A=(100*bar_C_N,100*bar_C_R,100*(bar_C_R-bar_C_N),bar_Wraw_N,bar_Wraw_R,bar_Wraw_R-bar_Wraw_N)`。

**接受带只作用于上述 6 种子等权均值；逐种子表和全行汇总只作对照，不进入函数。**

| 项目（按 T_A 顺序） | 接受带，全部含端点 |
|---|---|
| N 覆盖 | `[83.3,87.3]` %（85.3±2.0 个百分点） |
| R 覆盖 | `[87.9,91.9]` %（89.9±2.0 个百分点） |
| 覆盖增益 | `[2.6,6.6]` 个百分点（4.6±2.0） |
| N 原口径宽度 W_raw | `[16.74,20.46]` MPa（18.6±10%） |
| R 原口径宽度 W_raw | `[20.97,25.63]` MPa（23.3±10%） |
| 绝对原口径宽度差 ΔW_raw | `[3.2,6.2]` MPa（4.7±1.5） |

第 6.1 节就绪/审计/环境门通过后：若任一 `T_A[j]<lower[j]` 或 `>upper[j]`，返回失败；否则成功。为每个出带项报 `F_LOW_A_j` 或 `F_HIGH_A_j`，多项一并列，不把高侧偏出写成成功。空区间主 W/IS 无穷不改变此原入口数值门，但必须在首表单列 E、W_raw、W_set、W 和 H3。A 成功仅支持这六项数字对齐，不能掩盖区间质量缺陷。

容差是事前的复现误差预算：覆盖 ±2 个百分点用于容纳六划分均值的有限波动，宽度 ±10% 用于容纳数值幅度偏差，绝对差 ±1.5 MPa 额外约束宽度代价。它们是本次协议选择，不是论文标准、功效计算或可事后调整的阈值。

### 6.3 B：新划分稳定性，不对准原表水平

只取 B 单调 P93、20 个完整种子。用第 5.2 节重抽结果得到三项 95% CI：`I_R`（R 覆盖百分比）、`I_G`（配对覆盖增益，百分点）、`I_W`（配对主宽度差，MPa；仅所有 N/R 均宽有限时存在）。唯一覆盖接近名义的带为 `[87.0,93.0]` %，含端点；覆盖增益和绝对宽度差以 0 为方向界，成功要求严格大于 0。

第 6.1 节通过后执行：

```text
若 upper(I_R)<87 或 lower(I_R)>93：失败，分别 F_LOW_B_C / F_HIGH_B_C
否则若 upper(I_G)<=0：失败，F_NONPOS_B_GAIN
否则若 I_W 存在且 upper(I_W)<=0：失败，F_NONPOS_B_WIDTH
否则若任一 N/R 种子主 W 为合法 +∞：不确定，U_EMPTY_WIDTH
否则若 I_R 完整位于闭带 [87,93]，且 lower(I_G)>0 且 lower(I_W)>0：成功
否则：不确定，U_CI_BOUNDARY（列覆盖带或方向界重叠项）
```

先检查失败，之后才检查其余重叠或无穷，保证无重叠。例如覆盖 CI 已完全低于带且另一 CI 跨界，唯一终态为失败。任一空区间使宽度差不可定义，不能凭负宽度或集合宽度的 0 判成功；覆盖的有限、明确失败仍优先。B 只问恢复覆盖及其宽度代价能否在去重数据的新划分上保持，取消 v1.0 围绕原表的六项 B 容差；N 绝对水平、原表差和相对增幅仍可展示，不裁决。

`90±3` 个百分点是本次“接近名义”实用容差；增益/代价正方向是机制问题，没有另设最小幅度。多个边际 95% 区间的合取是保守的协议接受规则，**不是联合覆盖率已校正为 95% 的置信域，也不声称有精确校准的联合检验功效**。若仅 CI 与带/方向界重叠，应写“当前精度不足以裁决（与功效不足相容）”，不能断言机制失效或声称已经证明低功效。

### 6.4 H3：区间评分 IS 独立终态，先短路再 bootstrap

A 报全部六种子的 U/N/R 和状态，作为数值复现旁的描述；H3 的正式方向判定只取 B 单调 P93 的全部 20 个种子同底模 R/N。按第 6.1 节通过后，**在 bootstrap 之前**检查所有完整种子均值：

| 按序检查的条件 | 科学三态 | 诊断/原因码 |
|---|---|---|
| 任一 R 种子 IS 为 +∞，所有 N 种子 IS 有限 | 失败 | 评分恶化，F_IS_R_INFINITY；不做差/CI |
| N、R 均各有至少一无穷种子（不要求同种子） | 不确定 | 已交付“不可比较”，U_INCOMPARABLE；不做 `∞-∞` |
| 所有 R 有限、任一 N 为 +∞ | 不确定 | U_EMPTY_N；只记空区间修复，禁止据此判评分改善 |
| 两方法所有种子均有限 | 进入下述有限分支 | ΔIS 可定义，按第 5.2 节 bootstrap |

有限分支：若 `upper(CI_ΔIS)<0` 返回成功（评分改善）；否则若 `lower(CI_ΔIS)>0` 返回失败（评分恶化）；否则返回不确定（U_IS_ZERO，区间含 0）。任何 N/R 空区间都不能产生“成功/改善”；两方法同时无穷不是技术失败。A/B 其他主指标正常交付时，不可比较不妨碍交付完整性。

U 的合法无穷评分也须交付，其与 N/R 的比较只作描述，同样禁止对无穷做 bootstrap 或差值。覆盖恢复成功但 H3 失败/不确定时必须并列写出，不得宣布整体评分效率优越。所有 区间评分 IS 无限状态与空区间数量一并展示。`U_EMPTY_WIDTH`、`F_IS_R_INFINITY`、`U_INCOMPARABLE`、`U_EMPTY_N` 四码旁强制附第 9.2 节的 `infinity_detail`：N/R 各自的无穷种子数/ID、每个注册种子的空行数、其中近零空行数（包括零计数），W 和 IS 无穷分别统计；不借近零标签减轻惩罚或重判。

### 6.5 C 及结论组合

C 的诊断三态由两个 B/C 设计配对 CI 决定：`D_C=100*(bar_C_R,C-bar_C_R,B)`，`D_G=100*((bar_C_R-bar_C_N)_C-(bar_C_R-bar_C_N)_B)`。第 6.1 节通过后，先查任一 CI 上限 `<-2.0`：失败，F_GROUP_DEGRADE；否则若两个下限均 `≥-2.0`：成功，标签仅为“覆盖与增益未低于 -2 个百分点”；其余不确定，U_GROUP_BOUNDARY。阈值是允许的最小实用退化界，不约束宽度、不证明全面稳健或形式化组覆盖。C 的 W/IS、η、退回及所有无限诊断仍完整交付。

原数字复现结论 **完全等于 A 状态**；B/C/H3/D/压力不得翻转它。若另报“原数字对齐且新划分机制稳定”的组合结论，仅使用如下一个函数：A 或 B 任一失败则组合失败；否则 A/B 都成功则组合成功；其余组合不确定。该组合必须原样列 A/B 各自状态及全部原因码（包括环境限制码）；`F_NONPOS_B_WIDTH` 的固定报告句为“宽度正代价未成立”，即使覆盖增益为正也不改句；不能把 B 的失败描述为原表数字失败。整体任务未完成时仍可披露已完整分支的判定，但不得声称整轮全部交付。

结论首句和首表列的顺序固定：**任务完整性 → A 原数字复现 → B 新划分稳定性 → H3 区间评分 → C 配方诊断**；桥接 D 与压力随后。三态由同一个 `judge` 入口及其冻结原因码生成，报告不得二次改判。

## 7. 压力矩阵、主点同一性与归因诊断

### 7.1 注册矩阵及 P93 的唯一缓存

| 压力 | 固定配置 | 用途 |
|---|---|---|
| 经验上界收紧 | A/B/C 全种子，单调；`P99,P97,P95,P93,P91,P90,P88,P86`，训练标签分位 `method="linear"` | 全曲线、边界失配和退回；A 同时报告非单调 U |
| 去单调 | B/C 全 20 种子×全网格，方向全 0 | 同底模 U/N/R 机制比较，无调参 |
| 校准样本量 | B/C 单调 P93；`25,50,100,200,all`，有限 n 各 10 次嵌套子抽样 | 校准波动、空区间与退回，all 只算一次 |
| FRP | 20 个生成种子，每个 1500 行；固定 ρ 网格 | 合成机制诊断，不入任何主门槛 |

全网格的 A/B/C 单调 P93、完整校准 `all` 格**就是各对应主/配方任务**，必须复用已审计缓存及相同逐行结果哈希；all 也复用它。缓存键包含分支、数据哈希、划分 ID 哈希、种子、底模参数/源码哈希、边界来源/P、校准 ID 哈希、秩规则。禁止复制再拟合或另算一套 P93 供择优。主缓存自身的引用键、逐行哈希与独立重算必须在对象判定前按 O7 审计。重复压力格尚未产生时，O7 不要求其存在；主格与同配置重复格在对象锁定前均已存在且哈希不一致，才使该对象返回 U_AUDIT，保留两份证据，不择优。对象已通过门并锁定后，后来产生的重复格冲突只记 T2 失败及不可覆盖的偏差/影响说明，整轮未完成；不能重新调用 judge 改写已锁定主终态。非 P93 格错误也仅归轮次完整性。

非单调 P93、校准有限子样本 P93 是不同配置的压力格，不能冒充主缓存。同 P93 标签仅指相同训练边界定义，不指相同预测/阈值。压力只报告事前列出的描述量/固定 bootstrap 摘要，不做逐格 p 值、不按 η 重标主点、不计显著格子数，不用其有利格替代 A/B。

### 7.2 校准子样本

不改训练模型、U 或测试集，不从测试集补校准。对分支编号 `b=0`（B）、`b=1`（C），种子 s 和重复 `r=0..9`，用 `RandomState(200000+1000*b+10*s+r)`。B 复制当前校准 ID 顺序后原地 shuffle；C 对升序校准组 ID 原地 shuffle，逐组串接该组原行序记录。分别取序列前 25/50/100/200 行，保证同 r 的嵌套；C 可只取组内部分行，但不将该组移至其他分区、不宣称行独立。all 只复用完整结果，不作 10 次重复。

容量不足时该 n 的全部对应重复格预先标 `not_applicable_capacity`，保留实际容量/组数，不替代 n、不新增种子；这是已交付的合法不可执行状态。预计 B 校准 251 行，C 以冻结分区实际数量为准。单列覆盖、各宽度口径、IS、η_cal、退回次数、组数和原因计数，10 次重复不能充作 200 个独立种子。

### 7.3 A 的非裁决归因列

均限定 A 同一六种子、同一模型/校准/测试缓存、单调 P93；不得作为技术重跑，以下预先列出的三种额外配置互不叠加：

1. **旧秩**：边界仍为训练 P93，只改为固定提交的 `legacy=True`，即 `np.quantile(scores,min(1,k/n),method="higher")`。并列 Q、覆盖、W_raw/W_set/W、IS 和与主配置的偏差；不取代 `legacy=False`。
2. **全局 P93**：只改上界为原始 1030 行 `np.percentile(y_all,93,method="linear")`，L=0，仍主秩。仅量化假设性全局分位泄漏的偏差。
3. **加载器全局默认边界**：只改上界为 `1.15*np.percentile(y_all,99,method="linear")`，其余同主配置，量化该默认边界误用的偏差。

2/3 明确标为泄漏归因诊断，不是部署边界或有效确认性结果；全部在 A 主结果锁定后、独立 namespace 中计算，不能将全局 U 值传回主流水线。主试验继续禁止调用 `load_concrete()` 的边界逻辑。只读源码核验表明 `robust_concrete.py` 忽略加载器返回边界并从 y_train 算曲线，不能预设表 8/9 就是泄漏产物；这些列只检验可能归因，不推定原文有错。原口径 W_raw 始终并列保留，旧秩/全局列无论接近或远离原表，都不能翻转 A。

### 7.4 FRP 的完整生成式与固定接口

生成种子 `d=2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019`；每个调用 `load_frp(n=1500,seed=d)`。以下是固定提交生成器的数值定义，不引用其人员注释：同一个 `rs=np.random.RandomState(d)`，严格依次抽取长度 1500 的向量：

```python
Vf = rs.uniform(0.10, 0.48, 1500)
Ef = rs.uniform(69, 700, 1500)        # GPa
Em = rs.uniform(2, 10, 1500)          # GPa
radius = rs.uniform(3, 12, 1500)      # 微米；无关特征
MMA = rs.uniform(0, 5, 1500)          # 度
V = Vf*Ef + (1-Vf)*Em                # 算术混合上界
R0 = 1.0/(Vf/Ef + (1-Vf)/Em)         # 调和混合下界
position = np.clip(0.97-0.04*MMA, 0.05, 0.999)
E11 = R0 + position*(V-R0)
y = np.clip(E11 + rs.normal(0, 0.04*(V-R0)), R0, V)
X = np.column_stack([Vf, Ef, Em, radius, MMA])
```

噪声为独立零均值正态，逐点标准差 `0.04*(V-R0)`；原始目标只裁进原始 `[R0,V]`。定义任意特征点的 `R0(X)=1/(Vf/Ef+(1-Vf)/Em)`、`V(X)=Vf*Ef+(1-Vf)*Em`；实验边界 `L=R0(X)`、`U_ρ=L+ρ*(V(X)-L)`，`ρ∈{1.00,0.99,0.98,0.965,0.95,0.90}`。**收紧后的走廊不能再裁 y**，以免抹掉失配。

执行 FRP 前须通过第 9.1 节 T3：固定提交生成器与本段独立转录式的种子 2000 X/y 哈希逐一相同，具体字节格式和冻结字段见该节；不一致则 FRP 未交付，方程不修改，主对象不读该审计。本次没有生成数据或填入未经计算的摘要。

划分种子 `s=d-2000`，使用第 3.2 节两次调用；单调方向及模型参数遵循第 4 节，同底模 U/N/R，α=0.1、legacy=False。输出单位 GPa，不与 Concrete MPa 混合均值。ρ 和 d 不按测试 η 调整。η_test≥10% 的格仍保留，用退回和有限测试前沿解释，不能声称每个有限校准集必达 90%。

## 8. 未来执行顺序、拟合上限及停止规则

### 8.1 准备与完整任务清单

本次只修订文件。未来正式运行前先完成依赖安装/输入审计、按第 9.1 节 T3 在已锁定环境进行 FRP 种子 2000 生成物校验（不拟合或评价）、生成任务清单和判定函数，哈希冻结后才计时。预先规定的计时烟测只用人工数值输入（515×8、独立随机种子 300001、单调方向同 Concrete），拟合两个分位模型一次，只保存耗时/资源及可运行性，不读取科学测试标签、不生成覆盖/评分结果、不进入确认性表；该未来烟测不允许用于调参或选择环境。准备阶段安装和烟测若实际上放在正式命令内，则全部计入正式预算，不能超时后再开预算。烟测标签须为种子 300001 生成的合成有限数，不读取注册数据标签；两个模型拟合后丢弃，不进入正式预测缓存。缓存 namespace 固定为 `smoke`，正式任务固定为 `formal`，两类键不相交，禁止复用烟测模型。本次没有执行该烟测。

正常唯一拟合次数按**一次 estimator.fit 为一次**：

| 模型缓存族 | 最大唯一 fit 次数 |
|---|---:|
| A：6 划分×2 种单调性×2 分位 | 24 |
| B：20×2×2 | 80 |
| C：20×2×2 | 80 |
| D：6×1×2 | 12 |
| FRP：20×1×2 | 40 |
| 正式合计 | **236** |

每个缓存族只拟合一次，八个上界、U/N/R、校准重复、legacy/全局诊断不再 fit。相同配置的唯一一次技术重跑若须重建模型，也计 fit；最坏正式 fit 尝试上限为 472（每个唯一拟合最多两次，依赖恢复不得导致第三次），另有正式前人工烟测 2 次。清单逐 fit 写缓存键、用途和计数，不能以计时烟测复用正式结果。

按方法结果记录计数，计划最多 7344 个任务行：A 单调全网格 144＋非单调 U 48，B 960，C 960，D 18，FRP 360；B/C 有限校准子样本额外 4800；A 三种归因配置额外 54。全网格 P93 与 all 已含在原格中，不能重复计为新任务。容量不足仍保留计划行并标合法不可执行，不能减少清单。划分、底模缓存和方法任务各自计数，避免混用。

正式顺序：环境/输入审计 → A P93 及对象独立汇总/门审计/锁定判定 → B P93 及 B/H3 对象独立汇总/门审计/锁定判定 → C P93 及对象独立汇总/门审计/锁定判定 → D P93 → A/B/C 单调全网格（复用主 P93）→ B/C 非单调全网格 → 校准样本量 → A 非裁决归因列 → FRP → 轮次独立汇总与完整性核查。分支内种子升序，网格按第 7 节顺序；结果写原子文件，缓存只复用，不重试种子。

### 8.2 预算及严格技术重跑

**正式命令启动起墙钟硬限 6 小时（21600 秒），所有进程 user+system 累计 CPU≤24 核时（86400 秒）；任一先到即停。** 正式命令内安装、下载、审计、拟合、预测、bootstrap、独立汇总及技术重试全计入预算。人工写作不计，但运行开始后的暂停不能重置时钟，不能重新启动预算拼接本轮。

仅 CPU，最多 4 个工作进程，单进程 OpenMP/BLAS/MKL 线程各 1，峰值内存预算 8 GiB；独立监督进程用单调时钟，监控全部子进程和自身 CPU，至少每 5 秒检查。CPU 账本分列监督进程、拟合工作、其他计算/收尾进程的 user+system，硬限采用三者合计，已退出进程累计值保留，不重复计数；监督 CPU 计入 24 核时，4 个工作进程不能各自占满 6 小时。在墙钟达到 21580 秒或 CPU 达到 86320 秒首次检查时，对整组发 `SIGTERM`；最多等 5 秒，未退出即 `SIGKILL` 整组。正常目标在墙钟 21590 秒前结束，预留轮询与终止资源；不承诺不可中断内核任务一定按时释放。实际超过任一硬限就记预算违规/未完成，不能因已发终止信号而算完成。保留已原子提交输出，不使用半写文件。

只因预算、冻结数据/代码完整性失败、不可恢复资源/数值错误停止；不看结果是否有利。瞬时运输下载错误最多 2 次重试，不能换来源/数据。**同配置技术重跑最多 1 次**，仅限：有栈/退出码的进程崩溃（含资源故障）、监督进程记录的挂起，或底预测/底分数/有限指标计算出现不在第 4–5 节合法名单中的 NaN/±∞。挂起定义为工作进程连续 600 秒没有任务心跳且 CPU/磁盘进度监控也无变化；预算到限优先停止，不能用重跑续时。

技术重跑必须相同数据、划分、参数、种子、代码哈希和环境，保留原失败日志/时间/资源及恢复尝试；不能选择两份完整有效结果中较好的一份。**与原表不符、覆盖低、出带、空区间、底分位交叉、合法 Q=+∞、合法 W/IS=+∞、评分恶化或不可比较，全部不是技术错误，不准技术重跑。** 冻结对象断言 O1–O7 未通过时停止相关对象并交付 U_AUDIT；轮次断言 T1–T4 未通过记整轮未完成，不污染已通过的对象门；不以修补代码后同名重跑来替换本轮结果。需改实现则另记带时间戳偏差/新代码哈希，不能自动恢复本版本成功资格。

超时/缺任务报告“未完成”，附计划/完成/合法不可执行/失败计数、未完成种子及格子、墙钟/核时和触发原因。不得事后减少 20 划分、网格、重复或 FRP 数量来宣称完整。已通过对象门并原子锁定的 A/B/H3/C 仍保留唯一科学三态及原因码；预算到限/实际超限或后续诊断缺失只把整轮标为未完成，不改写对象为 U_AUDIT。若到限前对象本身未齐全或未完成门审计，则该对象 U_DELIVERY；不能在预算外补跑汇总来宣称已通过。整轮不完整不能宣称所有诊断交付。

## 9. 输出、穷尽的机械审计及公开材料

### 9.1 审计清单：对象门与轮次完整性

这两类清单在正式前连同检查代码、记录格式和哈希冻结，编号穷尽。对象门只有本对象所需任务的交付、O1–O7 和第 10 节环境门；轮次完整性 T1–T4 不进入 `judge`。对象审计须在本对象判定前完成并原子锁定，通过后禁止以出带、后续压力缺失或预算补设资格条件。错误须引用编号、时间和可复核证据。

**对象门（按 A/B/H3/C 各自输入范围投影，不要求无关诊断存在）：**

- **O1 输入及源码完整性**：冻结 Concrete CSV 哈希、1030/1005 行、8+1 列、有限数值/单位；使用配方映射的对象核对 427 组及映射。固定提交、执行代码/判定函数/补丁/对象清单哈希、配置及对象任务键均符合冻结值。FRP 文件/输出不是对象输入。
- **O2 划分及数量**：本对象注册种子完整，行分区互斥且并集完整；A/B 的行数量、C 的组互斥并集和 213/107/107 组符合第 3 节，C 亦查其 B 比较输入。每个分母/行 ID 与对象清单一致，分支内方法共用同一输入；不查非主压力子样本。
- **O3 秩及非有限来源**：k 必须由第 4.2 节整数式得到，finite_count、负分及主 legacy=False 状态正确；正无穷只来自边界外置分或秩不足；底预测/原分数不得 NaN 或非有限；逐行无限评分只能来自第 5.1 节合法状态，入口布尔标志、CI 跳过和 infinity_detail 与逐行来源一致。
- **O4 阈值次序**：同底模同校准 `Q_R>=Q_N-epsilon`；扩展实数显式比较，相同正无穷视相等，禁止无穷相减。
- **O5 包含与走廊**：N 非空时 R 非空，且 `lo_R<=lo_N+epsilon, hi_R>=hi_N-epsilon`；N 为空时集合包含为真。有限非空投影端点在 `[L-epsilon,U+epsilon]` 内；空求交的一端可在走廊外。无穷退回必须 `[L,U]`，宽度等于走廊宽度。
- **O6 独立逐行重算及入口契约**：独立程序只读本对象新逐样本表，重算 C、W_raw、W_set、主 W、IS、E、退回/η/原因计数、种子等权向量、有限 CI、无穷短路及判定树。入口的类型、长度、单位、NaN、非法 None/无穷、CI 顺序和状态关联在函数内部校验，不只依靠上游保证。存储 float64 端点及一次差值、empty/near_zero_empty、无限状态/原因精确一致；有限汇总绝对误差<=1e-8、相对误差 0，覆盖百分比/百分点核查<=1e-10，全部合法无限状态精确一致。四个无穷原因码的明细含所有种子的空行/近零计数。第 4 节手算例须通过，原表 JSON 不作期待值。
- **O7 本对象 P93 缓存同一性**：本对象主 P93 缓存键、模型引用和逐样本 SHA-256 一致，独立重算记录已提交。仅当主格及同配置 P93/all/网格重复格在对象锁定前均存在时比较二者；冲突触发本对象 U_AUDIT。尚未产生的重复格不触发失败、不延迟完整对象判定。后续才出现的冲突按 T2 处理。非主模型总 fit 数和预算不属于 O7。

对象缺交付为 U_DELIVERY；O1–O7 失败为 U_AUDIT；事前环境不匹配为 U_ENV，不把它们合并成科学性失败。三态与门检查快照、锁定时间/哈希不可覆盖。锁定后发现具体问题须追加偏差和影响说明，后续版本才能重新评价；不得在本版本取消不利结果或换另一份值。

**轮次完整性（只生成完成/未完成和轮次原因，不返回科学三态）：**

- **T1 全任务交付**：第 8.1 节 7344 个注册方法任务行、D、完整网格、非单调、有限校准重复、A 归因列和 FRP 均须有结果或仅容量不足的预注册 `not_applicable_capacity`。任一缺失/资源中断/技术失败令整轮未完成；本对象不需要的行不会触发 U_DELIVERY。
- **T2 全局缓存与诊断审计**：检查全网格 P93/all 复用、非主压力格输入/划分/秩/端点/独立重算、子样本嵌套与容量状态、归因 namespace、D、模型复用及 236 唯一 fit/472 最坏尝试上限。后续重复格冲突保留双方哈希及影响，记轮次审计失败/未完成，不改写已锁定对象。
- **T3 FRP 生成物一致性**：在同一事前锁定环境，用固定提交 `frp_experiment.py` 的 `load_frp(n=1500,seed=2000)` 与第 7.4 节独立转录式各生成一次未收紧、未划分的 `X,y`。只提取该函数，不调用模块主入口，不拟合模型。断言 X 为 (1500,5)、y 为 (1500,)，float64、有限、列/行次序一致。分别转为小端 float64、C 连续内存；`H_X=SHA256(X_bytes)`、`H_y=SHA256(y_bytes)`，其中 bytes 为 `np.ascontiguousarray(a,dtype="<f8").tobytes(order="C")`；同时登记 shape/dtype/特征顺序、种子、n、环境组及两实现源码哈希，不能改格式追求相同哈希。清单必填 `reference_X_sha256, reference_y_sha256, transcription_X_sha256, transcription_y_sha256, match`，两对哈希逐一精确相等才通过。正式前把固定提交参考摘要填入锁定清单并哈希冻结；正式使用种子 2000 的生成物再与该参考摘要核对，其他种子保留生成物哈希。参考不是转录式自行生成的摘要，不能用双方同调用同一实现伪装独立验证。本次只写文件，不生成 FRP，故不虚构尚未得到的 64 位摘要；上述必填字段在未来准备阶段填实，未填实也不算 T3 通过。不一致/缺失令 FRP 未交付、整轮未完成；第 7.4 节诊断方程保持冻结，不能修改方程或环境后静默重获资格。A/B/H3/C 不读该审计或 FRP 输出。
- **T4 预算与原子记录**：监控、阶段/监督/拟合/其他 CPU 账本、原子提交/未完成记录及重跑证据齐全；实际墙钟<=21600 秒且全部进程 user+system<=86400 秒。到限停止且尚有任务即未完成；任何实际超限即预算违规/未完成，即使所有科学对象已完成也一样。不会据此返回 U_AUDIT 或覆盖已通过对象门的 A/B/H3/C。

整轮仅在 T1–T4 全通过时为“完成”。例：A 的六种子及对象门已通过后 FRP 缺失或收尾越过硬限，A 保留原科学终态，整轮为“未完成”；P93 压力格还未产生不触发 O7；锁定前已有主/重复格哈希冲突则该对象唯一返回 U_AUDIT。

### 9.2 必须交付的输出

保存冻结稿/代码/任务/环境哈希、划分与匿名组映射、模型缓存清单、逐样本表、逐任务指标、种子/分支汇总、技术日志与未完成账本、资源监控、环境及补丁说明。逐样本表至少含 `endpoint_delta, near_zero_empty` 和 float64 端点无损存储信息（保留可精确 round-trip 的表示），以及分支、种子、底模、P/ρ、边界来源、秩规则、校准 n/r/ID 哈希、原行号/匿名组、y、L/U、底分位、Q 状态、端点、命中、主宽度 w、原始端点差、集合宽度及走廊宽度字段、IS 状态、空区间/退回/各根因标志。指标表给分母和有限/无穷/缺失种子数；H3 不能只报非空子集。`infinity_detail` 按方法 N/R 分别给 `W_infinite_seed_ids/count, IS_infinite_seed_ids/count`，并在各方法的 `rows` 字段中按种子升序列 `seed, test_n, empty_rows, near_zero_empty_rows`（0 也列）；四个无穷原因码旁直接显示这些数和明细引用，不能只在附录说“有空区间”。另外交付对象门快照/锁定时间与轮次 T1–T4 账本、FRP 的两实现种子 2000 哈希审计及逐种子生成物哈希、分项 CPU 账本；资源汇总不含身份标识。

结果保存工作目录内独立子目录，不覆盖上游证据或原结果。原表 `out/concrete_sweep_rows.json` 等不得进入新指标、预测或独立重算程序；参考数字只进入隔离的冻结门槛/最后对照。主边界不读取全局、校准或测试标签；第 7.3 节已注册泄漏归因列是唯一例外，输出隔离，主流水线不调用加载器默认边界。

禁止为追数删行/种子/空区间、换 P93、改主秩、截负分、调参、轮换环境，禁止把 FRP 裁剪机制当独立实测证据、把组内龄期当独立材料、把零端点违规当覆盖可靠性证明。不得推送、发布、向他人发送消息或在工作目录外落盘。

公开稿、回复及结果只含论文题名、DOI、匿名分支/编号、必要无账户标识的来源链接、标准算法/评分术语和无身份标识的环境规格；不得出现受限内部标识、任何自然人姓名、机器名或用户名，URL/代码/路径/日志字段也必须检查。本稿统一采用区间评分 IS 和 R0/V 边界符号，不列姓名式术语或人员归属，具体公式不变。含账户标识的原来源 URL 只进私有清单，不按 v1.0 的“来源 URL 原样保留”规则带入公开稿。第三方全文、原源码、提交元数据、环境全集、原日志和整个证据目录不得整体公开打包。

## 10. 环境锁定与各对象资格门

论文披露 Python 3.12.3、scikit-learn 1.8.0、NumPy 2.4.4、SciPy 1.17.1、pandas 3.0.2、matplotlib 3.10.8、NGBoost 0.5.11、SHAP 0.52.0。此处不宣称当前环境已匹配或跑通，上游宽松 requirements 不是锁定文件。

正式前仅能选定**一组明确到补丁号的 Python/NumPy/scikit-learn/pandas/SciPy 版本**，写入环境锁和任务清单并 SHA-256 冻结。默认目标分别为 3.12.3/2.4.4/1.8.0/3.0.2/1.17.1；若依赖不能满足，须在看到结果前登记实际五元组与理由，不能只写“兼容环境”。凡调用 `judge` 的 A/B/H3/C 必须来自同一实际五元组，C 的 B 比较输入也一样；D、压力和 FRP 同组，禁止中途换版本。未登记锁或实际不等于该锁时，对象 U_ENV；若安装/运行因此无所需结果，先依交付优先级 U_DELIVERY。

**A 的额外精确环境门**要求实际五元组匹配论文上述值；缺少或不匹配只返回“不确定，U_ENV”，不得称数字复现成功或失败。B/H3/C 在完整匹配事前锁且对象门通过时可以按科学树判定；若该锁不同于论文，所有树终态附 `E_LOCKED_NONPAPER`，成功句固定附“仅在事前锁定的所报补丁版本组内成立；与论文环境不同（E_LOCKED_NONPAPER）”。B/C 成功与 A 的 U_ENV 分列，不能借前者推翻 A。环境限制码是解释限定，不是第四态或另一套可择优门；默认论文组匹配时不附该码。

正式计算最低直接依赖 NumPy、pandas、scikit-learn，SciPy 作为依赖也须锁定；NGBoost/SHAP 未使用即可记未安装。仅制图的 matplotlib 不改变数值门，使用时披露版本。第 4.1 节对上游早停的静态对齐说明只适用于论文披露组，非论文组须按本协议固定参数执行并披露环境限制，不能声称已证实预测相同。

执行前记录实际依赖、安装渠道、锁文件/哈希、发行版/架构、CPU 型号及可用核心数、RAM 上限、BLAS/OpenMP 与线程、完整模型参数、随机数生成器与种子、计时工具、源码/补丁哈希和相对路径启动命令。执行后记录开始/结束北京时间和 UTC、阶段耗时、峰值与累计资源。公开信息不含机器名、用户名、绝对用户目录、凭证或环境变量全集；依赖阻塞不等同科学性失败。

## 11. DEVIATIONS 与版本约束

本稿优先于候选文稿的实验细节，候选推荐/排序不变。以下偏差和细化已在任何正式测试结果出现前登记：

1. 本题独立预算为 6 小时墙钟且≤24 核时，不采用候选总述的 24 小时墙钟；正式开始后不可重置。
2. A 原始 1030 行；B/C/D 先按 9 列完整重复去重为 1005；D 是唯一桥接臂，不再加另一路桥接；各臂目的与样本差异公开。
3. C 不复制上游 60/20/20、0–9 种子、非单调配置的表 11；改为与 B 相同数据与 100–119、50/25/25 组比例，组键/排序/舍入均独立定义。
4. A 原始数字门的宽度由 v1.0 的集合宽度改为源码原口径 W_raw，三项数值带不变；主效率宽度与 区间评分 IS 对空区间共同记正无穷，几何 W_set 并列。所有修改发生在正式实验前。
5. B 取消原表水平的六带检验，改为名义附近 `[87,93]` 及覆盖增益/主宽度差正方向；原表数值复现只由 A 决定。原 B ±20%/绝对宽度带已撤销，不能执行时再启用。
6. 主秩始终 legacy=False；A 额外列 legacy=True、全局 P93 和加载器默认全局边界，只诊断不裁决。原入口已有训练分位曲线，不预断其泄漏。
7. 明确 early_stopping=False、数值数组 categorical_features=None 及全部影响拟合的参数；该小样本设置与锁定版本的上游 auto 关闭行为对齐，不扩大模型或调参。
8. 加同底模单调 U，A 原非单调 U 保留；FRP 用生成器的数据/单调方向和本协议未归一化 CQR，同底模诊断，不运行带额外模型的完整生成器主入口。
9. 无穷短路、交付与不可比较分开；断言清单和判定函数执行前哈希，技术重跑只在已列故障中同配置重试一次，性能差不算技术错误。
10. 本次只写 v1.2 与逐条回复/状态文件，没有正式计算、烟测、FRP 生成或发布。v1.2 进一步把对象门与轮次完整性拆分，入口封闭契约外值，秩采用整数上取整，各对象共用事前冻结补丁版本组，空区间差值与无穷明细和 FRP 哈希审计按第 4/6/9/10 节执行。执行后的新偏差另开时间戳条目，列预注册值、实际值、原因、是否已看结果及对解释的影响；不得覆盖已冻结稿或以偏差静默重获成功资格。

## 12. 执行后报告模板

结论首句按“任务完整性、A、B、区间评分 IS、C”的顺序展开，并使用第 6 节生成的三态及原因码。首表同时列 W_raw/W_set/主 W/IS 的数值或合法状态、原参考值、未取整等权估计与展示值、95% CI/跳过原因、容差、E/退回次数和环境资格。原数字复现与新划分稳定性分列；组合结论若使用，严格按第 6.5 节并原样附 A/B 全部原因码；`F_NONPOS_B_WIDTH` 只写“宽度正代价未成立”，H3 另成句。B/C（及 H3）非论文环境的成功句必须附 E_LOCKED_NONPAPER 和第 10 节限制句，A 的 U_ENV 单列。四个无穷原因码旁列 N/R 无穷种子数、每种子空行数/近零空行数，不改变正无穷惩罚。

次表为 D 桥接、A 旧秩/全局边界归因，之后完整压力格、逐种子/配方摘要、环境差异、错误及未完成/预算账本。已交付但不可比较的 H3 显示“不确定（不可比较）”，不能填成未交付；未完成范围明确列出，不填不存在的数字。首句的任务完整性用 T1–T4 判定；明确预算/压力/FRP 引发的整轮“未完成”不覆盖对象门已锁定的科学三态，分别列对象门和轮次审计编号。CPU 账本分列监督、拟合与其他 CPU 后再合计。区分经验边界可信度、覆盖恢复、宽度代价、评分效率与跨配方变化，不把复核成功扩张成全部材料可靠性的保证。

## 13. 修订记录（v1.0 → v1.1）

2026-10-14（北京时间），正式实验前，根据首轮两份审稿修订；逐条理由和位置见 `response_R13_rev1.md`。

- 第 2/3/6 节：A 独占原数字复现；A 六项只读取完整六种子等权均值，宽度按源码 W_raw；B 改为三项机制稳定性门，新增去重＋原种子桥接 D。
- 第 3/4 节：写死两次行划分调用及 ceil 数量（其切片措辞由 v1.2 更正为返回值第一数组训练/校准、置换前缀测试）；配方 np.round 最近偶数舍入、Python 元组排序、原地 shuffle 和 427 硬门；固定独立双分位、early_stopping=False、全部参数及手算例。
- 第 5/6 节：空区间主宽度/IS 同为正无穷；无穷 bootstrap 前短路；无穷变有限不判改善；不可比较映射不确定但属于已交付；固定三态优先级、原因码与结论次序，比例增幅只描述。
- 第 7 节：P93/all/全网格缓存必须同一；固定 legacy/两种全局边界诊断与 FRP 完整生成式；压力不入主判定、不报逐格 p 值。
- 第 8–10 节：236 唯一 fit/472 最坏正式尝试；登记未来人工烟测而本次不执行；5 秒终止宽限；穷尽机械清单、严格技术重跑及 A 精确环境资格门。
- 第 9/11/12 节：公开材料去除账户/身份链接及标识，保留 DOI、提交指纹与匿名来源定位；补齐偏差声明和完整报告要求。v1.0 原文件保留，不覆盖。

## 14. 修订记录（v1.1 → v1.2）

2026-10-14（北京时间），正式实验前，只依据唯一有效第二审 `grok_review2.md` 修订；逐条决定、理由和位置见 `response_R13_rev2.md`。v1.1 和首轮回复保留。本次未运行训练、烟测、FRP 生成或正式实验。

- R-01：第 6.1/7.1/8.1–8.2/9.1/12 节拆分 O1–O7 对象门与 T1–T4 轮次完整性；对象及时独立汇总并锁定；缺压力格不触发缓存冲突，锁定前双缓存冲突才可 U_AUDIT；预算超限保持整轮未完成，不改写已通过门的 A/B/H3/C。
- R-02：第 6.1/9.1 节入口内部验证类型、字段、长度、单位、NaN、非法 None/无穷、CI 和状态一致性；未知对象/解析异常也收成唯一 U_DELIVERY/U_AUDIT，A 不再用 zip 截断。
- R-03：第 3.2/9.1 节明确必须调用锁定 train_test_split，测试为置换前缀，返回第一数组才为训练/校准，第二次保持 tmp 返回顺序。
- R-04：第 4.2/9.1 节主秩改为 k=(9*(n_cal+1)+9)//10，保留原手算例并补整数点核查，不改有限分数不足时的无穷规则。
- R-05：第 6.1/10/12 节凡入 judge 的对象共用同一冻结补丁版本组；A 仍有论文精确组额外门，非论文组 B/H3/C 附 E_LOCKED_NONPAPER 并限定成功句。
- R-06：第 4.2/5.1/6.4/9.1–9.2/12 节存储 float64 端点仅相减一次并共用差值；四个无穷原因码旁列每方法无穷种子数、全部种子的空行/近零空行数，正无穷和空 N 修复规则不变。
- R-07：第 7.4/8.1/9.1/9.2 节登记固定提交与独立转录式的 FRP 种子 2000 X/y 生成物哈希精确核对、字节格式和必填清单；未来准备阶段填实冻结摘要，不一致则 FRP 未交付，主判定不读它。
- R-08–R-10：第 6.5/8.1–8.2/9.2/12 节固定宽度码报告句、组合附 A/B 原因码、监督/拟合/其他 CPU 分账；实现规格用 ASCII -1 与 1e-8，烟测用合成标签、拟合后丢弃、smoke/formal 缓存键不相交。
