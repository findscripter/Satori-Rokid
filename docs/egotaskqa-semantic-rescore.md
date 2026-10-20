# EgoTaskQA 任务问答语义重评分

日期：2026年10月20日。这是对七模型主实验中 EgoTaskQA 200 题的补充分析，**正式分数仍是字面匹配分**。没有重新调用被评测模型，也没有改动原始回答、标准答案或已有分数。本分析基于10月18日的原始回答，没有纳入10月20日补跑得到的新答案（见[实验结果](experiments.md)的补跑说明）。

## 为什么要补这一项

EgoTaskQA 的查询题标准答案使用数据集内部的标签词：对象带实例编号或连字符（`watermelon1`、`bottle-water`、`juicer-lid`），属性用专门名称（`poweredness`、`openess`、`wateredness`），动作写成固定格式（`Turn off TV with remote`、`Put bowl to microwave`）。主实验要求模型用简短英文自由作答，再统一大小写、标点、英文冠词和空白后逐字比较（字面匹配），因此“water bottle”“turned off the TV using the remote control”这类意思正确的答案也会被判错。

## 方法

1. **确定需要判断的答案。** 五个有合格答案的模型共1000个“模型—题目”结果：
   - 字面已判对、接口或格式失败、弃答（模型按要求返回空答案），三类沿用原结果。
   - 标准答案为 yes/no 且模型也只回答了相反的 yes/no 的，直接判错。
   - 其余485个答案按“题目＋归一化答案”去重为393项，交给语义判分。
2. **盲化判分。** 判分代理只看到问题、标准答案和候选答案，看不到模型名称，也不看视频；只判断两个答案的含义是否相同，不判断答案在画面上是否成立。
3. **三票独立判分。** 三个独立的 Claude Haiku 5.5 判分副本，各自使用不同的题目顺序和分批，每项给出 equivalent（语义等价）、partial（部分正确）或 wrong（错误），取多数票。
4. **阴性对照。** 混入40个故意配错的答案：把另一道同语义类型题目的真实模型答案配给当前题目，预期全部判错。
5. **对抗复核。** 怀疑者复核全部82个判对或部分对的项，专门找过宽的判定；辩护者复核全部311个判错项，专门找过严的判定。原计划由主控 AI（主会话中的 Claude Opus 5.5）按细则裁决有分歧和标签不一致的项，并随机抽查40个判错项。由于主控本身也是被评模型，它的裁决不计入主结果，只作为对照（见下文）。
6. **独立复算。** 用另一段代码，从原始回答和原始投票重新计算全部分数，结果一致。

判分细则、判分项和分批在判分开始前冻结。判分开始后有三处修订，都有书面记录：一是第一次判分运行因模型服务的名称问题没有产生任何结果，改用同一模型在服务商处的另一个名称重跑；二是在看到三票结果之后，追加了辩护者对全部311个判错项的复核（原计划只由主控抽查40项）；三是发现主控与被评模型之一相同后，主结果改为只用三票多数，主控的裁决只作对照。复核代理与判分代理同为 Haiku 5.5。

## 结果（每个模型分母200，失败和弃答计错）

| 模型 | 字面匹配（10月18日回答） | 语义等价 | 含部分正确 |
|---|---:|---:|---:|
| Claude Opus 5.5 | 22.50%（45） | **36.50%（73）** | 39.50%（79） |
| GPT-6-astra | 19.50%（39） | 31.00%（62） | 33.00%（66） |
| GPT-6.1-sol | 17.50%（35） | 27.50%（55） | 29.50%（59） |
| Mage-VL | 2.00%（4） | 4.00%（8） | 7.50%（15） |
| VideoChat3-4B | 4.00%（8） | 6.00%（12） | 8.00%（16） |

上表只用三个判分代理的多数票。主控 AI 的13项裁决因利益冲突不计入：主控与被评模型之一同为 Claude Opus 5.5，其中4项“语义等价→部分正确”的降级都落在 GPT 的答案上，Claude 的答案没有被降级。若计入裁决，语义等价为 Claude 73、GPT-6-astra 59、GPT-6.1-sol 53、Mage-VL 8、VideoChat3-4B 12（含部分正确为80、68、62、16、17），排名不变。

GPT-5.6-sol 在10月18日200题全部接口失败，Dispider 没有合格输出，两者都没有可判断的答案。10月20日补跑得到的新答案没有做语义判分，涉及 GPT-6-astra 5题、GPT-6.1-sol 17题和 GPT-5.6-sol 200题，所以本页 GPT-6.1-sol 的字面匹配仍是补跑前的17.50%（主表补跑后为21.00%）。所有题固定回答“no”的基线为18%，不受语义判分影响。

**提升全部来自查询题。** 132道查询题中，Claude 从2题升到30题，GPT-6-astra 从3题升到26题，GPT-6.1-sol 从3题升到23题；68道是非题的结果不变。

**按题目类型拆分**（三个数字依次为：字面匹配 / 语义等价 / 题数。类型取自数据集标注：多人互动、意图是题目范围，反事实、解释是推理类型。多人互动只计单独标为多人互动的13题，另有30题同时标为环境与多人互动；反事实只计单独标为反事实的40题，不含同时标为预测的8题）：

| 题目类型 | Claude Opus 5.5 | GPT-6-astra | GPT-6.1-sol |
|---|---|---|---|
| 多人互动 | 0 / 0 / 13 | 0 / 0 / 13 | 0 / 0 / 13 |
| 反事实 | 11 / 12 / 40 | 9 / 9 / 40 | 9 / 10 / 40 |
| 意图 | 1 / 8 / 22 | 2 / 9 / 22 | 2 / 7 / 22 |
| 解释 | 6 / 17 / 45 | 8 / 19 / 45 | 7 / 15 / 45 |

意图和解释类题目原来的低分主要来自用词不匹配；多人互动和反事实推理按意思评分后几乎没有提升，是当前真正的短板。

## 判分可靠性

| 检查 | 结果 |
|---|---|
| 三票完全一致 | 393个真实项中92.4%，Fleiss κ = 0.855（连同40个对照共433项时为93.1%，κ = 0.858）；393项都有多数票 |
| 40个阴性对照 | 多数票0个判对或部分对；单票也没有 |
| 对抗复核分歧 | 393项中7项 |
| 主控 AI 裁决（不计入主结果） | 13项：4项从语义等价降为部分正确，1项从部分正确降为错误，7项从错误升为部分正确，另有1项维持部分正确 |
| 主控随机抽查判错项（不计入主结果） | 40项中38项确认错误，2项应为部分正确 |

## 局限

- 这是看到结果后追加的补充分析，细则中的对象和属性同义词示例有一部分来自对已有答案的阅读。
- 判分和复核都由 AI 完成，只比较文字、不看视频；全程没有人工标注。
- 主控 AI 与被评模型之一同为 Claude Opus 5.5，所以它的裁决不计入主结果。判分和复核代理也都是 Anthropic 的 Haiku 5.5，与 Claude 同属一家；盲评只隐藏了模型名，答案的文风仍可能透露来源。
- 阴性对照多为明显不相关的答案，比真实的近似错误更容易识别，假阳性率可能被低估。
- “部分正确”的边界有主观性，所以主结论使用“语义等价”一列。
- 输入仍是每题4张稀疏帧；题目是重新抽样的，原论文对这些题的评分方式本项目没有核实，所以本结果不能与原论文的分数直接比较。

原始判分记录、批次文件和计分脚本保存在本地实验工程中；其中含有数据集题目原文，按本仓库不分发数据集的约定未上传。

## 判分细则（原文，原样发给判分代理）

```text
You grade answers to questions about egocentric (first-person) household videos from the EgoTaskQA dataset. Each item has a question, a reference (the dataset's gold answer) and a candidate (a model's answer). Decide whether the candidate gives the same answer as the reference.

You do not see the video. Do not judge whether the candidate is visually plausible and do not guess what happened in the video; judge only whether the candidate states the same answer as the reference.

How to read reference answers (dataset annotation vocabulary):
1. Objects. A trailing number marks an instance (watermelon1, cup2, closet1); hyphenated labels are compound names (bottle-water = water bottle, juicer-lid = the juicer's lid, knife-base = knife block/holder, game-console = game console). A candidate naming the same kind of object matches; instance numbers are ignored. Common names for the same physical object match, e.g. juicer = blender / blender cup / juicer cup; closet = cabinet / cupboard; tank = fish tank / fishbowl; tv = television; remote = remote control; vacuum = vacuum cleaner; wrapping = wrapper / packaging. A different object (e.g. 'plate' for 'bowl', 'cutting board' for 'lettuce') does not match.
2. Attributes. poweredness = power on/off; openess (sic) = open/closed; wateredness = watered / wet / moisture; cleanliness = clean/dirty; emptiness = empty/full/filled; spatial relationships = location / position / what it is on, in or attached to / being moved; shape = whole vs cut/sliced/broken; edibility = cooked/raw/edible; cuttability = whether it can be cut; state of mixture or mixing = mixed/blended/stirred; visibility to me = whether the camera wearer can see it. When the question asks which attribute or status changed or is wanted, the candidate matches if it names that attribute, a synonym, or a concrete change that clearly belongs to that attribute (e.g. 'turned off' for poweredness, 'moved to the table' for spatial relationships). Naming a different attribute does not match.
3. Locations and states. 'on top of X' = on X; 'in X' = inside X; 'attached to X' = held by, stuck on or skewered on X; 'attached to me' or 'in me' = held by the camera wearer; 'attached to the other person' or 'in the other person' = held by the other person. Plain states ('on', 'off', 'clean', 'cooked', 'open', 'closed', ...) match their synonyms (e.g. 'fried' implies 'cooked').
4. Actions (e.g. 'Put bowl to microwave', 'Get kettle from stove', 'Turn off TV with remote', 'Point to cup'). The candidate matches if it describes the same action: the same kind of verb and the same main object. It may omit or paraphrase the source, destination or tool as long as it does not contradict them. 'Get X from Y' = take/pick up X from Y; 'Put X to Y' = put/place X on or in Y; 'Point to X' = point at X.
5. yes/no references. The candidate must have the same polarity; a short justification is fine.

Labels:
- equivalent: the candidate states the reference answer (paraphrase, synonym, or extra detail consistent with it).
- partial: related but not the same answer. It lists several alternatives of which one matches; or it matches only part of a multi-part reference (e.g. reference 'Put cup and spoon to table', candidate mentions only the cup); or it is vaguer and merely compatible with the reference.
- wrong: a different answer, a contradiction, an unrelated, empty or refusal answer ('none', 'unknown'), or a different object, attribute or action.
If the candidate mentions the reference but also asserts something that contradicts it, choose wrong. When torn between two labels, choose the less favorable one.
```
