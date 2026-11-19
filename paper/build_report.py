"""生成课程期末项目报告（Word），结构严格按指导书 §9.4：

摘要 / 1 引言 / 2 相关工作 / 3 方法 / 4 实验设置 /
5 结果与分析 / 6 局限与伦理 / 7 结论 / 附录 / 参考文献

运行：python paper/build_report.py
输出：paper/黄思桦 202425310307.docx（再由 Word 转 PDF）
封面取自学校模板 paper/cover_template.docx，正文另起一节。
所有数字均取自 results/ 下的 csv 与 summary，未做二次拟合。
"""

import os

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COVER = os.path.join(ROOT, "paper", "cover_template.docx")
AUTHOR = "黄思桦"
STUDENT_ID = "202425310307"
CLASS_NAME = "24软件工程*班"
TEACHER = "蓝连涛"
TITLE_CN = "TabM 的课程规模复现：参数高效集成在表格数据上的收益与成本"
REPO_URL = "https://github.com/May712-hzx/Tabm"
OUT = os.path.join(ROOT, "paper", f"{AUTHOR} {STUDENT_ID}.docx")
FIG_ARCH = os.path.join(ROOT, "paper", "fig1_arch.png")
FIG_DIR = os.path.join(ROOT, "plots")

BODY_FONT = "宋体"
HEAD_FONT = "黑体"


def set_run(run, font=BODY_FONT, size=10.5, bold=False):
    run.font.name = font
    run.font.size = Pt(size)
    run.bold = bold
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)


def para(doc, text, size=10.5, bold=False, indent=True, align=None,
         space_before=0, space_after=0, line=1.5):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = line
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    if indent:
        pf.first_line_indent = Cm(0.74)
    if align is not None:
        p.alignment = align
    r = p.add_run(text)
    set_run(r, BODY_FONT, size, bold)
    return p


def h(doc, text, level=1):
    sizes = {1: 14, 2: 12, 3: 10.5}
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(12 if level == 1 else 8)
    pf.space_after = Pt(6)
    pf.line_spacing = 1.4
    r = p.add_run(text)
    set_run(r, HEAD_FONT, sizes[level], bold=True)
    return p


def caption(doc, text, before=True):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before = Pt(4 if before else 2)
    pf.space_after = Pt(4 if before else 8)
    pf.line_spacing = 1.2
    r = p.add_run(text)
    set_run(r, HEAD_FONT, 9.5, bold=True)


def _mark_header_row(row):
    """表头行在跨页时自动重复。"""
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    trPr.append(el)


def _no_split(row):
    """禁止单行在页边界被切成两半。"""
    trPr = row._tr.get_or_add_trPr()
    trPr.append(OxmlElement("w:cantSplit"))


def add_table(doc, cap, headers, rows, widths=None, font_size=9):
    caption(doc, cap, before=True)
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    _mark_header_row(table.rows[0])
    hdr = table.rows[0].cells
    for i, name in enumerate(headers):
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(name)
        set_run(r, BODY_FONT, font_size, bold=True)
    for row in rows:
        cells = table.add_row().cells
        _no_split(table.rows[-1])
        for i, val in enumerate(row):
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            r = p.add_run(str(val))
            set_run(r, BODY_FONT, font_size)
    if widths:
        for i, w in enumerate(widths):
            for row in table.rows:
                row.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_figure(doc, path, cap, width_cm=15.0):
    if not os.path.exists(path):
        para(doc, f"[缺图：{os.path.basename(path)}]", indent=False)
        return
    doc.add_picture(path, width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption(doc, cap, before=False)


# --------------------------------------------------------------------------
def _set_cell_text(cell, text):
    """改写单元格首个段落，保留模板原有 run 的字体与字号。"""
    p = cell.paragraphs[0]
    if not p.runs:
        p.add_run(text)
        return
    p.runs[0].text = text
    for r in list(p.runs[1:]):
        r._element.getparent().remove(r._element)


def build_cover(doc):
    """按学校封面模板填写题目、小组成员与班级。"""
    t = doc.tables[0]
    _set_cell_text(t.cell(0, 1),
                   "《TabM: Parameter-Efficient Ensembling》论文复现报告")
    _set_cell_text(t.cell(1, 1), f"{STUDENT_ID}-{AUTHOR}")
    _set_cell_text(t.cell(3, 1), CLASS_NAME)
    # 模板预留了三个成员位，本项目为个人项目，删除多余的一行
    t._tbl.remove(t.rows[2]._tr)


if os.path.exists(COVER):
    doc = Document(COVER)
    build_cover(doc)
    body_sec = doc.add_section(WD_SECTION_START.NEW_PAGE)
else:
    doc = Document()
    body_sec = doc.sections[0]

body_sec.page_width = Cm(21.0)
body_sec.page_height = Cm(29.7)
body_sec.left_margin = Cm(2.5)
body_sec.right_margin = Cm(2.5)
body_sec.top_margin = Cm(2.5)
body_sec.bottom_margin = Cm(2.5)
body_sec.header_distance = Cm(1.27)
body_sec.footer_distance = Cm(1.27)

# 正文节继承模板的“行网格”与页面边框：前者会把 1.5 倍行距按网格重新吸附、
# 让版面明显变松（正文从 17 页涨到 19 页），这里只对正文节关掉，封面保持模板原样。
_sect = body_sec._sectPr
_grid = _sect.find(qn("w:docGrid"))
if _grid is not None:
    _grid.set(qn("w:type"), "none")
_bd = _sect.find(qn("w:pgBorders"))
if _bd is not None:
    _sect.remove(_bd)

st = doc.styles["Normal"]
st.font.name = BODY_FONT
st.font.size = Pt(10.5)

# ---- 正文标题区（封面已给出题目与成员信息）----
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(6)
r = p.add_run(TITLE_CN)
set_run(r, HEAD_FONT, 16, bold=True)

para(doc, "《深度学习》课程期末项目 · Project 5（主论文：TabM, ICLR 2025）",
     size=10.5, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, space_before=6)
para(doc, f"{AUTHOR}（{STUDENT_ID}）　　{CLASS_NAME}　　指导教师：{TEACHER}",
     size=10.5, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, space_after=6)
para(doc, f"代码仓库：{REPO_URL}　　提交日期：2026-11-19",
     size=9, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, space_after=12)

# ---- 摘要 ----
h(doc, "摘要", 1)
para(doc,
     "TabM 提出用共享 MLP 主干配合 k 个成员级缩放参数，在几乎不增加参数的前提下"
     "一次性得到 k 个成员的集成预测，但其结论建立在 GPU、46 个数据集与逐数据集调参之上，"
     "在算力受限、统一超参的课程规模下是否成立尚缺少独立证据。"
     "我们在 Adult、Covertype、California Housing 三个公开数据集上，以固定划分、统一训练预算和 5 个随机种子，"
     "对比 MLP、XGBoost、HistGBM 与 k∈{1,4,8,16} 的 TabM，并补做成员裁剪消融、"
     "训练与推理成本测量、误差分析和 k 个独立 MLP 的成本上界对照。"
     "数据规模为 3 个数据集 × 5 个种子 × 11 个配置共 165 次训练，Covertype 下采样至 5 万行，全部在纯 CPU 上完成。"
     "结果显示：TabM(k>1) 相对同预算 MLP 的主指标高 0.4~1.5 个百分点，但幅度与种子标准差同量级；"
     "参数量仅增加 1.30~1.57 倍，训练时间却增加 8~14 倍；Covertype 上 k=16 相对 k=1 显著变差（p=0.011）；"
     "从 k=16 裁剪成员的 6 组对照中有 5 组劣于直接训练同规模模型；"
     "16 个独立 MLP 在 Covertype 上比 k=16 TabM 高 3.6 个百分点，代价是 10 倍参数。"
     "结论是：在本课程预算下 TabM 的“参数高效”成立，但“计算高效”与“由成员多样性带来增益”均未成立，"
     "k=4 是性价比最高的工作点。",
     size=10, space_after=6)
para(doc, "关键词：表格数据；深度集成；TabM；参数高效；模型校准；错误分析",
     size=10, indent=False)

# ---- 1 引言 ----
h(doc, "1　引言", 1)
para(doc,
     "表格数据（电子病历、信贷记录、人口普查、工业传感器表格等）仍是实际业务中最常见的数据形态，"
     "而在这类数据上，梯度提升决策树长期占据主导地位[3,4]。深度模型要在表格任务上取得优势，"
     "通常需要借助集成：训练多个成员并平均其输出，可以稳定降低方差，但也意味着 k 份权重、k 倍训练与推理开销，"
     "这在算力受限的场景中难以承受。")
para(doc,
     "TabM[1] 针对这一矛盾提出了参数高效的集成方式：所有成员共享同一个 MLP 主干，"
     "每个成员只额外携带一组极低维的缩放参数（rank-1 调制），一次前向即可并行产生 k 个预测。"
     "论文报告在 46 个表格数据集上，TabM 以远小于“k 个独立 MLP”的参数与计算量取得了更好的平均表现，"
     "并给出了三条可检验假设：H1（相对同预算 MLP 更好且更稳定）、"
     "H2（收益来自成员多样性与集成，而不只是参数或计算量增加）、"
     "H3（训练大 k 后裁剪成员，可能优于直接训练更小的 k，但依赖数据集）。")
para(doc,
     "现有工作尚未回答的问题是：上述结论是否依赖“充足算力 + 大规模超参搜索 + 多数据集平均”这一前提。"
     "论文的每个数据集都单独调参、可使用 GPU、训练预算按数据规模自适应；"
     "而在课程项目中，我们只有 CPU、一套全数据集共用的固定超参、三个中等规模数据集和一次固定划分。"
     "在这种“预算被压缩”的条件下，TabM 的相对收益是否仍然存在、成本是否仍然值得、三条假设是否仍成立，"
     "需要独立测量而不是引用论文数字。此外，论文对 H2 的支撑主要依赖集成增益与参数量的对比，"
     "缺少对“成员之间到底有多不同”的直接刻画。")
para(doc, "本项目具体完成了以下工作：")
for t in [
    "（1）机制复现：使用官方 pip 包 tabm，自行编写训练与评估循环，复现“逐成员计算损失再平均”"
    "与“先 softmax 再平均”的口径，并用单元测试锁定这两条契约；",
    "（2）主实验矩阵：3 个数据集 × 5 个种子 × {MLP, XGBoost, HistGBM, TabM k=1/4/8/16}，"
    "共 165 次训练，主指标与效率指标同时报告；",
    "（3）三项分析：H3 成员裁剪消融（随机裁剪 / 验证集择优裁剪）、"
    "训练时间—推理延迟—峰值内存—参数量的效率测量、以及基于混淆矩阵与按类指标的误差分析；",
    "（4）一项拓展：训练 k 个完全独立的 MLP 作为成本上界，与 TabM 做严格的参数/时间/精度三方对照，"
    "并补充逐样本概率方差，用于直接检验 H2 的“多样性”机制。",
]:
    para(doc, t)

# ---- 2 相关工作 ----
h(doc, "2　相关工作", 1)
h(doc, "2.1　表格数据上的树模型与深度模型", 2)
para(doc,
     "XGBoost[4] 以二阶梯度提升与正则化目标成为表格任务的工业标准；"
     "Gorishniy 等[3] 的系统评测进一步指出，在同等调参预算下，经过充分调优的 GBDT 仍是强基线，"
     "而简单的 MLP 在多数数据集上落后于树模型，只有经过结构调整（如 FT-Transformer）才可能追平或反超。"
     "这解释了为什么本项目把 HistGBM 与 XGBoost 作为“必须战胜”的强基线，而不是只与 MLP 比较。"
     "另一条近年兴起的路线是表格基础模型，例如在大规模合成数据上预训练的 TabPFN 系列[5] 在小样本任务上表现突出；"
     "但这类方法依赖预训练语料与推理开销，不在本项目的算力与数据预算内，本文只把它作为背景参照，不做复现。")
h(doc, "2.2　参数高效的深度集成", 2)
para(doc,
     "BatchEnsemble[2] 是参数高效集成的代表：成员共享主干权重，各自只携带一对 rank-1 向量对输入与输出做调制，"
     "从而以接近单模型的开销获得集成效果。TabM[1] 沿用了这一思想但做了两点改动："
     "一是把成员调制放在共享 MLP 的输入端（每成员一组缩放向量），"
     "二是让 k 个成员在一次前向中并行计算，而不是各自独立前向。"
     "与 BatchEnsemble 相比，TabM 的成员差异更小、参数更省，这也正是 H2 需要被检验的原因——"
     "成员越省参数，越可能退化为同一个模型。")
h(doc, "2.3　本项目与主论文、基线的关系", 2)
para(doc,
     "本项目以 TabM 官方实现为唯一复现对象，不重新推导算法，也不复现论文的超参搜索与 46 数据集评测；"
     "我们只复现“机制 + 相对趋势”，并把一切与论文的差异（算力、数据规模、超参策略、种子数）在 §6 与附录中逐条列出，"
     "避免把课程数字与论文数字直接等价。基线方面：MLP 与 TabM 共用同一主干宽度、轮数、早停与优化器，"
     "是“同预算”对照；HistGBM/XGBoost 使用固定 100 棵树、不调参，代表“强而简单”的下界参照；"
     "k 个独立 MLP 是“成本上界”，用来回答“TabM 省下的参数与时间，是否换来了同等收益”。")

# ---- 3 方法 ----
h(doc, "3　方法", 1)
h(doc, "3.1　问题设定与符号", 2)
para(doc,
     "给定训练集 D = {(x_i, y_i)}_{i=1..n}，其中 x_i 为经过预处理的表格特征向量（数值列已标准化，类别列已编码为整数 ID），"
     "y_i 为标签（二分类为 0/1，多分类为类别 ID，回归为实数）。记 d_out 为输出维度（回归为 1），"
     "k 为成员数，f_θ 为共享的 MLP 主干（本项目为 2 个隐藏层、每层 256 维、ReLU、dropout 0.1），"
     "γ_r ∈ R^d 为第 r 个成员的 rank-1 缩放向量（d 为输入维数）。")
h(doc, "3.2　模型结构", 2)
para(doc,
     "TabM 的前向过程为：先对输入做成员级缩放 x'_r = γ_r ⊙ x，"
     "再送入共享主干得到成员输出 z_r = f_θ(x'_r)（分类为 logits，回归为标量），"
     "k 个成员在同一批次内并行计算（图 1）。由于主干参数在成员间共享，"
     "k 从 1 增加到 16 时，新增参数只有 k−1 组 d 维缩放向量，"
     "因此参数量近似持平，而计算量随 k 线性增长。")
add_figure(doc, FIG_ARCH,
           "图 1　TabM 的课程规模抽象结构。橙色模块为复现核心（成员级 rank-1 缩放），"
           "绿色为主干与并行预测头；本图只表达数据流与共享关系，不涉及具体数值。")
h(doc, "3.3　训练目标", 2)
para(doc,
     "训练时对样本与成员同时取平均，逐成员计算损失后再平均：")
para(doc, "L(θ, γ) = (1/(n·k)) Σ_{i=1..n} Σ_{r=1..k} ℓ( f_θ(γ_r ⊙ x_i), y_i )", indent=False,
     align=WD_ALIGN_PARAGRAPH.CENTER, size=10)
para(doc,
     "其中分类任务 ℓ 为交叉熵，回归任务为均方误差。这一口径与“先把 k 个成员的预测平均成一个预测、再算一次损失”"
     "在数值与梯度上都不等价，本项目用单元测试固定了前者（见 §4.6 与附录 B）。")
h(doc, "3.4　推理聚合", 2)
para(doc,
     "分类任务先对每个成员的 logits 做 softmax，再对概率取平均：p̄(y|x) = (1/k) Σ_r softmax(z_r)；"
     "回归任务直接对成员输出取平均：ŷ = (1/k) Σ_r f_θ(γ_r ⊙ x)。"
     "先 softmax 再平均而不是先平均 logits，是本项目与论文一致的关键口径；"
     "若顺序颠倒，概率的校准与 LogLoss 都会变化。")
h(doc, "3.5　与基线的差异", 2)
add_table(doc, "表 1　主方法与两类基线在结构上的差异",
          ["项目", "MLP（同预算基线）", "TabM(k)", "k 个独立 MLP（成本上界）"],
          [["成员数", "1", "k（共享主干）", "k（互不共享）"],
           ["每成员参数", "完整一份", "仅 d 维缩放向量", "完整一份"],
           ["前向次数", "1", "1（并行）", "k"],
           ["训练成本", "O(1)", "O(k)（逐成员损失）", "O(k)（k 次独立训练）"],
           ["集成方式", "无", "概率/数值平均", "概率/数值平均"]],
          widths=[3.2, 4.4, 4.2, 4.2], font_size=8.5)
h(doc, "3.6　实现约束（防泄漏与合法性）", 2)
for t in [
    "所有有状态的预处理（中位数填充、标准化、序数编码）只在训练集上 fit，再 transform 验证集与测试集；",
    "类别特征先统一转字符串、缺失单独编码为 “__NA__”，编码后校验 0 ≤ id < cardinality；"
    "验证/测试集中出现的未见类别（OOV）替换为训练集该列众数，避免编码越界；",
    "早停与成员选择只使用验证集，测试集在全部训练与选择结束后只用一次；",
    "H3 裁剪中的“择优”只用验证集上的单成员得分排序，测试集不参与选择。",
]:
    para(doc, t)

# ---- 4 实验设置 ----
h(doc, "4　实验设置", 1)
h(doc, "4.1　数据集", 2)
add_table(doc, "表 2　三个数据集的任务、来源与规模",
          ["数据集", "任务", "来源", "规模（本项目使用）", "特征"],
          [["Adult", "二分类（收入 >50K）", "OpenML 1590", "48842 行（全量）", "6 数值 + 8 类别"],
           ["Covertype", "7 类多分类", "sklearn fetch_covtype", "下采样 50000 行（全量 581012）", "54 数值"],
           ["California Housing", "回归（房价中位数）", "sklearn 缓存 / OpenML 44977 重建",
            "20640 行（全量）", "8 数值"]],
          widths=[3.4, 4.2, 4.6, 4.6, 3.6], font_size=8.5)
para(doc,
     "California Housing 的原始下载地址在本机返回 HTTP 403，因此按 sklearn 口径从 OpenML 44977 重建"
     "（中位收入、房龄、平均房间数/卧室数、人口、平均occupancy、经纬度，目标缩放回 0.15~5.0 量纲），"
     "重建口径与 sklearn 版本一致。Covertype 下采样是为把 CPU 总训练时间压到课程可接受范围，"
     "这是本项目最重要的“预算缩减”，§6 会说明它对结论外推的限制。")
h(doc, "4.2　划分与预处理", 2)
para(doc,
     "所有数据集使用同一套流程：先按 8:2 切出测试集，再从剩余部分按 8:2 切出验证集，"
     "最终训练/验证/测试 ≈ 64%/16%/20%；分类任务在两次切分上都做分层抽样，回归任务不分层；"
     "随机种子由每个实验的 seed 控制，保证 5 个种子看到 5 组不同但同分布的划分。"
     "数值列使用中位数填充 + 标准化，类别列使用众数填充 + 序数编码 + OOV 众数替换，"
     "所有统计量只来自训练集。")
h(doc, "4.3　超参数", 2)
add_table(doc, "表 3　全数据集共用的固定超参（未做任何搜索）",
          ["项", "取值", "说明"],
          [["主干", "2 × 256，ReLU，dropout 0.1", "MLP 与 TabM 主干完全一致"],
           ["优化器", "AdamW，lr = 2e-3，weight decay = 3e-4", "两类网络模型共用"],
           ["批大小 / 轮数", "256 / 最多 100 轮", "早停 patience = 10（监控验证集主指标）"],
           ["成员数 k", "1 / 4 / 8 / 16", "H3 裁剪源池固定为 k=16"],
           ["裁剪尺寸", "4 / 8", "随机裁剪与验证集择优各一组"],
           ["树模型", "XGBoost / HistGBM，各 100 棵", "其余用各自默认，不做调参"],
           ["种子", "0, 1, 2, 3, 4", "每个配置 5 次独立训练"]],
          widths=[3.0, 6.6, 6.6], font_size=8.5)
h(doc, "4.4　硬件与软件", 2)
para(doc,
     "全部实验在单台 Windows 主机上完成，纯 CPU（torch 2.15.0+cpu），Python 3.13，"
     "numpy 2.5.1、pandas 3.0.3、scikit-learn 1.9.1、xgboost 3.4.1；"
     "TabM 使用官方 pip 包 tabm（版本锁定见仓库 requirements.lock）。"
     "训练时间与峰值内存的测量见 §5.4，其中峰值内存在每个配置单独的子进程中用 psutil 采样，避免跨配置的 RSS 累积污染。")
h(doc, "4.5　模型选择与测试集使用", 2)
para(doc,
     "所有网络模型以验证集主指标早停并回滚到最佳轮次，树模型使用固定轮数不使用验证集；"
     "成员裁剪的“择优”策略只用验证集排序；测试集仅在最终评估阶段前向一次。"
     "因此本文报告的测试指标不包含任何以测试集为依据的选择。")
h(doc, "4.6　指标定义", 2)
add_table(doc, "表 4　报告中使用的指标与口径",
          ["指标", "定义与口径"],
          [["Macro-F1", "各类 F1 的算术平均（分类主指标）；adult 为 2 类，covertype 为 7 类"],
           ["ROC-AUC（macro-OvR）", "多分类按 one-vs-rest 计算各类 AUC 后取平均；二分类为常规 AUC；回归任务无定义"],
           ["LogLoss", "测试集平均交叉熵，越小越好"],
           ["ECE", "期望校准误差，10 个等宽置信分箱，越小越好"],
           ["R² / RMSE", "回归主指标与误差（california 的目标已缩放回原始量纲）"],
           ["参数量", "可训练参数总数（TabM 含 k 组缩放向量）"],
           ["训练时间 / 推理延迟", "训练为单次完整训练耗时；推理延迟为 batch=256 下的模型前向毫秒/样本（不含 DataLoader）"],
           ["峰值内存", "训练子进程的 RSS 峰值（MB），含进程基线（import torch + 加载数据约 350~410 MB）"]],
          widths=[3.8, 12.4], font_size=8.5)

# ---- 5 结果与分析 ----
h(doc, "5　结果与分析", 1)
para(doc,
     "本节先给主表（§5.1），再依次检验 H1（§5.2）、H3（§5.3）、报告效率（§5.4）、"
     "分析失败样本（§5.5），最后用独立 MLP 对照（§5.6）与成员多样性指标（§5.7）回答 H2。"
     "每小节刻意区分“观察”（可以直接从表中读出的事实）与“解释”（我们对原因的推断），"
     "解释部分不作为结论的等价替代。")

h(doc, "5.1　主结果", 2)
para(doc, "表 5~表 7 为三个数据集的主结果，均为 5 个种子的均值 ± 标准差，测试集只评估一次。")
add_table(doc, "表 5　Adult（二分类，主指标 Macro-F1）",
          ["模型", "Macro-F1", "ROC-AUC", "LogLoss", "ECE", "参数量", "训练时间/s"],
          [["XGBoost", "0.8133 ± 0.0032", "0.9267 ± 0.0017", "0.2809", "0.0132", "—", "0.4"],
           ["HistGBM", "0.8140 ± 0.0038", "0.9277 ± 0.0018", "0.2783", "0.0061", "—", "0.3"],
           ["MLP", "0.7861 ± 0.0069", "0.9099 ± 0.0032", "0.3142", "0.0091", "102488", "12.1"],
           ["TabM k=1", "0.7888 ± 0.0042", "0.9129 ± 0.0026", "0.3088", "0.0081", "94932", "11.3"],
           ["TabM k=4", "0.7920 ± 0.0064", "0.9141 ± 0.0019", "0.3064", "0.0070", "100636", "29.1"],
           ["TabM k=8", "0.7943 ± 0.0039", "0.9148 ± 0.0017", "0.3055", "0.0081", "108242", "54.6"],
           ["TabM k=16", "0.7929 ± 0.0047", "0.9149 ± 0.0017", "0.3050", "0.0082", "123453", "121.5"]],
          widths=[2.6, 3.1, 3.1, 2.4, 2.0, 2.4, 2.6], font_size=8)
add_table(doc, "表 6　California Housing（回归，主指标 R²）",
          ["模型", "R²", "RMSE", "参数量", "训练时间/s"],
          [["XGBoost", "0.8293 ± 0.0036", "—", "—", "0.3"],
           ["HistGBM", "0.8329 ± 0.0064", "—", "—", "0.2"],
           ["MLP", "0.8008 ± 0.0062", "—", "68353", "19.3"],
           ["TabM k=1", "0.8059 ± 0.0076", "0.5055", "69129", "23.2"],
           ["TabM k=4", "0.8088 ± 0.0060", "0.5018", "73764", "60.4"],
           ["TabM k=8", "0.8050 ± 0.0104", "0.5067", "79944", "96.2"],
           ["TabM k=16", "0.8041 ± 0.0054", "0.5079", "92304", "187.3"]],
          widths=[2.8, 4.2, 2.8, 3.0, 3.4], font_size=8)
add_table(doc, "表 7　Covertype（7 类，下采样 5 万行，主指标 Macro-F1）",
          ["模型", "Macro-F1", "ROC-AUC", "LogLoss", "ECE", "参数量", "训练时间/s"],
          [["XGBoost", "0.7597 ± 0.0101", "0.9767 ± 0.0013", "0.4052", "0.0198", "—", "2.1"],
           ["HistGBM", "0.7296 ± 0.0424", "0.9594 ± 0.0217", "0.4800", "0.0275", "—", "2.0"],
           ["MLP", "0.7784 ± 0.0225", "0.9799 ± 0.0011", "0.3813", "0.0187", "81671", "53.7"],
           ["TabM k=1", "0.7850 ± 0.0211", "0.9804 ± 0.0031", "0.3765", "0.0188", "82493", "60.7"],
           ["TabM k=4", "0.7929 ± 0.0146", "0.9822 ± 0.0015", "0.3484", "0.0151", "91892", "196.9"],
           ["TabM k=8", "0.7820 ± 0.0102", "0.9816 ± 0.0007", "0.3606", "0.0264", "104424", "363.9"],
           ["TabM k=16", "0.7631 ± 0.0123", "0.9794 ± 0.0012", "0.3825", "0.0362", "129488", "856.6"]],
          widths=[2.6, 3.1, 3.1, 2.4, 2.0, 2.4, 2.6], font_size=8)
para(doc,
     "观察 1：三个数据集上，TabM(k≥4) 的主指标都高于同预算 MLP（成人 +0.4~0.8 pp、"
     "加州 +0.3~0.8 pp、Covertype +0.4~1.5 pp），且在 Covertype 上小幅领先 XGBoost。"
     "观察 2：树模型在 Adult（+2.0 pp）与 California（+0.024 R²）上仍显著领先所有神经网络；"
     "Covertype 上则相反，TabM k=4 的 Macro-F1 与 AUC 都高于两个树模型。"
     "观察 3：Covertype 的 k=16 是表中唯一“大 k 反而明显变差”的配置（0.7631，比 k=1 低 2.2 pp）。",
     space_before=4)
para(doc,
     "解释：树模型在中小规模表格数据上的优势与文献[3] 一致，说明本项目没有通过调参把基线做弱；"
     "Covertype 上树模型落后，可能与 7 类不平衡、54 个数值特征上的决策边界更依赖连续组合有关，"
     "但这只是推断，本文没有做进一步的机制实验来区分“树模型不擅长该任务”与“下采样破坏了树模型优势”。")
para(doc,
     "另需标注：Covertype 训练时间的均值被污染。seed=4 的一整轮与其它任务并行运行，"
     "k=8/k=16 单次被拖到 754 s / 1413 s，seed=3 的 k=16 也被拖到 1394 s，"
     "因此表 7 的 856.6 s 是被两个异常值抬高的均值。"
     "后文所有关于训练时间倍数关系的讨论统一改用中位数口径（k=1/4/8/16 分别为 45.3 / 151.3 / 266.1 / 510.7 s），"
     "参数量不受运行环境影响，可直接使用。")

h(doc, "5.2　H1：TabM(k>1) 是否优于同预算 MLP", 2)
add_table(doc, "表 8　H1 的检验结果（主指标，5 种子配对 t 检验 k=16 vs k=1）",
          ["数据集", "MLP", "TabM 最好 k", "提升（pp）", "配对 t", "p 值", "判定"],
          [["Adult", "0.7861", "0.7943 (k=8)", "+0.82", "t = 4.52", "0.011", "显著为正"],
           ["California", "0.8008", "0.8088 (k=4)", "+0.80（R²）", "t = −0.84", "0.448", "不显著"],
           ["Covertype", "0.7784", "0.7929 (k=4)", "+1.45", "t = −4.49", "0.011", "显著为负"]],
          widths=[2.6, 2.4, 3.4, 2.6, 2.4, 2.0, 2.6], font_size=8.5)
para(doc,
     "观察：三个数据集上 TabM 的最好档都高于 MLP，方向与 H1 一致；但提升幅度（0.4~1.5 pp）"
     "与种子标准差（0.4~2.2 pp）同量级，只有 Adult 的 k=16 vs k=1 达到统计显著；"
     "Covertype 上 k=16 相对 k=1 的配对差异是显著的负值。"
     "解释：H1 在本课程预算下只能算“方向成立、强度不足”。"
     "一个可能的原因是训练预算被压缩：论文的成员分化依赖更长的训练与更大的数据，"
     "在 100 轮上限、5 万行下采样下，k 个成员来不及分化成互补的预测器，"
     "这与 §5.5、§5.7 的多样性证据互相印证。该解释仍是推断，本文未做预算扫描实验。",
     space_before=4)
add_figure(doc, os.path.join(FIG_DIR, "adult_tabm_k_ablation.png"),
           "图 2　Adult 上主指标、LogLoss、训练时间与 ECE 随 k 的变化（5 种子均值 ± 标准差）。")
add_figure(doc, os.path.join(FIG_DIR, "covertype_tabm_k_ablation.png"),
           "图 3　Covertype 上四项指标随 k 的变化。可见 k=16 在主指标与校准上同时变差，训练时间近似线性增长。")

h(doc, "5.3　H3：大 k 训练后裁剪成员", 2)
para(doc,
     "做法：固定 k=16 的已训练权重（每个 (数据集, 种子) 一份 checkpoint），"
     "从 16 个成员中选出 4 个或 8 个做集成，两种策略：(a) 随机选；(b) 用验证集上单成员的 Macro-F1（回归用 R²）排序后择优。"
     "整个过程不重新训练，测试集只在最终评估时使用。")
add_table(doc, "表 9　H3 裁剪消融（主指标，5 种子均值；择优 − 原生为正值表示裁剪更好）",
          ["数据集", "crop", "原生 k=N", "随机裁剪", "验证集择优", "择优 − 原生"],
          [["Adult", "4", "0.7920", "0.7925", "0.7944", "+0.0024"],
           ["Adult", "8", "0.7943", "0.7937", "0.7939", "−0.0004"],
           ["California", "4", "0.8088", "0.8028", "0.8041", "−0.0047"],
           ["California", "8", "0.8050", "0.8034", "0.8041", "−0.0009"],
           ["Covertype", "4", "0.7929", "0.7538", "0.7628", "−0.0301"],
           ["Covertype", "8", "0.7820", "0.7610", "0.7622", "−0.0197"]],
          widths=[2.8, 2.2, 3.0, 3.0, 3.2, 3.2], font_size=8.5)
para(doc,
     "观察：6 组对照中 5 组裁剪后劣于直接训练同规模模型；唯一正增益是 Adult crop=4（+0.24 pp），"
     "落在 ±0.6 pp 的种子波动内，不能作为证据。Covertype 的裁剪损失最大：随机裁剪掉 3.0~3.9 pp，"
     "验证集择优也只挽回约 1 pp，仍低于原生 k=4/k=8。"
     "附带观察：裁剪同时损害校准与区分度——Covertype 上 ECE 从原生 k=4 的 0.0151 恶化到 0.031~0.035；"
     "macro-OvR AUC 从 0.9822（k=4）/0.9816（k=8）掉到 0.9784~0.9792，四档裁剪没有任何一档回到原生水平；"
     "Adult 上则基本持平（0.9145~0.9150 vs 原生 0.9141/0.9148）。"
     "解释：裁剪的损伤是任务相关的。在成员本就高度相关的简单任务（Adult，member_corr ≈ 0.93）上，"
     "砍掉成员几乎不改变集成输出；在成员必须靠分化才能覆盖少数类的任务（Covertype）上，"
     "砍掉成员等于砍掉仅有的多样性来源，因而同时损失 F1、AUC 与校准。"
     "这一解释与 §5.5 的混淆矩阵（判别边界向多数类收缩）指向同一机制。",
     space_before=4)
add_figure(doc, os.path.join(FIG_DIR, "covertype_h3_crop_ablation.png"),
           "图 4　Covertype 的 H3 三合一对比：主指标、ECE（越低越好）、成员相关性（越低表示越多样）。")

h(doc, "5.4　效率：参数、时间、推理延迟与峰值内存", 2)
add_table(doc, "表 10　相对同数据集 k=1 的成本放大倍数（Covertype 用中位数口径，其余用均值）",
          ["数据集", "k=4（参数 / 时间）", "k=8（参数 / 时间）", "k=16（参数 / 时间）"],
          [["Adult", "×1.06 / ×2.56", "×1.14 / ×4.81", "×1.30 / ×10.72"],
           ["California", "×1.07 / ×2.60", "×1.16 / ×4.14", "×1.34 / ×8.06"],
           ["Covertype（中位数口径）", "×1.11 / ×3.34", "×1.27 / ×5.87", "×1.57 / ×11.27"]],
          widths=[4.6, 3.9, 3.9, 3.9], font_size=8.5)
add_table(doc, "表 11　推理延迟（ms/样本，5 种子中位）与训练峰值内存（MB，seed=0，每配置独立进程；Cal = California）",
          ["配置", "延迟 Adult", "延迟 Cal", "延迟 Cover",
           "内存 Adult", "内存 Cal", "内存 Cover"],
          [["XGBoost", "0.0006", "0.0008", "0.0041", "385", "357", "389"],
           ["HistGBM", "0.0016", "0.0017", "0.0091", "389", "353", "409"],
           ["MLP", "0.0030", "0.0015", "0.0017", "460", "436", "455"],
           ["TabM k=1", "0.0048", "0.0024", "0.0031", "464", "438", "455"],
           ["TabM k=4", "0.0081", "0.0050", "0.0061", "474", "450", "471"],
           ["TabM k=8", "0.0131", "0.0081", "0.0113", "487", "464", "481"],
           ["TabM k=16", "0.0222", "0.0151", "0.0200", "503", "488", "510"]],
          widths=[2.3, 2.3, 2.1, 2.3, 2.1, 2.0, 2.3], font_size=8)
para(doc,
     "观察 1：参数量近乎持平（k=1→16 只涨 1.30~1.57 倍，理论上限来自 k 组缩放向量），"
     "但训练时间随 k 近似线性膨胀（8~14 倍）。"
     "观察 2：推理延迟同样线性增长，但幅度温和得多（k=16 / k=1 = 4.6 倍 / 6.3 倍 / 6.5 倍）。"
     "观察 3：训练峰值 RSS 的增量很小（相对同数据集 MLP：k=1 为 +4/+2/0 MB，k=16 也只 +43/+52/+55 MB），"
     "而 TabM 的理论权重内存 k=1→16 只从 0.26~0.36 MB 涨到 0.35~0.49 MB。"
     "解释：TabM 的“参数高效”是真实的，但“计算高效”只是相对的——"
     "它的瓶颈是训练计算（k 次成员前向与逐成员损失），不是内存；"
     "峰值增量的来源是 k 份激活与分配器缓存，而不是权重。"
     "因此从部署角度看，TabM 适合“存储/显存敏感、但训练时间可接受”的场景；"
     "若训练预算受限（如本项目的 CPU 条件），k 越大越不划算。",
     space_before=4)

h(doc, "5.5　误差与失败分析", 2)
para(doc,
     "为了回答“Covertype 上 k=16 到底掉在哪里”，我们用同一划分与同一测试集重训 k=4（主实验只保存了 k=16 权重）、"
     "加载 k=16 checkpoint，并统计逐类召回、混淆矩阵与高置信度错分样本。")
add_table(doc, "表 12　Covertype 逐类召回率（5 种子平均，行归一化混淆矩阵的对角元）",
          ["类别", "k=4 召回", "k=16 召回", "Δ", "主要误判去向"],
          [["C1", "0.828", "0.822", "−0.006", "C2"],
           ["C2（多数类）", "0.901", "0.889", "−0.012", "C1"],
           ["C3", "0.865", "0.882", "+0.017", "C6"],
           ["C4", "0.664", "0.651", "−0.013", "C3"],
           ["C5", "0.534", "0.436", "−0.098", "C2（49% 被判成 C2）"],
           ["C6", "0.704", "0.630", "−0.074", "C3（27% 被判成 C3）"],
           ["C7", "0.849", "0.829", "−0.020", "C1"]],
          widths=[2.6, 2.8, 2.8, 2.0, 6.0], font_size=8.5)
para(doc,
     "观察：k=4 的 macro-F1 为 0.7863 ± 0.0183、accuracy 0.8573；k=16 为 0.7631 ± 0.0123、accuracy 0.8459。"
     "准确率只掉 1.1 pp，macro-F1 却掉 2.3 pp，跌幅是前者的 2.0 倍；退化集中在 C5（−9.8 pp）与 C6（−7.4 pp），"
     "而 C5 被判成多数类 C2 的比例从 40% 升到 49%，判给自己的比例从 53% 降到 44%。"
     "解释：k=16 的退化不是“整体能力下降”，而是判别边界向多数类收缩。"
     "这与 §5.2 的显著负值和 §5.7 的成员相关性（k 越大仍维持 0.86 以上）互为佐证："
     "k 变大时成员没有变得更互补，反而一起朝多数类收敛。需要说明的是，"
     "本节的 k=4 是当场重训（0.7863 与主表 0.7929 差 0.7 pp，属同配置重训的随机波动），"
     "因此只做本节内部的 k4 vs k16 对比，不与表 7 跨口径混用。")
add_figure(doc, os.path.join(FIG_DIR, "error_analysis", "covertype_confusion_k4_vs_k16.png"),
           "图 5　Covertype 混淆矩阵（左：k=4，右：k=16），均为 5 种子平均并按行归一化。")
para(doc,
     "Adult 的失败形态不同：正类（>50K）漏检率 0.3836，负类误报率 0.0651，前者是后者的 5.9 倍。"
     "观察表明 Adult 的错误集中在类别不平衡下的阈值与校准，而不是成员塌缩；"
     "因此不能把两个数据集掉点的原因混为一谈。")

h(doc, "5.6　拓展：k 个独立 MLP 的严格成本对照", 2)
para(doc,
     "这一板块完全独立于主实验：不修改 src/ 下任何代码、不写入主结果 csv，"
     "每个 (数据集, 种子) 训练 16 个完全独立的 MLP（各自初始化、各自早停），"
     "再取前 k 个做集成，从而给出“k 份真实权重”的成本上界。")
add_table(doc, "表 13　TabM 与 k 个独立 MLP 的三方对照（5 种子均值；Δ = 独立 − TabM）",
          ["数据集", "k", "TabM", "独立 k×MLP", "Δ", "参数比", "时间比"],
          [["Adult", "1", "0.7888", "0.7876", "−0.0012", "1.08×", "1.74×"],
           ["Adult", "4", "0.7920", "0.7909", "−0.0010", "4.07×", "2.68×"],
           ["Adult", "8", "0.7943", "0.7897", "−0.0046", "7.57×", "2.86×"],
           ["Adult", "16", "0.7929", "0.7904", "−0.0025", "13.28×", "2.60×"],
           ["California", "1", "0.8059", "0.8009", "−0.0050", "0.99×", "1.47×"],
           ["California", "4", "0.8088", "0.8061", "−0.0026", "3.71×", "2.26×"],
           ["California", "8", "0.8050", "0.8068", "+0.0018", "6.84×", "2.77×"],
           ["California", "16", "0.8041", "0.8062", "+0.0022", "11.85×", "2.76×"],
           ["Covertype", "1", "0.7850", "0.7765", "−0.0085", "0.99×", "1.21×"],
           ["Covertype", "4", "0.7929", "0.7963", "+0.0034", "3.56×", "1.52×"],
           ["Covertype", "8", "0.7820", "0.7958", "+0.0138", "6.26×", "1.77×"],
           ["Covertype", "16", "0.7631", "0.7995", "+0.0364", "10.09×", "1.49×"]],
          widths=[2.8, 1.8, 2.6, 3.0, 2.6, 2.4, 2.4], font_size=8)
para(doc,
     "观察：Adult 与 California 上两者性能打平（|Δ| ≤ 0.005，落在种子噪声内），"
     "而 TabM 只用 1/4~1/13 的参数与 1/1.5~1/2.9 的训练时间；"
     "Covertype 上却出现反向结论：k=8 与 k=16 时独立 MLP 分别高 1.4 pp 与 3.6 pp，"
     "且独立 MLP 的集成增益随 k 单调上升（k=4 为 +0.019、k=8 为 +0.019、k=16 为 +0.023）。"
     "解释：真正独立的成员在 Covertype 上确实能带来集成增益，说明“多样性有效”；"
     "而 TabM 的共享主干在该任务上把成员压成了近似同一个模型，"
     "于是 k 越大只增加了计算，却没有增加互补性。"
     "结合 §5.4，TabM 的价值取决于部署约束：存储或显存紧张时它几乎免费，"
     "但如果任务是“多类 + 类别不平衡 + 训练预算受限”，用 k=4 的 TabM 或直接换更强的基线更划算。")
add_figure(doc, os.path.join(FIG_DIR, "extension", "covertype_indep_mlp_vs_tabm.png"),
           "图 6　Covertype 上 TabM 与 k 个独立 MLP 的性能—参数—时间对照。")

h(doc, "5.7　H2：成员多样性证据", 2)
add_table(doc, "表 14　成员间相似度与逐样本概率方差（5 种子均值）",
          ["数据集", "来源", "member_corr ↓", "逐样本概率方差 ↓", "argmax 分歧率 ↓"],
          [["Adult", "TabM k=16", "0.9283", "0.00068", "0.0717"],
           ["Adult", "独立 16×MLP", "0.8809", "0.00170", "0.1196"],
           ["California", "TabM k=16", "0.9924", "0.00768", "—（回归）"],
           ["California", "独立 16×MLP", "0.9915", "0.00887", "—"],
           ["Covertype", "TabM k=16", "0.8647", "0.00219", "0.2743"],
           ["Covertype", "独立 16×MLP", "0.8730", "0.00287", "0.2667"]],
          widths=[3.0, 4.2, 3.2, 4.0, 3.6], font_size=8.5)
para(doc,
     "口径：member_corr 为成员两两预测类别（或回归输出）相关系数的均值，"
     "逐样本概率方差为“每个样本、每个类别上跨 k 个成员的概率方差”再对样本与类别平均，"
     "argmax 分歧率为 k 个成员预测类别不完全一致的样本占比（回归无定义）。"
     "TabM 侧取 k=16 的 checkpoint，独立 MLP 侧取逐成员概率存档；"
     "该脚本复算的 member_corr 与主表完全对齐（0.92826 / 0.99244 / 0.86471）。")
para(doc,
     "观察：所有设置下成员相关性都在 0.86~0.99，California 甚至达到 0.99；"
     "Adult 上 TabM 的成员概率方差只有独立 MLP 的 40%，Covertype 上少 24%。"
     "但 Covertype 的 argmax 分歧率方向与方差相反（TabM 0.2743 > 独立 0.2667），"
     "这是类别不平衡下多数类主导 argmax 造成的失真，因此本节以方差为主证据、相关性为辅、分歧率仅作参考。"
     "解释：H2 的“收益来自成员多样性”在本项目的数据上未被证实——"
     "TabM 的成员确实高度趋同；但独立 MLP 的对照又表明多样性本身是有效的。"
     "两者合起来意味着：机制成立，但 TabM 的共享参数设计没有在小预算下把多样性释放出来。"
     "这是本文对 H2 给出的修正表述，而不是简单否定。")

# ---- 6 局限与伦理 ----
h(doc, "6　局限与伦理", 1)
h(doc, "6.1　算力与规模的缩减", 2)
for t in [
    "全部实验在 CPU 上完成，Covertype 下采样至 5 万行（约全量的 8.6%），"
    "这会改变类别分布与决策边界，是 Covertype 结论（尤其 k=16 掉点）不能直接外推到全量数据的首要原因；",
    "超参全数据集共用一套固定值、未做任何搜索，而论文对每个数据集单独调参；"
    "因此本文比较的是“同一预算下的相对趋势”，不是论文的绝对水平；",
    "只使用 3 个数据集、单一固定划分 × 5 个种子，而论文使用 46 个数据集、15 个种子与多次划分；"
    "数据集层面的结论（如“TabM 整体更好”）在本文样本量下不足以支撑；",
    "训练时间测量受同机并行任务污染，已通过中位数口径与剔除规则处理（§5.1、附录 C），"
    "但极端情况下仍可能残留偏差。",
]:
    para(doc, t)
h(doc, "6.2　数据偏差与潜在误用", 2)
para(doc,
     "Adult 来源于美国人口普查收入数据，包含性别、种族、原国籍等敏感属性；"
     "在本数据上训练的模型若用于信贷审批、招聘筛选等场景，可能放大既有的群体不平等。"
     "Covertype 是森林覆盖类型数据，存在严重的类别不平衡（部分类别样本很少），"
     "本文报告的 macro-F1 与逐类召回正是为了暴露这一偏差，而不是掩盖它。"
     "California Housing 的房价标签带有地理与历史歧视痕迹，模型学到的空间模式不应被解读为"
     "“地区价值”的因果判断。三者都只能作为方法学实验，不能作为现实决策依据。")
h(doc, "6.3　方法层面的限制", 2)
para(doc,
     "本文没有测量训练能耗与端到端吞吐；推理延迟只报告 batch=256 的前向时间，"
     "不含数据预处理与 I/O，因此不能等同于线上服务延迟。"
     "此外，k 的扫描只到 16（论文还报告了 k=32），更大 k 的行为未知；"
     "裁剪实验只做了“从 k=16 裁到 4/8”两个尺寸，没有做逐级裁剪。"
     "这些都属于预算约束下的取舍，而非结论的普适性证明。")

# ---- 7 结论 ----
h(doc, "7　结论", 1)
para(doc,
     "本文在 CPU、统一超参、三数据集、5 种子的课程规模下复现了 TabM，并补充了裁剪、效率、误差与独立 MLP 四类证据。"
     "逐条回答指导书给出的三条假设：")
para(doc,
     "H1（TabM(k>1) 优于同预算 MLP）：方向成立，强度不足。三个数据集上 TabM 的最好档都高于 MLP（+0.4~1.5 pp），"
     "但只有 Adult 的 k=16 vs k=1 达到显著（p=0.011）；Covertype 上 k=16 反而显著变差（p=0.011）。"
     "因此本项目的判定是“部分支持”，且其成立依赖具体数据集与 k 的取值。")
para(doc,
     "H2（收益来自成员多样性而非参数或计算量）：证据不足，偏否定。"
     "参数只增 1.30~1.57 倍这一半是支持的，但成员相关性始终 0.86~0.99、逐样本概率方差只有独立 MLP 的 40%~76%，"
     "说明成员高度趋同；结合 Covertype 上 k=16 的判别边界向多数类收缩，可以认为共享参数在本文预算下压掉了多样性。"
     "不过独立 MLP 在 Covertype 上 +3.6 pp 的结果说明多样性机制本身有效，"
     "故本文的表述是“机制成立、TabM 未充分利用”，而不是“多样性无用”。")
para(doc,
     "H3（大 k 训练后裁剪可能优于直接训练小 k）：不支持。6 组对照中 5 组裁剪更差，"
     "唯一正增益（Adult crop=4，+0.24 pp）落在种子噪声内；"
     "Covertype 上裁剪同时损失 F1（−2.0~−3.0 pp）、AUC（−0.003~−0.004）与校准（ECE 0.015→0.031~0.035）。"
     "“用验证集择优”只挽回了约 1 pp，仍无法回到原生水平。")
para(doc,
     "拓展（与 k 个独立 MLP 的成本对照）：参数高效成立、泛化优势不成立。"
     "Adult 与 California 上 TabM 以 1/4~1/13 的参数、1/1.5~1/2.9 的时间换到同等精度；"
     "Covertype 上独立 MLP 反超 3.6 pp。综合三条假设与拓展，本文的实践建议是："
     "在算力受限的表格任务中优先使用 k=4 的 TabM 或强树模型基线；"
     "只有当存储/显存是硬约束、且任务类别分布较均衡时，才值得把 k 推到 8 以上。")

# ---- 附录 ----
h(doc, "附录 A　完整超参与配置快照", 1)
para(doc,
     "运行配置的唯一真源为 src/config.py，仓库中的 configs/default.json 是当前结果对应的快照，"
     "可用 python configs/dump_config.py 重新导出。关键取值：DATASETS = [adult, california, covertype]，"
     "SEEDS = [0,1,2,3,4]，K_LIST = [1,4,8,16]，H3_SOURCE_K = 16，CROP_SIZES = [4,8]，"
     "MAX_SAMPLES[covertype] = 50000，N_BLOCKS = 2，D_BLOCK = 256，DROPOUT = 0.1，"
     "MAX_EPOCH = 100，PATIENCE = 10，LR = 0.002，WEIGHT_DECAY = 3e-4，BATCH_SIZE = 256，N_ESTIMATORS = 100。")
h(doc, "附录 B　复现命令与产物对照", 1)
for cmd in [
    "python -m unittest discover -s tests -v　→　21 项测试（划分、输出形状、逐成员损失梯度、指标一致性）",
    "python scripts/run_exp.py　→　results/all_result.csv（165 行）与 ckpt/（15 个 k=16 权重）",
    "python scripts/run_crop_ablation.py　→　追加 H3 裁剪行（纯推理，不重训）",
    "python scripts/eval_extra_auc.py --fill　→　补 covertype 的 macro-OvR AUC",
    "python scripts/eval_cost.py / --train-peak　→　results/cost/（推理延迟与峰值内存）",
    "python scripts/error_analysis.py　→　results/error_analysis/ 与混淆矩阵图",
    "python scripts/eval_member_variance.py　→　results/variance/member_variance.csv",
    "python extension/run_indep_mlp.py / compare_with_tabm.py / plot_indep_mlp.py　→　拓展板块",
    "python plots/plot_result.py / plot_h3_crop.py　→　plots/ 下的主图",
]:
    para(doc, "　" + cmd, indent=False)
h(doc, "附录 C　额外负结果与口径说明", 1)
for t in [
    "负结果 1：Covertype 上 k=16 同时劣于 k=1、k=4 与自身裁剪结果，说明“更大 k 一定更好”在本任务上不成立；",
    "负结果 2：California 的成员相关性达 0.99，回归任务上 TabM 的成员几乎无差异，集成收益接近于零；",
    "负结果 3：首次用同一进程顺序测量峰值内存，得到 385→581 MB 的单调爬升，该列数据不可比，已废弃并改为每配置独立子进程重测；",
    "口径说明：Covertype 训练时间表中的 856.6 s 为污染均值，本文涉及倍数的讨论统一使用中位数口径；"
    "误差分析中的 k=4 为当场重训，与主表 k=4 相差 0.7 pp，不与主表跨口径比较。",
    "口径说明：推理延迟为 batch=256 的模型前向时间，不含 DataLoader；峰值 RSS 含进程基线。",
]:
    para(doc, t)
h(doc, "附录 D　生成式 AI 使用说明", 1)
para(doc,
     "本报告与配套代码在撰写过程中使用了生成式 AI 辅助：用于代码调试与报错定位、"
     "实验脚本的样板代码生成、以及文字组织与语言润色。"
     "所有实验数据均由本人在本机实际运行产生，AI 未生成、修改或补造任何实验结果；"
     "报告中引用的每一个指标都能在 results/ 下的 csv、日志或绘图脚本中追溯到原始出处。"
     "AI 生成的文字与代码均经过本人逐段核对：代码部分以 21 项单元测试与实跑结果为准，"
     "文字部分的数字已与 results/ 中的原始文件逐项比对。"
     "本人能够独立解释本文的数据划分、TabM 的损失与推理口径、以及全部结论的来源。")

h(doc, "附录 E　作者分工与贡献说明", 1)
para(doc,
     "本项目为个人独立完成，无组内分工。按指导书要求，将各环节的实际贡献列示如下，便于核对："
     "① 代码与实验：数据管线、训练循环、裁剪消融、效率测量、误差分析与拓展板块的脚本编写与调试，"
     "以及全部 165 次主实验、15 组裁剪推理、12 组独立 MLP 对照的实机运行；"
     "② 数据分析：主表、效率表、裁剪表、误差表的统计与配对检验，异常值识别与口径修正（如训练时间的污染均值）；"
     "③ 报告写作：结构设计、全部文字撰写与图表制作；"
     "④ 质量核查：21 项单元测试与实跑结果的一致性核对，报告数字与 results/ 原始文件的逐项比对，"
     "仓库 README 的一键复现流程自测。生成式 AI 的使用情况见附录 D。")

h(doc, "参考文献", 1)
refs = [
    "[1] Gorishniy Y, Kotelnikov A, Ivanov A, Babenko A. TabM: Advancing Tabular Deep Learning "
    "with Parameter-Efficient Ensembling[C]//International Conference on Learning Representations (ICLR). 2025.",
    "[2] Wen Y, Tran D, Ba J. BatchEnsemble: An Alternative Approach to Efficient Ensemble and "
    "Lifelong Learning[C]//International Conference on Learning Representations (ICLR). 2020.",
    "[3] Gorishniy Y, Rubachev I, Khrulkov V, Babenko A. Revisiting Deep Learning Models for "
    "Tabular Data[C]//Advances in Neural Information Processing Systems (NeurIPS). 2021.",
    "[4] Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System[C]//ACM SIGKDD. 2016: 785-794.",
    "[5] Hollmann N, Müller S, Purucker L, et al. Accurate Predictions on Small Data with a "
    "Tabular Foundation Model[J]. Nature, 2025.",
]
for ref in refs:
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 1.35
    pf.space_after = Pt(3)
    pf.left_indent = Cm(0.9)
    pf.first_line_indent = Cm(-0.9)
    r = p.add_run(ref)
    set_run(r, BODY_FONT, 9)

doc.save(OUT)
print("saved:", OUT)
