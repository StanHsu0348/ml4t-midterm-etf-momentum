"""Build slides/midterm_presentation.pptx (11 slides, 10-minute talk) from results/.

Numbers are read from results/tables so the deck and the report cannot disagree.
"""
import json
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
TAB, FIG = ROOT / "results" / "tables", ROOT / "results" / "figures"
OUT = ROOT / "slides" / "midterm_presentation.pptx"

INK = RGBColor(0x1F, 0x2A, 0x30)      # dominant dark slate
TEAL = RGBColor(0x2A, 0x7F, 0x7A)     # secondary
AMBER = RGBColor(0xE3, 0x9B, 0x2D)    # accent (momentum)
RED = RGBColor(0xC0, 0x4B, 0x3E)
MUTED = RGBColor(0x5F, 0x6B, 0x73)
TINT = RGBColor(0xEE, 0xF3, 0xF3)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LATIN, EA = "Calibri", "Microsoft JhengHei"

val_ic = pd.read_csv(TAB / "04_ic_summary.csv", index_col=0)
paired = pd.read_csv(TAB / "04_paired.csv").set_index("comparison")
head = pd.read_csv(TAB / "07_headline.csv")
ho_perf = pd.read_csv(TAB / "06_holdout_performance.csv", index_col=0)
val_perf = pd.read_csv(TAB / "05_performance_validation.csv", index_col=0)
tri = pd.read_csv(TAB / "03_triage.csv")
ho_inf = json.loads((TAB / "06_holdout_inference.json").read_text())
val_inf = json.loads((TAB / "05_inference.json").read_text())
mech = pd.read_csv(TAB / "05_mechanism_vol_terciles.csv", index_col=0)
counts = tri.decision.value_counts()

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
W = 13.333


def font(run, size, bold=False, color=INK, italic=False):
    run.font.size, run.font.bold, run.font.italic = Pt(size), bold, italic
    run.font.color.rgb = color
    run.font.name = LATIN
    rpr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rpr.find(qn(tag))
        if el is None:
            el = rpr.makeelement(qn(tag), {})
            rpr.append(el)
        el.set("typeface", EA)


def text(slide, x, y, w, h, paras, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, name=None, spacing=6):
    """paras: list of str or list of (str, dict) runs per paragraph."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(spacing)
        runs = p if isinstance(p, list) else [(p, {})]
        for s, o in runs:
            r = para.add_run()
            r.text = s
            font(r, o.get("size", size), o.get("bold", bold), o.get("color", color), o.get("italic", False))
    return tb


def box(slide, x, y, w, h, fill=TINT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, line=None, name=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line; s.line.width = Pt(1.25)
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.08
    s.shadow.inherit = False
    if name:
        s.name = name
    return s


def badge(slide, x, y, label, fill=TEAL, d=0.55, size=16):
    c = box(slide, x, y, d, d, fill=fill, shape=MSO_SHAPE.OVAL)
    tf = c.text_frame; tf.margin_left = tf.margin_right = 0
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = label; font(r, size, True, WHITE)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    return c


def title(slide, s, sub=None, dark=False):
    text(slide, 0.6, 0.45, W - 1.2, 0.8, [s], size=34, bold=True, color=WHITE if dark else INK, name="Title")
    if sub:
        text(slide, 0.6, 1.2, W - 1.2, 0.5, [sub], size=16, color=RGBColor(0xC9, 0xD6, 0xD6) if dark else MUTED, name="Subtitle")


def footer(slide, n, dark=False):
    text(slide, 0.6, 7.0, 8, 0.3, ["ML4T 期中 Proposal · 跨資產 ETF 動能 × 波動狀態"], size=10, color=MUTED if not dark else RGBColor(0x9A, 0xA8, 0xA8))
    text(slide, W - 1.2, 7.0, 0.6, 0.3, [str(n)], size=10, color=MUTED if not dark else RGBColor(0x9A, 0xA8, 0xA8), align=PP_ALIGN.RIGHT)


def bg(slide, color):
    f = slide.background.fill; f.solid(); f.fore_color.rgb = color


def new(dark=False):
    s = prs.slides.add_slide(BLANK)
    bg(s, INK if dark else WHITE)
    return s


def stat(slide, x, y, w, big, label, color=TEAL, big_size=40):
    text(slide, x, y, w, 0.8, [big], size=big_size, bold=True, color=color)
    text(slide, x, y + 0.85, w, 0.7, [label], size=13, color=MUTED)


# ---------------------------------------------------------------- 1 title
s = new(dark=True)
text(s, 0.8, 1.6, 11.5, 1.6, ["跨資產 ETF 動能 × 波動狀態"], size=48, bold=True, color=WHITE)
text(s, 0.8, 2.75, 11.5, 0.8, ["一次完整的 ML4T 研究循環：從假說到只碰一次的 holdout"], size=22, color=RGBColor(0xC9, 0xD6, 0xD6))
for i, (big, lab) in enumerate([("100", "檔跨資產 ETF，2006–2025"), ("93", "次記帳試驗（餵給 DSR）"), ("1", "次 holdout 觸碰（有紀錄）")]):
    text(s, 0.8 + i * 3.9, 4.3, 3.6, 0.9, [big], size=54, bold=True, color=AMBER)
    text(s, 0.8 + i * 3.9, 5.25, 3.6, 0.5, [lab], size=15, color=WHITE)
text(s, 0.8, 6.45, 11, 0.4, ["徐旻鉉　·　Machine Learning for Algorithmic Trading　·　2026-11"], size=14, color=RGBColor(0x9A, 0xA8, 0xA8))
s.notes_slide.notes_text_frame.text = ("大家好，我的期中是把 ML4T 流程完整走一圈：100 檔跨資產 ETF、月頻只做多。"
    "重點不是 Sharpe 有多高，而是流程：所有調整都記帳，總共 93 次試驗；holdout 在第一週封存，最後只碰一次，而且有存取紀錄。")

# ---------------------------------------------------------------- 2 mechanism
s = new(); title(s, "機制優先：為什麼動能應該存在", "來源：資金流可預測（主）＋ 承擔崩潰風險的補償（輔）")
box(s, 0.6, 1.9, 6.2, 4.7, fill=TINT)
text(s, 0.95, 2.15, 5.6, 4.3, [
    [("機制（兩句話）", {"bold": True, "size": 18, "color": TEAL})],
    "資金移動緩慢、投資人反應不足，使 ETF 過去 6–12 個月的相對強弱在下個月延續；對手方是依日曆再平衡的配置者。",
    "波動剛飆升後，崩跌的輸家劇烈反彈（momentum crash），所以波動狀態應能「開關」這個訊號。",
], size=16, spacing=10)
text(s, 7.3, 1.95, 5.5, 0.5, [[("事先寫下的證偽條件", {"bold": True, "size": 18, "color": INK})]])
fals = ["驗證期 IC 的 NW 95% CI 含 0 → 停止", "損益兩平成本 < 2× 假設成本 → 不可交易", "holdout 淨 Sharpe < 0 或 IC ≤ 0 → 未確認", "高波動時動能 IC 不比平靜時低 → 開關被否定"]
for i, f in enumerate(fals):
    badge(s, 7.3, 2.65 + i * 1.0, str(i + 1), fill=AMBER if i == 3 else TEAL)
    text(s, 8.05, 2.7 + i * 1.0, 4.8, 0.6, [f], size=16, anchor=MSO_ANCHOR.MIDDLE)
footer(s, 2)
s.notes_slide.notes_text_frame.text = ("我採機制優先。兩句話的機制如投影片。四個證偽條件在看任何結果之前就寫進 setup.yaml，"
    "第四條是專門檢驗『波動開關』這半邊的假說，後面會看到它在兩段資料給出相反答案。")

# ---------------------------------------------------------------- 3 setup + protocol timeline
s = new(); title(s, "交易設定 v1 與驗證協定：先承諾、再看結果", "setup.yaml 在第一個 commit；holdout 2026-11-08 封存，程式閘門強制只開一次")
segs = [("2008–2014.11", "特徵篩選\n(discovery)", 2.2, MUTED), ("2014", "調參\n(固定切分)", 1.1, MUTED),
        ("2015 – 2023", "Walk-forward 驗證\n9 個年度 fold，purge 22 日", 5.5, TEAL), ("2024–2025", "Holdout\n封存，只碰一次", 2.6, AMBER)]
x = 0.6
for lab, desc, w, col in segs:
    box(s, x, 2.0, w - 0.08, 0.9, fill=col)
    text(s, x + 0.12, 2.12, w - 0.3, 0.7, [[(lab, {"bold": True, "color": WHITE, "size": 15})]], anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 0.05, 3.0, w - 0.2, 0.8, desc.split("\n"), size=13, color=INK, spacing=2)
    x += w
cards = [("決策時點", "月底收盤算分數\n→ 隔日開盤成交"), ("分數→交易", "前 10 名、等權 10%\n只做多、滿倉"),
         ("標籤", "open[t+22]/open[t+1]−1\n與成交規則一致"), ("成本", "$0.0035/股＋分級半價差\n用未調整價換算")]
for i, (h, b) in enumerate(cards):
    cx = 0.6 + i * 3.08
    box(s, cx, 4.35, 2.88, 2.2, fill=TINT)
    text(s, cx + 0.25, 4.55, 2.4, 0.45, [[(h, {"bold": True, "color": TEAL, "size": 17})]])
    text(s, cx + 0.25, 5.15, 2.45, 1.3, b.split("\n"), size=15, spacing=4)
footer(s, 3)
s.notes_slide.notes_text_frame.text = ("協定的六項承諾：擴張窗、一年一步、每年重訓、標籤 21 日、purge 22 日、最長回看 252 日。"
    "篩選、調參、驗證、holdout 四段互不重疊；檢查後我把篩選與調參窗的尾端也 purge，截在 2014-11-28。"
    "標籤從隔日開盤算到第 22 天開盤，和回測成交價完全一致。")

# ---------------------------------------------------------------- 4 data
s = new(); title(s, "資料：先證明時間點正確，再看報酬", "Yahoo Finance（yfinance 1.8.0），470,662 列；與課本快照逐日比對差異 > 10bp 的比例 = 0")
stat(s, 0.6, 1.95, 3.0, "0", "OHLC 違反 / 重複列")
stat(s, 3.7, 1.95, 3.0, "158", "零成交量列：標記並排除\n（不默默補值）", color=AMBER)
stat(s, 6.8, 1.95, 3.0, "87", "每月可交易檔數中位數\n（ADV ≥ $10M、歷史 ≥ 273 日）")
box(s, 0.6, 4.0, 5.9, 2.6, fill=TINT)
text(s, 0.9, 4.2, 5.4, 2.3, [[("發現：Yahoo 的 Close 已做分割調整", {"bold": True, "size": 17, "color": RED})],
     "SMH 一拆二當天「原始」報酬 +2.1% → 每股手續費會算錯。",
     "以未來分割比例重建實際成交價：未調整 −48.9%、調整後 +2.1%。"], size=15, spacing=8)
pit = [("數值", "截斷測試：所有特徵在 t 與全樣本完全相同", "PASS", TEAL),
       ("股票池", "資格只用 t 以前資訊；下市標的不在樣本", "FAIL", RED),
       ("代號", "100 檔在 2006–2023 無代號變更", "PASS", TEAL)]
text(s, 6.9, 4.0, 6, 0.45, [[("時間點正確性（三層）", {"bold": True, "size": 17})]])
for i, (lv, desc, res, col) in enumerate(pit):
    y = 4.6 + i * 0.68
    box(s, 6.9, y, 1.0, 0.5, fill=col)
    text(s, 6.9, y, 1.0, 0.5, [[(res, {"bold": True, "color": WHITE, "size": 13})]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 8.05, y - 0.02, 4.8, 0.6, [[(lv + "　", {"bold": True, "size": 14}), (desc, {"size": 14})]], anchor=MSO_ANCHOR.MIDDLE)
footer(s, 4)
s.notes_slide.notes_text_frame.text = ("資料品質報告的重點：沒有 OHLC 違反與重複；158 列零成交量直接排除。"
    "我發現 Yahoo 的 Close 其實已做分割調整，所以用分割事件重建真正成交的價格，所有美元量都用它。"
    "股票池那一格是 FAIL：名單是 2025 年選的，有存活者偏誤，報告裡有討論方向。")

# ---------------------------------------------------------------- 5 triage
s = new(); title(s, "特徵：22 個候選、五道關卡，大多是 REVISE", "篩選只用 2008–2014；Benjamini–Hochberg（α = 0.05）下沒有任何特徵顯著")
for i, (lab, col) in enumerate([("PROCEED", TEAL), ("REVISE", AMBER), ("STOP", RED)]):
    n = int(counts.get(lab, 0))
    text(s, 0.6 + i * 2.25, 1.95, 2.1, 1.0, [str(n)], size=60, bold=True, color=col)
    text(s, 0.6 + i * 2.25, 3.05, 2.1, 0.4, [lab], size=15, bold=True, color=MUTED)
gates = ["正確性：截斷測試＋時間位移安慰劑", "關聯：逐年 IC、NW 區間、IC 衰減", "形狀：五分位單調性", "可行性：價差 > 2× 成本", "多重檢定：BH 校正"]
for i, g in enumerate(gates):
    badge(s, 0.6, 3.75 + i * 0.6, str(i + 1), d=0.45, size=13)
    text(s, 1.2, 3.77 + i * 0.6, 5.6, 0.45, [g], size=14, anchor=MSO_ANCHOR.MIDDLE)
box(s, 7.3, 1.95, 5.45, 4.65, fill=TINT)
b = tri.set_index("feature").loc["beta_spy_126"]
text(s, 7.6, 2.15, 4.9, 4.3, [[("最強的特徵被我 STOP 了", {"bold": True, "size": 18, "color": RED})],
     f"beta_spy_126：IC {b.ic_mean:.3f}　[{b.ic_lo:.3f}, {b.ic_hi:.3f}]",
     "NW 區間不含 0（BH 後不顯著），但方向和「betting-against-beta」機制相反。",
     "看了資料才把方向反過來，就等於一個新假說 → 留給 v2。",
     [("機制檢驗（discovery）：", {"bold": True}), ("高波動三分位的動能 IC 並沒有比較低 → 開關未獲支持。", {})]], size=15, spacing=10)
footer(s, 5)
s.notes_slide.notes_text_frame.text = ("每個特徵都有可命名的機制與預期方向。五道關卡下 0 個 PROCEED、9 個 REVISE、8 個 STOP，"
    "因為 discovery 期只有約 86 個有效 IC 天，檢定力很低。最強的 beta 特徵方向與機制相反，所以 STOP；"
    "這是避免看結果再改假說。")

# ---------------------------------------------------------------- 6 model ladder (native chart)
s = new(); title(s, "模型階梯：只在成對檢定顯著時才往上爬", "驗證 2015–2023，日 Spearman IC（Newey–West 95% CI，lag 21）")
cd = CategoryChartData()
cd.categories = ["12-1 規則", "Ridge", "LightGBM", "集成"]
cd.add_series("平均日 IC", [round(val_ic.loc[k, "IC"], 3) for k in ("rule", "ridge", "lgbm", "ensemble")])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.9), Inches(6.6), Inches(4.7), cd)
ch = gf.chart; ch.has_legend = False
pl = ch.plots[0]; pl.has_data_labels = True; pl.gap_width = 70
dl = pl.data_labels; dl.number_format = "0.000"; dl.number_format_is_linked = False; dl.position = XL_LABEL_POSITION.OUTSIDE_END
dl.font.size = Pt(14); dl.font.color.rgb = INK
for i, pt in enumerate(pl.series[0].points):
    pt.format.fill.solid(); pt.format.fill.fore_color.rgb = AMBER if i == 2 else RGBColor(0xA9, 0xB8, 0xBC)
ch.value_axis.has_major_gridlines = True
ch.value_axis.major_gridlines.format.line.color.rgb = RGBColor(0xE2, 0xE6, 0xE8)
ch.value_axis.tick_labels.font.size = Pt(12); ch.category_axis.tick_labels.font.size = Pt(14)
ch.value_axis.tick_labels.font.color.rgb = MUTED; ch.category_axis.tick_labels.font.color.rgb = INK
ch.value_axis.format.line.fill.background()
rows = [("規則", val_ic.loc["rule"]), ("Ridge", val_ic.loc["ridge"]), ("LightGBM", val_ic.loc["lgbm"]), ("集成", val_ic.loc["ensemble"])]
text(s, 7.6, 1.95, 5.2, 0.45, [[("95% 信賴區間", {"bold": True, "size": 17})]])
for i, (k, r) in enumerate(rows):
    text(s, 7.6, 2.5 + i * 0.5, 5.2, 0.45, [[(f"{k}", {"bold": True, "size": 15}), (f"　[{r.CI_lo:.3f}, {r.CI_hi:.3f}]", {"size": 15, "color": RED if r.CI_lo <= 0 else INK})]])
p1, p2 = paired.loc["lgbm - ridge"], paired.loc["ensemble - lgbm"]
box(s, 7.6, 4.65, 5.15, 1.95, fill=TINT)
text(s, 7.85, 4.8, 4.7, 1.7, [[("部署規則（事先登錄）", {"bold": True, "size": 15, "color": TEAL})],
     f"LightGBM − Ridge：下界 {p1['lo']:+.4f} > 0 → 部署 LightGBM",
     f"集成 − LightGBM：下界 {p2['lo']:+.4f} → 不採用",
     [("邊際很薄，而且是修正問題後才成立", {"italic": True, "color": RED})]], size=14, spacing=5)
footer(s, 6)
s.notes_slide.notes_text_frame.text = ("規則的 IC 信賴區間含 0；LightGBM 0.078，區間不含 0。部署規則是：相對『最佳的較簡單階』成對差的下界大於 0 才往上爬。"
    "LightGBM 對 Ridge 的下界只有 +0.0026，而且修正前因為比較對象寫錯、調參窗沒 purge，這個值是 −0.0015。這件事我在報告裡完整揭露。")

# ---------------------------------------------------------------- 7 headline table
s = new(); title(s, "總表：驗證期與 holdout 並列", f"確認檢定（IC > 0 且淨 Sharpe > 0）：{'通過' if ho_inf['confirmatory_test_passed'] else '未通過'}，但兩個信賴區間都含 0")
rows_keep = [0, 1, 2, 3, 4, 5, 8]
tbl = s.shapes.add_table(len(rows_keep) + 1, 4, Inches(0.6), Inches(1.95), Inches(12.1), Inches(4.4)).table
widths = [1.6, 3.9, 3.3, 3.3]
for j, w in enumerate(widths):
    tbl.columns[j].width = Inches(w)
hdr = ["階段", "指標", "驗證 2015–2023", "Holdout 2024–2025"]
for j, h in enumerate(hdr):
    c = tbl.cell(0, j); c.fill.solid(); c.fill.fore_color.rgb = INK
    c.text_frame.paragraphs[0].runs or c.text_frame.paragraphs[0].add_run()
    r = c.text_frame.paragraphs[0].runs[0]; r.text = h; font(r, 15, True, WHITE)
for i, ri in enumerate(rows_keep, start=1):
    vals = head.iloc[ri][["Stage", "Metric", "Validation 2015-2023", "Holdout 2024-2025"]].tolist()
    for j, v in enumerate(vals):
        c = tbl.cell(i, j); c.fill.solid(); c.fill.fore_color.rgb = TINT if i % 2 else WHITE
        p = c.text_frame.paragraphs[0]; r = p.add_run(); r.text = str(v)
        font(r, 14, j >= 2, INK)
        c.margin_top = c.margin_bottom = Inches(0.06)
text(s, 0.6, 6.45, 12.1, 0.4, [f"基準（holdout 淨 Sharpe）：12-1 規則 {ho_perf.loc['rule_mom_252_21', 'net_sharpe']:.2f} · 等權 {ho_perf.loc['EW_all_eligible', 'net_sharpe']:.2f} · 60/40 {ho_perf.loc['60/40 SPY/AGG', 'net_sharpe']:.2f} · SPY {ho_perf.loc['SPY', 'net_sharpe']:.2f}"], size=14, color=MUTED)
footer(s, 7)
s.notes_slide.notes_text_frame.text = ("總表依四個關卡加統計推論排列。holdout 平均 IC 0.064、淨 Sharpe 1.08，確認檢定通過；"
    "但 IC 區間 [−0.014, 0.142]、Sharpe 區間 [−0.31, 2.47] 都含 0，兩年日資料的 Sharpe 標準誤大約 ±0.7。"
    "DSR 在 93 次試驗下只有 0.17。最下面那行很重要：holdout 裡規則、等權、60/40 都比部署模型好。")

# ---------------------------------------------------------------- 8 equity figure
s = new(); title(s, "扣成本後：驗證期領先，holdout 中規則縮小差距", "圖 1　淨值曲線（log），黑線為驗證／holdout 分界")
s.shapes.add_picture(str(FIG / "07_fig1_equity.png"), Inches(0.6), Inches(1.85), width=Inches(8.6))
box(s, 9.5, 1.95, 3.25, 4.6, fill=TINT)
text(s, 9.75, 2.15, 2.8, 4.3, [[("怎麼讀", {"bold": True, "size": 17, "color": TEAL})],
     f"驗證期淨 Sharpe：部署 {val_perf.loc['deployed_lgbm', 'net_sharpe']:.2f}、規則 {val_perf.loc['rule_mom_252_21', 'net_sharpe']:.2f}、60/40 {val_perf.loc['60/40 SPY/AGG', 'net_sharpe']:.2f}",
     "整段累積 +219% vs 規則 +190%；但 holdout 規則 +56% vs 部署 +39%。",
     f"對 SPY 的 beta ≈ 0.90：大部分報酬是美股多頭。"], size=15, spacing=10)
footer(s, 8)
s.notes_slide.notes_text_frame.text = ("這張圖回答：部署模型扣成本後有沒有贏？驗證期贏規則與等權，但輸 60/40；holdout 被規則追上。"
    "加上 10% 波動目標的虛線，回撤變小但 Sharpe 也變低，覆蓋只是精煉、不能救策略。")

# ---------------------------------------------------------------- 9 rolling IC + drawdown
s = new(); title(s, "穩定性與路徑風險", "圖 2　63 日滾動 IC　·　圖 6　回撤與停損階梯（5 / 10 / 15 / 20%）")
s.shapes.add_picture(str(FIG / "07_fig2_rolling_ic.png"), Inches(0.6), Inches(1.85), width=Inches(6.0))
s.shapes.add_picture(str(FIG / "07_fig6_drawdown.png"), Inches(6.75), Inches(1.85), width=Inches(6.0))
text(s, 0.6, 4.25, 6.0, 2.4, [[("訊號會翻號", {"bold": True, "size": 16, "color": TEAL})],
     "2016、2018、2022、2023 有較長的負區段；holdout 大致為正，2025 年例外。",
     [("機制在驗證期得到支持：", {"bold": True}), (f"平靜 IC {mech.loc['calm', 'mean']:.3f}，壓力 {round(mech.loc['stressed', 'mean'] - 1e-12, 3):.3f}", {})]], size=14, spacing=8)
text(s, 6.75, 4.25, 6.0, 2.4, [[("停損階梯什麼時候觸發", {"bold": True, "size": 16, "color": TEAL})],
     "20% 平倉只在 2020 年 3 月觸發（最大回撤 −29.9%，114 日回到前高）。",
     "holdout 最深 −19.9%（2025 年 4 月），差一點觸發。"], size=14, spacing=8)
footer(s, 9)
s.notes_slide.notes_text_frame.text = ("左圖：IC 的平均值會掩蓋翻號，所以看滾動 IC。驗證期的壓力三分位動能 IC 是 −0.075，支持波動開關；"
    "但 discovery 期相反，所以我的結論是『未定』。右圖：回撤加上停損門檻線。")

# ---------------------------------------------------------------- 10 diagnosis + next step
s = new(dark=True); title(s, "診斷：價值流失在投組轉換，不在訊號或成本", "Ch. 20 型態：never resolved（兩段淨 Sharpe 區間都跨 0）", dark=True)
gates = [("訊號", f"holdout IC {ho_inf['ic_mean']:.3f}\n仍高於規則", TEAL, "過"), ("投組", f"淨 Sharpe {ho_inf['net_sharpe']:.2f}\n< 規則 {ho_perf.loc['rule_mom_252_21', 'net_sharpe']:.2f}", RED, "敗"),
         ("成本", f"兩平 {ho_inf['breakeven_bps_per_leg']:.0f} bp/腿\n假設約 3.4 bp", TEAL, "過"), ("推論", f"DSR {ho_inf['dsr_all_trials']:.2f}\n（N = {int(ho_inf['n_trials_all'])}）", RED, "敗")]
for i, (g, d, col, mark) in enumerate(gates):
    x = 0.6 + i * 3.08
    box(s, x, 1.95, 2.88, 2.3, fill=RGBColor(0x2B, 0x39, 0x40))
    badge(s, x + 0.25, 2.15, mark, fill=col, d=0.6, size=20)
    text(s, x + 1.0, 2.2, 1.8, 0.5, [[(g, {"bold": True, "size": 20, "color": WHITE})]])
    text(s, x + 0.25, 3.0, 2.5, 1.2, d.split("\n"), size=15, color=RGBColor(0xDD, 0xE5, 0xE5), spacing=3)
text(s, 0.6, 4.55, 12.1, 0.5, [[("Portfolio-translation drift：", {"bold": True, "color": AMBER, "size": 17}),
     ("IC 穩定、Sharpe 下降。換手 14×/年、前 10 名集中，在全面多頭裡把小幅 IC 優勢吃掉。", {"color": WHITE, "size": 17})]])
box(s, 0.6, 5.3, 12.1, 1.35, fill=AMBER)
text(s, 0.9, 5.42, 11.5, 1.15, [[("唯一的下一步（setup v2）：", {"bold": True, "size": 18, "color": INK})],
     [("分數先做 3 個月指數平滑再取前 10 名，把換手壓到 ≈ 6–7×/年，用同一協定重測投組關卡；不再調模型參數，也不投入資金。", {"size": 16, "color": INK})]], spacing=4)
footer(s, 10, dark=True)
s.notes_slide.notes_text_frame.text = ("用 Chapter 20 的四個關卡看：訊號關過了、成本關過了，掉在投組關與統計推論。"
    "這是 portfolio-translation drift。下一步只有一個：降低換手、重測投組關；這是映射規則的改變，所以是 v2，需要新的 holdout。")

# ---------------------------------------------------------------- 11 backup: what was knowable
s = new(); title(s, "備用：決策當下知道什麼？", "老師的第一個問題 —— 每個時點用到的資訊")
steps = [("t 收盤", "算特徵與資格\n只用 ≤ t 的價量", TEAL), ("t+1 開盤", "成交（延遲 1 日）\n成本用未調整開盤價", AMBER),
         ("持有", "權重每日漂移\n缺價日不交易", MUTED), ("t+22 開盤", "標籤出場價\n= 下次再平衡成交", TEAL)]
for i, (h, d, col) in enumerate(steps):
    x = 0.6 + i * 3.1
    box(s, x, 2.0, 2.8, 0.75, fill=col)
    text(s, x, 2.0, 2.8, 0.75, [[(h, {"bold": True, "color": WHITE, "size": 18})]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 0.1, 2.95, 2.6, 1.0, d.split("\n"), size=15, spacing=3)
    if i < 3:
        a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + 2.83), Inches(2.22), Inches(0.24), Inches(0.3))
        a.fill.solid(); a.fill.fore_color.rgb = INK; a.line.fill.background()
box(s, 0.6, 4.35, 12.1, 2.25, fill=TINT)
text(s, 0.9, 4.55, 11.5, 2.0, [[("怎麼證明", {"bold": True, "size": 17, "color": TEAL})],
     "截斷測試：把資料截在 t 重算全部特徵，與全樣本逐值相同（6 個日期＋之後另測 12 個月底）。",
     "Purge：訓練最後一列的標籤出場日 < 驗證首日（每個 fold 間隔 23 個交易日）。",
     "Holdout：凍結規格雜湊 7e75995d042147ae；存取紀錄 touch #1，之後無再觸碰。"], size=15, spacing=8)
footer(s, 11)
s.notes_slide.notes_text_frame.text = "備用投影片：回答『決策時點可知道什麼』。"

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print("wrote", OUT, "slides:", len(prs.slides))
