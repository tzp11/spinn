#!/usr/bin/env python3
"""
Parse PROJECT_OPTIMIZATION_SUMMARY.md and generate academic-style artifacts:

- CSV tables
- Markdown / LaTeX tables
- SVG figures for slides/papers

No third-party dependencies required.
"""

from __future__ import annotations

import csv
import html
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "PROJECT_OPTIMIZATION_SUMMARY.md"
OUT_DIR = ROOT / "report_assets" / "optimization"


@dataclass
class Step:
    index: int
    title: str
    before_ms: float | None
    after_ms: float | None
    claimed_speedup: float | None
    note: str
    category: str

    @property
    def measured_speedup(self) -> float | None:
        if self.before_ms and self.after_ms and self.after_ms > 0:
            return self.before_ms / self.after_ms
        return None

    @property
    def reduction_pct(self) -> float | None:
        if self.before_ms and self.after_ms and self.before_ms > 0:
            return (1.0 - self.after_ms / self.before_ms) * 100.0
        return None


@dataclass
class ComparisonRow:
    system: str
    single_thread_ms: float
    multi_thread_ms_low: float
    multi_thread_ms_high: float


STEP_RE = re.compile(
    r"\*\*优化\s*(\d+):\s*(.*?)\s*\((.*?)\)\*\*"
)
ARROW_RE = re.compile(
    r"(?:(?:多核时间由)\s*)?([0-9.]+)\s*ms\s*➔\s*([0-9.]+)\s*ms"
)
SPEEDUP_RE = re.compile(r"约\s*([0-9.]+)\s*x")
FINAL_ST_RE = re.compile(r"单核（1T）推理耗时从约\s*([0-9.]+)\s*ms\s*降至\s*([0-9.]+)\s*ms")
FINAL_MT_RE = re.compile(r"四核（4T）多线程推理耗时降低至\s*([0-9.]+)\s*ms")
def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def infer_category(step_index: int) -> str:
    if step_index in (1, 2):
        return "Memory / I-O"
    if step_index in (3, 4):
        return "Graph / Algorithm"
    if step_index in (5, 6, 7):
        return "Kernel / SIMD"
    if step_index in (8, 9, 10):
        return "Parallel / Deployment"
    return "Other"


def parse_steps(text: str) -> list[Step]:
    steps: list[Step] = []
    for match in STEP_RE.finditer(text):
        idx = int(match.group(1))
        title = " ".join(match.group(2).split())
        paren = match.group(3)

        before_ms = None
        after_ms = None
        claimed_speedup = None

        arrow_match = ARROW_RE.search(paren)
        if arrow_match:
            before_ms = float(arrow_match.group(1))
            after_ms = float(arrow_match.group(2))

        speedup_match = SPEEDUP_RE.search(paren)
        if speedup_match:
            claimed_speedup = float(speedup_match.group(1))

        note = paren
        steps.append(
            Step(
                index=idx,
                title=title,
                before_ms=before_ms,
                after_ms=after_ms,
                claimed_speedup=claimed_speedup,
                note=note,
                category=infer_category(idx),
            )
        )
    return steps


def parse_summary(text: str) -> tuple[float, float, float]:
    st_match = FINAL_ST_RE.search(text)
    mt_match = FINAL_MT_RE.search(text)
    if not st_match or not mt_match:
        raise ValueError("Failed to parse summary latency numbers from markdown.")
    initial_st = float(st_match.group(1))
    final_st = float(st_match.group(2))
    final_mt = float(mt_match.group(1))
    return initial_st, final_st, final_mt


def parse_comparison(text: str) -> list[ComparisonRow]:
    rows: list[ComparisonRow] = []
    for line in text.splitlines():
        if "| **" not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        system = cells[0].replace("**", "").strip()
        st_match = re.search(r"([0-9.]+)\s*ms", cells[1])
        mt_range_match = re.search(r"([0-9.]+)\s*~\s*([0-9.]+)\s*ms", cells[3])
        mt_single_match = re.search(r"([0-9.]+)\s*ms", cells[3])
        if not st_match:
            continue
        if mt_range_match:
            mt_low = float(mt_range_match.group(1))
            mt_high = float(mt_range_match.group(2))
        elif mt_single_match:
            mt_low = float(mt_single_match.group(1))
            mt_high = mt_low
        else:
            continue
        rows.append(
            ComparisonRow(
                system=system,
                single_thread_ms=float(st_match.group(1)),
                multi_thread_ms_low=mt_low,
                multi_thread_ms_high=mt_high,
            )
        )
    return rows


def fmt(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def write_csv(steps: list[Step], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "category",
                "title",
                "before_ms",
                "after_ms",
                "measured_speedup_x",
                "claimed_speedup_x",
                "reduction_pct",
                "note",
            ]
        )
        for step in steps:
            writer.writerow(
                [
                    step.index,
                    step.category,
                    step.title,
                    step.before_ms,
                    step.after_ms,
                    step.measured_speedup,
                    step.claimed_speedup,
                    step.reduction_pct,
                    step.note,
                ]
            )


def write_markdown_table(steps: list[Step], path: Path) -> None:
    lines = [
        "# SPINN Optimization Steps",
        "",
        "| Step | Category | Optimization | Before (ms) | After (ms) | Speedup (x) | Reduction (%) |",
        "|:---:|:---|:---|---:|---:|---:|---:|",
    ]
    for step in steps:
        lines.append(
            "| {idx} | {cat} | {title} | {before} | {after} | {speedup} | {reduction} |".format(
                idx=step.index,
                cat=step.category,
                title=step.title.replace("|", "\\|"),
                before=fmt(step.before_ms, 1),
                after=fmt(step.after_ms, 1),
                speedup=fmt(step.measured_speedup, 2),
                reduction=fmt(step.reduction_pct, 1),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def write_latex_table(steps: list[Step], path: Path) -> None:
    lines = [
        "% Requires: \\usepackage{booktabs}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Performance optimization stages of SPINN on ResNet-101.}",
        "\\label{tab:spinn_optimization_stages}",
        "\\begin{tabular}{c l p{5.8cm} r r r r}",
        "\\toprule",
        "Step & Category & Optimization & Before (ms) & After (ms) & Speedup ($\\times$) & Reduction (\\%) \\\\",
        "\\midrule",
    ]
    for step in steps:
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} \\\\".format(
                step.index,
                latex_escape(step.category),
                latex_escape(step.title),
                fmt(step.before_ms, 1),
                fmt(step.after_ms, 1),
                fmt(step.measured_speedup, 2),
                fmt(step.reduction_pct, 1),
            )
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_comparison_latex(rows: list[ComparisonRow], path: Path) -> None:
    lines = [
        "% Requires: \\usepackage{booktabs}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Latency comparison between SPINN and ONNX Runtime on ResNet-101.}",
        "\\label{tab:spinn_ort_comparison}",
        "\\begin{tabular}{l r c}",
        "\\toprule",
        "System & 1T latency (ms) & 4T/8T latency (ms) \\\\",
        "\\midrule",
    ]
    for row in rows:
        mt = f"{row.multi_thread_ms_low:.1f}--{row.multi_thread_ms_high:.1f}"
        lines.append(
            "{} & {:.1f} & {} \\\\".format(
                latex_escape(row.system), row.single_thread_ms, mt
            )
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        '.title{font:700 22px "Times New Roman", serif; fill:#111827;}',
        '.subtitle{font:14px "Times New Roman", serif; fill:#4b5563;}',
        '.axis{stroke:#374151; stroke-width:1.2;}',
        '.grid{stroke:#d1d5db; stroke-width:1; stroke-dasharray:4 4;}',
        '.label{font:14px "Times New Roman", serif; fill:#111827;}',
        '.small{font:12px "Times New Roman", serif; fill:#374151;}',
        '.legend{font:13px "Times New Roman", serif; fill:#111827;}',
        '</style>',
    ]


def write_step_latency_svg(steps: list[Step], path: Path) -> None:
    valid = [s for s in steps if s.after_ms is not None]
    width, height = 1400, 760
    left, right, top, bottom = 95, 55, 90, 95
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_y = max((s.before_ms or 0) for s in valid)
    max_y = max(max_y, max((s.after_ms or 0) for s in valid))
    ticks = [0, 4000, 8000, 12000, 16000]
    max_tick = max(ticks)

    def x_pos(i: int) -> float:
        if len(valid) == 1:
            return left + plot_w / 2
        return left + i * (plot_w / (len(valid) - 1))

    def y_pos(v: float) -> float:
        return top + plot_h - (v / max_tick) * plot_h

    lines = svg_header(width, height)
    lines += [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        f'<text class="title" x="{left}" y="40">SPINN 在 ResNet-101 上的优化轨迹图</text>',
        f'<text class="subtitle" x="{left}" y="64">时延数据自动解析自 PROJECT_OPTIMIZATION_SUMMARY.md</text>',
    ]

    for tick in ticks:
        y = y_pos(tick)
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>')
        lines.append(f'<text class="small" x="{left-12}" y="{y+4:.1f}" text-anchor="end">{tick}</text>')

    lines.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>')
    lines.append(f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>')

    before_points = []
    after_points = []
    for i, step in enumerate(valid):
        x = x_pos(i)
        before_points.append(f"{x:.1f},{y_pos(step.before_ms):.1f}")
        after_points.append(f"{x:.1f},{y_pos(step.after_ms):.1f}")
        lines.append(f'<line x1="{x:.1f}" y1="{y_pos(step.before_ms):.1f}" x2="{x:.1f}" y2="{y_pos(step.after_ms):.1f}" stroke="#9ca3af" stroke-width="1"/>')
        lines.append(f'<text class="small" x="{x:.1f}" y="{height-bottom+22}" text-anchor="middle">S{step.index}</text>')
        category_label = {
            "Memory": "访存层",
            "Graph": "图层",
            "Kernel": "内核层",
            "Parallel": "并行层",
        }.get(step.category.split("/")[0].strip(), step.category)
        lines.append(f'<text class="small" x="{x:.1f}" y="{height-bottom+38}" text-anchor="middle">{html.escape(category_label)}</text>')

    lines.append(f'<polyline fill="none" stroke="#94a3b8" stroke-width="3" points="{" ".join(before_points)}"/>')
    lines.append(f'<polyline fill="none" stroke="#1d4ed8" stroke-width="4" points="{" ".join(after_points)}"/>')

    for i, step in enumerate(valid):
        x = x_pos(i)
        yb = y_pos(step.before_ms)
        ya = y_pos(step.after_ms)
        lines.append(f'<circle cx="{x:.1f}" cy="{yb:.1f}" r="4.5" fill="#94a3b8"/>')
        lines.append(f'<circle cx="{x:.1f}" cy="{ya:.1f}" r="5.5" fill="#1d4ed8"/>')
        lines.append(f'<text class="small" x="{x:.1f}" y="{ya-10:.1f}" text-anchor="middle">{step.after_ms:.0f}</text>')

    legend_x = width - right - 210
    legend_y = top + 8
    lines.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+30}" y2="{legend_y}" stroke="#94a3b8" stroke-width="3"/>')
    lines.append(f'<text class="legend" x="{legend_x+40}" y="{legend_y+4}">该阶段优化前</text>')
    lines.append(f'<line x1="{legend_x}" y1="{legend_y+24}" x2="{legend_x+30}" y2="{legend_y+24}" stroke="#1d4ed8" stroke-width="4"/>')
    lines.append(f'<text class="legend" x="{legend_x+40}" y="{legend_y+28}">该阶段优化后</text>')

    lines.append(f'<text class="label" transform="translate(24 {top + plot_h/2:.1f}) rotate(-90)">推理时延 (ms)</text>')
    lines.append(f'<text class="label" x="{left + plot_w/2:.1f}" y="{height-20}" text-anchor="middle">优化阶段</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_waterfall_svg(steps: list[Step], initial_ms: float, final_ms: float, path: Path) -> None:
    valid = [s for s in steps if s.before_ms is not None and s.after_ms is not None and s.index <= 9]
    width, height = 1500, 760
    left, right, top, bottom = 90, 50, 90, 110
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_y = max(initial_ms, *(s.before_ms for s in valid))

    def y_pos(v: float) -> float:
        return top + plot_h - (v / max_y) * plot_h

    bar_w = plot_w / (len(valid) + 2) * 0.58
    gap = plot_w / (len(valid) + 2)

    lines = svg_header(width, height)
    lines += [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        f'<text class="title" x="{left}" y="40">SPINN 优化过程的时延下降瀑布图</text>',
        f'<text class="subtitle" x="{left}" y="64">ResNet-101 单线程推理时延</text>',
    ]

    ticks = [0, 4000, 8000, 12000, 16000]
    for tick in ticks:
        y = y_pos(tick)
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>')
        lines.append(f'<text class="small" x="{left-10}" y="{y+4:.1f}" text-anchor="end">{tick}</text>')

    lines.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>')
    lines.append(f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>')

    current = initial_ms
    x = left + gap * 0.75
    # initial bar
    lines.append(
        f'<rect x="{x:.1f}" y="{y_pos(initial_ms):.1f}" width="{bar_w:.1f}" height="{height-bottom-y_pos(initial_ms):.1f}" fill="#0f766e"/>'
    )
    lines.append(f'<text class="small" x="{x+bar_w/2:.1f}" y="{height-bottom+24}" text-anchor="middle">初始</text>')
    lines.append(f'<text class="small" x="{x+bar_w/2:.1f}" y="{y_pos(initial_ms)-10:.1f}" text-anchor="middle">{initial_ms:.0f}</text>')

    for i, step in enumerate(valid, start=1):
        next_val = step.after_ms
        drop = current - next_val
        base_y = y_pos(current)
        top_y = y_pos(next_val)
        x = left + gap * (i + 0.75)
        lines.append(
            f'<rect x="{x:.1f}" y="{top_y:.1f}" width="{bar_w:.1f}" height="{base_y-top_y:.1f}" fill="#2563eb"/>'
        )
        lines.append(
            f'<line x1="{x-bar_w*0.25:.1f}" y1="{base_y:.1f}" x2="{x+bar_w:.1f}" y2="{base_y:.1f}" stroke="#6b7280" stroke-dasharray="3 3"/>'
        )
        lines.append(f'<text class="small" x="{x+bar_w/2:.1f}" y="{height-bottom+22}" text-anchor="middle">S{step.index}</text>')
        lines.append(f'<text class="small" x="{x+bar_w/2:.1f}" y="{height-bottom+38}" text-anchor="middle">下降 {drop:.0f}</text>')
        current = next_val

    x = left + gap * (len(valid) + 1.75)
    lines.append(
        f'<rect x="{x:.1f}" y="{y_pos(final_ms):.1f}" width="{bar_w:.1f}" height="{height-bottom-y_pos(final_ms):.1f}" fill="#b91c1c"/>'
    )
    lines.append(f'<text class="small" x="{x+bar_w/2:.1f}" y="{height-bottom+24}" text-anchor="middle">最终</text>')
    lines.append(f'<text class="small" x="{x+bar_w/2:.1f}" y="{y_pos(final_ms)-10:.1f}" text-anchor="middle">{final_ms:.0f}</text>')

    overall = initial_ms / final_ms
    lines.append(f'<text class="label" x="{width-right-210}" y="{top+20}">总加速比: {overall:.1f}x</text>')
    lines.append(f'<text class="label" transform="translate(22 {top + plot_h/2:.1f}) rotate(-90)">推理时延 (ms)</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_comparison_svg(rows: list[ComparisonRow], path: Path) -> None:
    width, height = 1100, 620
    left, right, top, bottom = 95, 55, 90, 85
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_y = max(max(r.single_thread_ms, r.multi_thread_ms_high) for r in rows) * 1.2

    def y_pos(v: float) -> float:
        return top + plot_h - (v / max_y) * plot_h

    ticks = [0, 50, 100, 150, 200, 250]
    lines = svg_header(width, height)
    lines += [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        f'<text class="title" x="{left}" y="40">SPINN 与 ONNX Runtime 的时延对比</text>',
        f'<text class="subtitle" x="{left}" y="64">ResNet-101, Intel Core i7-9700K</text>',
    ]
    for tick in ticks:
        y = y_pos(tick)
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>')
        lines.append(f'<text class="small" x="{left-10}" y="{y+4:.1f}" text-anchor="end">{tick}</text>')
    lines.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>')
    lines.append(f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>')

    group_w = plot_w / len(rows)
    bar_w = group_w * 0.22
    colors = ["#1d4ed8", "#0f766e"]
    for i, row in enumerate(rows):
        group_x = left + i * group_w + group_w * 0.25
        st_x = group_x
        mt_x = group_x + bar_w * 1.8
        st_h = height - bottom - y_pos(row.single_thread_ms)
        mt_mid = (row.multi_thread_ms_low + row.multi_thread_ms_high) / 2.0
        mt_h = height - bottom - y_pos(mt_mid)
        lines.append(f'<rect x="{st_x:.1f}" y="{y_pos(row.single_thread_ms):.1f}" width="{bar_w:.1f}" height="{st_h:.1f}" fill="{colors[0]}"/>')
        lines.append(f'<rect x="{mt_x:.1f}" y="{y_pos(mt_mid):.1f}" width="{bar_w:.1f}" height="{mt_h:.1f}" fill="{colors[1]}"/>')
        y_low = y_pos(row.multi_thread_ms_low)
        y_high = y_pos(row.multi_thread_ms_high)
        cx = mt_x + bar_w / 2
        lines.append(f'<line x1="{cx:.1f}" y1="{y_low:.1f}" x2="{cx:.1f}" y2="{y_high:.1f}" stroke="#111827" stroke-width="1.4"/>')
        lines.append(f'<line x1="{cx-8:.1f}" y1="{y_low:.1f}" x2="{cx+8:.1f}" y2="{y_low:.1f}" stroke="#111827" stroke-width="1.4"/>')
        lines.append(f'<line x1="{cx-8:.1f}" y1="{y_high:.1f}" x2="{cx+8:.1f}" y2="{y_high:.1f}" stroke="#111827" stroke-width="1.4"/>')
        lines.append(f'<text class="small" x="{st_x+bar_w/2:.1f}" y="{y_pos(row.single_thread_ms)-10:.1f}" text-anchor="middle">{row.single_thread_ms:.1f}</text>')
        lines.append(f'<text class="small" x="{mt_x+bar_w/2:.1f}" y="{y_pos(mt_mid)-10:.1f}" text-anchor="middle">{row.multi_thread_ms_low:.0f}-{row.multi_thread_ms_high:.0f}</text>')
        lines.append(f'<text class="label" x="{group_x+bar_w:.1f}" y="{height-bottom+30}" text-anchor="middle">{html.escape(row.system)}</text>')

    legend_x = width - right - 230
    legend_y = top + 6
    lines.append(f'<rect x="{legend_x}" y="{legend_y-10}" width="16" height="16" fill="{colors[0]}"/>')
    lines.append(f'<text class="legend" x="{legend_x+24}" y="{legend_y+3}">单线程时延</text>')
    lines.append(f'<rect x="{legend_x}" y="{legend_y+18}" width="16" height="16" fill="{colors[1]}"/>')
    lines.append(f'<text class="legend" x="{legend_x+24}" y="{legend_y+31}">多线程时延范围</text>')
    lines.append(f'<text class="label" transform="translate(24 {top + plot_h/2:.1f}) rotate(-90)">推理时延 (ms)</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_readme(paths: dict[str, str], path: Path) -> None:
    text = f"""# Optimization Assets

This directory is generated from `PROJECT_OPTIMIZATION_SUMMARY.md`.

## Files

- `{paths['csv']}`: structured optimization-step data
- `{paths['md_table']}`: academic-style Markdown table
- `{paths['latex_table']}`: LaTeX table for paper/thesis
- `{paths['latex_comparison']}`: LaTeX comparison table
- `{paths['step_svg']}`: stage-by-stage latency trajectory
- `{paths['waterfall_svg']}`: waterfall latency reduction chart
- `{paths['comparison_svg']}`: SPINN vs ONNX Runtime comparison

## Usage

Run:

```bash
python3 scripts/visualize_optimization.py
```

Then insert the generated `.svg` figures into slides and the `.tex` tables into your thesis.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = read_text(SOURCE)
    steps = parse_steps(text)
    initial_st, final_st, _ = parse_summary(text)
    comparisons = parse_comparison(text)

    csv_path = OUT_DIR / "optimization_steps.csv"
    md_table_path = OUT_DIR / "optimization_steps.md"
    latex_table_path = OUT_DIR / "optimization_steps.tex"
    latex_comparison_path = OUT_DIR / "spinn_vs_ort.tex"
    step_svg_path = OUT_DIR / "optimization_trajectory.svg"
    waterfall_svg_path = OUT_DIR / "optimization_waterfall.svg"
    comparison_svg_path = OUT_DIR / "spinn_vs_ort.svg"
    readme_path = OUT_DIR / "README.md"

    write_csv(steps, csv_path)
    write_markdown_table(steps, md_table_path)
    write_latex_table(steps, latex_table_path)
    write_comparison_latex(comparisons, latex_comparison_path)
    write_step_latency_svg(steps, step_svg_path)
    write_waterfall_svg(steps, initial_st, final_st, waterfall_svg_path)
    write_comparison_svg(comparisons, comparison_svg_path)
    write_readme(
        {
            "csv": csv_path.name,
            "md_table": md_table_path.name,
            "latex_table": latex_table_path.name,
            "latex_comparison": latex_comparison_path.name,
            "step_svg": step_svg_path.name,
            "waterfall_svg": waterfall_svg_path.name,
            "comparison_svg": comparison_svg_path.name,
        },
        readme_path,
    )

    print(f"Generated assets in: {OUT_DIR}")
    print(f"- {csv_path.name}")
    print(f"- {md_table_path.name}")
    print(f"- {latex_table_path.name}")
    print(f"- {latex_comparison_path.name}")
    print(f"- {step_svg_path.name}")
    print(f"- {waterfall_svg_path.name}")
    print(f"- {comparison_svg_path.name}")


if __name__ == "__main__":
    main()
