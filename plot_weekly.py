#!/usr/bin/env python3
"""
plot_weekly.py - 周会汇报科研绘图 (全中文标注)
生成 4 张对比图:
  1. ONNX vs ENNF 元数据体积对比
  2. SPINN vs ORT 运行时内存对比
  3. 区间打包内存复用效率
  4. SPINN 内存组成分解（堆叠柱状图）
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# ============ 全局风格 ============
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Noto Sans CJK JP', 'Droid Sans Fallback', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'font.size': 12,
    'axes.titlesize': 15,
    'axes.titleweight': 'bold',
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# 调色板
C_BLUE = '#2E86AB'
C_ORANGE = '#E8630A'
C_GREEN = '#2CA02C'
C_RED = '#D62728'
C_PURPLE = '#9467BD'
C_GRAY = '#7F7F7F'

outdir = '/home/tzp/work/SPINN/figures'
os.makedirs(outdir, exist_ok=True)

# ============================================================
# 图1: ONNX vs ENNF 元数据/图结构体积对比 (KB)
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))

models = ['MNIST', 'complex_test', 'YOLOv10n', 'ResNet101']
onnx_meta_kb = [787/1024, 26977/1024, 88238/1024, 432222/1024]
ennf_meta_kb = [896/1024, 3664/1024, 39504/1024, 28928/1024]

x = np.arange(len(models))
w = 0.35
bars1 = ax.bar(x - w/2, onnx_meta_kb, w, label='ONNX 图结构 (Protobuf)', color=C_ORANGE, edgecolor='white', linewidth=0.8)
bars2 = ax.bar(x + w/2, ennf_meta_kb, w, label='ENNF 元数据 (二进制表)', color=C_BLUE, edgecolor='white', linewidth=0.8)

for i in range(len(models)):
    ratio = ennf_meta_kb[i] / onnx_meta_kb[i] * 100
    reduction = 100 - ratio
    ax.annotate(f'{ennf_meta_kb[i]:.1f} KB',
                xy=(x[i] + w/2, ennf_meta_kb[i]),
                xytext=(0, 5), textcoords='offset points',
                ha='center', va='bottom', fontsize=9, color=C_BLUE, fontweight='bold')
    ax.annotate(f'{onnx_meta_kb[i]:.1f} KB',
                xy=(x[i] - w/2, onnx_meta_kb[i]),
                xytext=(0, 5), textcoords='offset points',
                ha='center', va='bottom', fontsize=9, color=C_ORANGE)
    if reduction > 0:
        ax.annotate(f'缩减{reduction:.0f}%',
                    xy=(x[i], min(onnx_meta_kb[i], ennf_meta_kb[i]) * 0.5),
                    ha='center', fontsize=9, fontweight='bold', color=C_GREEN)

ax.set_ylabel('元数据体积 (KB)')
ax.set_title('ONNX vs ENNF：图结构/元数据体积对比')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(loc='upper left')
ax.set_yscale('log')
ax.set_ylim(0.5, 800)

fig.tight_layout()
fig.savefig(os.path.join(outdir, 'fig1_metadata_comparison.png'))
plt.close(fig)
print("[OK] fig1_metadata_comparison.png")

# ============================================================
# 图2: SPINN vs ORT 运行时内存对比 (MB)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={'width_ratios': [1, 3]})

# 左图: MNIST
ort_mnist = 15.07
spinn_mnist = 19188 / 1024 / 1024

bars_m = ax1.bar(['ORT', 'SPINN'], [ort_mnist, spinn_mnist],
                 color=[C_ORANGE, C_BLUE], edgecolor='white', linewidth=0.8, width=0.5)
ax1.set_ylabel('内存占用 (MB)')
ax1.set_title('MNIST (小模型)')
ax1.annotate(f'{ort_mnist:.1f} MB', xy=(0, ort_mnist), xytext=(0, 5),
             textcoords='offset points', ha='center', fontsize=11, fontweight='bold', color=C_ORANGE)
ax1.annotate(f'{spinn_mnist*1024:.1f} KB', xy=(1, spinn_mnist), xytext=(0, 5),
             textcoords='offset points', ha='center', fontsize=11, fontweight='bold', color=C_BLUE)
ax1.annotate('差距 823 倍', xy=(0.5, ort_mnist * 0.45), ha='center', fontsize=13,
             fontweight='bold', color=C_RED,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEEEE', edgecolor=C_RED, alpha=0.9))

# 右图: YOLOv10n + ResNet101
models_lr = ['YOLOv10n', 'ResNet101']
ort_mem = [62.93, 307.75]
spinn_mem = [32.36, 181.07]

x = np.arange(len(models_lr))
w = 0.3
bars1 = ax2.bar(x - w/2, ort_mem, w, label='ONNX Runtime 1.23', color=C_ORANGE, edgecolor='white', linewidth=0.8)
bars2 = ax2.bar(x + w/2, spinn_mem, w, label='SPINN (本方案)', color=C_BLUE, edgecolor='white', linewidth=0.8)

for i in range(len(models_lr)):
    saving = ort_mem[i] - spinn_mem[i]
    pct = saving / ort_mem[i] * 100
    ax2.annotate(f'{ort_mem[i]:.1f} MB', xy=(x[i] - w/2, ort_mem[i]),
                 xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10, color=C_ORANGE)
    ax2.annotate(f'{spinn_mem[i]:.1f} MB', xy=(x[i] + w/2, spinn_mem[i]),
                 xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10, color=C_BLUE)
    mid_y = (ort_mem[i] + spinn_mem[i]) / 2
    ax2.annotate(f'节省 {saving:.0f} MB ({pct:.0f}%)',
                 xy=(x[i], mid_y), ha='center', fontsize=11, fontweight='bold', color=C_GREEN,
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#EEFFEE', edgecolor=C_GREEN, alpha=0.8))

ax2.set_ylabel('运行时内存 (MB)')
ax2.set_title('中大型模型')
ax2.set_xticks(x)
ax2.set_xticklabels(models_lr)
ax2.legend()

fig.suptitle('运行时内存对比：SPINN vs ONNX Runtime', fontsize=15, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(outdir, 'fig2_memory_vs_ort.png'))
plt.close(fig)
print("[OK] fig2_memory_vs_ort.png")

# ============================================================
# 图3: 区间打包内存复用效率
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5.5))

models_3 = ['MNIST', 'YOLOv10n', 'ResNet101']
sigma_mb = [5688/1024/1024, 293.16, 153.53]
peak_mb  = [2784/1024/1024, 23.44, 11.48]
savings  = [51.1, 92.0, 92.5]

x = np.arange(len(models_3))
w = 0.3

bars_s = ax.bar(x - w/2, sigma_mb, w, label='朴素分配 (每个张量独占内存)', color=C_ORANGE, edgecolor='white', linewidth=0.8)
bars_p = ax.bar(x + w/2, peak_mb, w, label='区间打包 (Arena 峰值)', color=C_BLUE, edgecolor='white', linewidth=0.8)

for i in range(len(models_3)):
    ax.annotate(f'节省 {savings[i]:.0f}%',
                xy=(x[i] + w/2, peak_mb[i]),
                xytext=(0, 8), textcoords='offset points',
                ha='center', fontsize=12, fontweight='bold', color=C_GREEN,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#EEFFEE', edgecolor=C_GREEN, alpha=0.8))
    if sigma_mb[i] >= 1:
        ax.annotate(f'{sigma_mb[i]:.1f} MB', xy=(x[i] - w/2, sigma_mb[i]),
                    xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10, color=C_ORANGE)
        ax.annotate(f'{peak_mb[i]:.1f} MB', xy=(x[i] + w/2, peak_mb[i]),
                    xytext=(25, -5), textcoords='offset points', ha='center', fontsize=10, color=C_BLUE)

ax.set_ylabel('特征图内存 (MB)')
ax.set_title('区间打包算法的内存复用效率')
ax.set_xticks(x)
ax.set_xticklabels(models_3)
ax.legend()

fig.tight_layout()
fig.savefig(os.path.join(outdir, 'fig3_memory_reuse.png'))
plt.close(fig)
print("[OK] fig3_memory_reuse.png")

# ============================================================
# 图4: SPINN 内存组成堆叠柱状图
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5.5))

models_4 = ['MNIST', 'YOLOv10n', 'ResNet101']
infra_mb = [1388/1024/1024, 60308/1024/1024, 44456/1024/1024]
arena_mb = [2784/1024/1024, 23.44, 11.48]
weight_mb = [15016/1024/1024, 8.87, 169.54]

x = np.arange(len(models_4))
w = 0.5

b1 = ax.bar(x, weight_mb, w, label='权重缓存', color=C_PURPLE, edgecolor='white', linewidth=0.8)
b2 = ax.bar(x, arena_mb, w, bottom=weight_mb, label='Arena (特征图)', color=C_BLUE, edgecolor='white', linewidth=0.8)
bottom2 = [weight_mb[i] + arena_mb[i] for i in range(len(models_4))]
b3 = ax.bar(x, infra_mb, w, bottom=bottom2, label='运行时基础设施', color=C_GREEN, edgecolor='white', linewidth=0.8)

totals = [bottom2[i] + infra_mb[i] for i in range(len(models_4))]
for i in range(len(models_4)):
    if totals[i] >= 1:
        ax.annotate(f'共 {totals[i]:.1f} MB', xy=(x[i], totals[i]),
                    xytext=(0, 5), textcoords='offset points', ha='center', fontsize=11, fontweight='bold')
    else:
        ax.annotate(f'共 {totals[i]*1024:.1f} KB', xy=(x[i], totals[i]),
                    xytext=(0, 5), textcoords='offset points', ha='center', fontsize=11, fontweight='bold')

    w_pct = weight_mb[i] / totals[i] * 100
    a_pct = arena_mb[i] / totals[i] * 100
    if weight_mb[i] / totals[i] > 0.15:
        ax.text(x[i], weight_mb[i]/2, f'权重 {w_pct:.0f}%', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    if arena_mb[i] / totals[i] > 0.15:
        ax.text(x[i], weight_mb[i] + arena_mb[i]/2, f'特征图 {a_pct:.0f}%', ha='center', va='center', fontsize=10, color='white', fontweight='bold')

ax.set_ylabel('内存占用 (MB)')
ax.set_title('SPINN 运行时内存组成分解')
ax.set_xticks(x)
ax.set_xticklabels(models_4)
ax.legend(loc='upper left')

fig.tight_layout()
fig.savefig(os.path.join(outdir, 'fig4_memory_breakdown.png'))
plt.close(fig)
print("[OK] fig4_memory_breakdown.png")

print(f"\n所有图表已保存到 {outdir}/")
