#!/usr/bin/env python3
"""Render an SVG performance chart from performance.log (latest llama-server session)."""
import re, math, statistics
from collections import defaultdict

LOG = 'performance.log'
OUT = 'performance.svg'

lines = open(LOG, encoding='utf-8', errors='replace').read().splitlines()

loads = [i for i, l in enumerate(lines) if 'load_tensors: loading model tensors' in l]
sess = lines[:loads[1]]  # newest -> reload at 10:26:27

ts_re   = re.compile(r'^sep 16 (\d+):(\d+):(\d+)')
tid_re  = re.compile(r'task (\d+) \|')
ctx_re  = re.compile(r'task\.n_tokens = (\d+)')
pp_re   = re.compile(r'prompt processing, n_tokens = *(\d+), progress = *[0-9.]+, t = *([0-9.]+) s / *([0-9.]+) tokens per second')
tg_re   = re.compile(r'(?<!prompt )eval time = *([0-9.]+) ms / *(\d+) tokens \(\s*[0-9.]+ ms per token,\s*([0-9.]+) tokens per second')
da_re   = re.compile(r'draft acceptance = ([0-9.]+) \(\s*(\d+) accepted /\s*(\d+) generated\)')

def secs(l):
    m = ts_re.match(l)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) if m else None

ctx = {}
pp = defaultdict(list)   # task -> [(x, rate, t)]
tg = {}                  # task -> (x, tg, gen, t)
da = {}                  # task -> (x, acc, accepted, generated, t)

for l in sess:
    c = ctx_re.search(l)
    m = tid_re.search(l)
    if c and m:
        ctx[m.group(1)] = int(c.group(1))

for l in sess:
    m = tid_re.search(l)
    tid = m.group(1) if m else None
    p = pp_re.search(l)
    if p and tid:
        pp[tid].append((int(p.group(1)), float(p.group(3)), secs(l)))
    g = tg_re.search(l)
    if g and tid:
        tg[tid] = (ctx.get(tid), float(g.group(3)), int(g.group(2)), secs(l))
    d = da_re.search(l)
    if d and tid:
        da[tid] = (ctx.get(tid), float(d.group(1)), int(d.group(2)), int(d.group(3)), secs(l))

pp_series = {t: sorted(v) for t, v in pp.items()}
tg_pts = [v for v in tg.values() if v[0] is not None]
da_pts = [v for v in da.values() if v[0] is not None]

# ---- binned trends -------------------------------------------------------
def bin_median(points, binw, minn):
    b = defaultdict(list)
    for x, y in points:
        b[x // binw].append(y)
    return sorted((k * binw + binw / 2.0, statistics.median(v)) for k, v in b.items() if len(v) >= minn)

pp_all = [(x, r) for v in pp_series.values() for (x, r, _) in v]
pp_trend = bin_median(pp_all, 5000, 3)

bt = defaultdict(list)
for x, a, gen, _ in tg_pts:
    bt[x // 10000].append((a, gen))
tg_trend = []
for k in sorted(bt):
    v = bt[k]
    tg_trend.append((k * 10000 + 5000, sum(a * gen for a, gen in v) / sum(g for _, g in v)))

bd = defaultdict(list)
for x, a, acc, gen, _ in da_pts:
    bd[x // 10000].append((acc, gen))
da_trend = []
for k in sorted(bd):
    v = bd[k]
    da_trend.append((k * 10000 + 5000, 100.0 * sum(a for a, _ in v) / sum(g for _, g in v)))

# ---- event x positions ---------------------------------------------------
t_reload = 11 * 3600 + 31 * 60 + 15
t_end = 11 * 3600 + 39 * 60 + 19          # task 491839 release
t_eject = 11 * 3600 + 35 * 60 + 36
big = pp_series.get('491839', [])
x_reload = min(big, key=lambda z: abs(z[2] - t_reload))[0] if big else 97429
x_eject = x_reload + (106538 - x_reload) * (t_eject - t_reload) / (t_end - t_reload)

# ---- layout --------------------------------------------------------------
W, H = 1280, 800
X0, X1 = 112, 1168
P1T, P1B = 96, 502
P2T, P2B = 584, 738
XM, TGMAX, PPMAX = 120000.0, 60.0, 1200.0
DA_C, PP_C, TG_C = '#2f9e44', '#1971c2', '#e8590c'
AX_C, GRID_C, EVT_C = '#495057', '#e9ecef', '#adb5bd'

sx  = lambda x: X0 + (x / XM) * (X1 - X0)
syT = lambda v: P1B - (v / TGMAX) * (P1B - P1T)
syP = lambda v: P1B - (v / PPMAX) * (P1B - P1T)
syD = lambda v: P2B - (v / 100.0) * (P2B - P2T)

def smooth(pts):
    if len(pts) < 2:
        return ''
    d = 'M %.1f %.1f ' % pts[0]
    if len(pts) == 2:
        return d + 'L %.1f %.1f' % pts[1]
    for i in range(len(pts) - 1):
        p0 = pts[i - 1] if i > 0 else pts[0]
        p1, p2 = pts[i], pts[i + 1]
        p3 = pts[i + 2] if i + 2 < len(pts) else pts[-1]
        d += 'C %.1f %.1f %.1f %.1f %.1f %.1f ' % (
            p1[0] + (p2[0] - p0[0]) / 6.0, p1[1] + (p2[1] - p0[1]) / 6.0,
            p2[0] - (p3[0] - p1[0]) / 6.0, p2[1] - (p3[1] - p1[1]) / 6.0,
            p2[0], p2[1])
    return d.rstrip()

s = []
s.append('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" font-family="Helvetica,Arial,sans-serif">' % (W, H, W, H))
s.append('<rect width="%d" height="%d" fill="#ffffff"/>' % (W, H))
s.append('<text x="%d" y="30" font-size="23" font-weight="700" fill="#212529">llama-server performance vs. context length</text>' % X0)
s.append('<text x="%d" y="52" font-size="12.5" fill="%s">Qwen3.8-27B-ASCII-Condensed-DFlash2-Q2_K_S-MIX - n_ctx 160k, n_batch 256, n_ubatch 8, flash-attn</text>' % (X0, AX_C))
s.append('<text x="%d" y="70" font-size="11.5" fill="%s">latest session: Sep 16 10:26:27 - 11:51:54 - draft (DFlash2, block 8) active except after 11:35:36 - in-process model reload at 11:31:15</text>' % (X0, '#868e96'))

# draft-active shading (panel 2)
s.append('<rect x="%.1f" y="%d" width="%.1f" height="%d" fill="#ebfbee"/>' % (sx(0), P2T, sx(x_eject) - sx(0), P2B - P2T))

# horizontal grid (TG axis)
for v in range(0, 61, 10):
    y = syT(v)
    s.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s" stroke-width="1"/>' % (X0, y, X1, y, GRID_C))
# vertical grid (shared x)
for x in range(0, 120001, 20000):
    s.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s" stroke-width="1"/>' % (sx(x), P1T, sx(x), P1B, GRID_C))
    s.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s" stroke-width="1"/>' % (sx(x), P2T, sx(x), P2B, GRID_C))
# horizontal grid (DA axis)
for v in range(0, 101, 25):
    y = syD(v)
    s.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s" stroke-width="1"/>' % (X0, y, X1, y, GRID_C))

# event lines
for xv, lab, dy, anch, off in ((x_reload, 'model reload 11:31', 13, 'end', -5), (x_eject, 'draft ejected 11:35', 30, 'start', 5)):
    s.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s" stroke-width="1.4" stroke-dasharray="5 4"/>' % (sx(xv), P1T, sx(xv), P2B, EVT_C))
    s.append('<text x="%.1f" y="%d" font-size="11" fill="#868e96" text-anchor="%s">%s</text>' % (sx(xv) + off, P1T + dy, anch, lab))

# PP raw segments + trend
for t, v in pp_series.items():
    pts = ' '.join('%.1f,%.1f' % (sx(x), syP(r)) for x, r, _ in v)
    s.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="1.5" opacity="0.30"/>' % (pts, PP_C))
s.append('<path d="%s" fill="none" stroke="%s" stroke-width="3.4" stroke-linejoin="round"/>' % (smooth([(sx(x), syP(v)) for x, v in pp_trend]), PP_C))

# TG points + trend
gmax = max(g for _, _, g, _ in tg_pts)
for x, a, gen, _ in tg_pts:
    r = 3.0 + 4.0 * math.sqrt(gen / gmax)
    s.append('<circle cx="%.1f" cy="%.1f" r="%.2f" fill="%s" opacity="0.75"/>' % (sx(x), syT(a), r, TG_C))
s.append('<path d="%s" fill="none" stroke="%s" stroke-width="3.4" stroke-linejoin="round"/>' % (smooth([(sx(x), syT(v)) for x, v in tg_trend]), TG_C))

# DA points + trend
amax = max(c for _, _, _, c, _ in da_pts)
for x, a, acc, gen, _ in da_pts:
    r = 3.0 + 4.0 * math.sqrt(gen / amax)
    s.append('<circle cx="%.1f" cy="%.1f" r="%.2f" fill="%s" opacity="0.8"/>' % (sx(x), syD(acc * 100), r, DA_C))
s.append('<path d="%s" fill="none" stroke="%s" stroke-width="3.4" stroke-linejoin="round"/>' % (smooth([(sx(x), syD(v)) for x, v in da_trend]), DA_C))

# axes
s.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="1.5"/>' % (X0, P1T, X0, P1B, AX_C))
s.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="1.5"/>' % (X1, P1T, X1, P1B, AX_C))
s.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="1.5"/>' % (X0, P2B, X1, P2B, AX_C))

# y labels
for v in range(0, 61, 10):
    s.append('<text x="%d" y="%.1f" font-size="12" fill="%s" text-anchor="end">%d</text>' % (X0 - 8, syT(v) + 4, TG_C, v))
for v in range(0, 1201, 200):
    s.append('<text x="%d" y="%.1f" font-size="12" fill="%s" text-anchor="start">%d</text>' % (X1 + 8, syP(v) + 4, PP_C, v))
for v in range(0, 101, 25):
    s.append('<text x="%d" y="%.1f" font-size="12" fill="%s" text-anchor="end">%d</text>' % (X0 - 8, syD(v) + 4, DA_C, v))
# x labels
for x in range(0, 120001, 20000):
    lab = '0' if x == 0 else '%dk' % (x // 1000)
    s.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s" stroke-width="1.5"/>' % (sx(x), P2B, sx(x), P2B + 6, AX_C))
    s.append('<text x="%.1f" y="%d" font-size="12" fill="%s" text-anchor="middle">%s</text>' % (sx(x), P2B + 22, AX_C, lab))

# axis titles
s.append('<text x="%d" y="%.1f" font-size="13" font-weight="600" fill="%s" text-anchor="middle" transform="rotate(-90 %d %.1f)">decode / TG (tokens/s)</text>' % (X0 - 52, (P1T + P1B) / 2, TG_C, X0 - 52, (P1T + P1B) / 2))
s.append('<text x="%d" y="%.1f" font-size="13" font-weight="600" fill="%s" text-anchor="middle" transform="rotate(90 %d %.1f)">prefill / PP (tokens/s)</text>' % (X1 + 56, (P1T + P1B) / 2, PP_C, X1 + 56, (P1T + P1B) / 2))
s.append('<text x="%d" y="%.1f" font-size="13" font-weight="600" fill="%s" text-anchor="middle" transform="rotate(-90 %d %.1f)">draft acceptance (%%)</text>' % (X0 - 52, (P2T + P2B) / 2, DA_C, X0 - 52, (P2T + P2B) / 2))
s.append('<text x="%.1f" y="%d" font-size="13.5" font-weight="600" fill="%s" text-anchor="middle">context length (tokens)</text>' % ((X0 + X1) / 2, H - 12, AX_C))

# legend
lx, ly = X1 - 300, P1B - 92
s.append('<rect x="%d" y="%d" width="290" height="86" rx="6" fill="#ffffff" fill-opacity="0.94" stroke="#ced4da"/>' % (lx, ly))
def leg(y, color, label, dot=False):
    if dot:
        s.append('<circle cx="%d" cy="%d" r="4.5" fill="%s"/>' % (lx + 18, y - 4, color))
    else:
        s.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="3.2"/>' % (lx + 8, y - 4, lx + 30, y - 4, color))
    s.append('<text x="%d" y="%.1f" font-size="12.5" fill="#343a40">%s</text>' % (lx + 40, y, label))
leg(ly + 22, TG_C, 'decode TG/s (dots: gen tokens)', True)
leg(ly + 44, PP_C, 'prefill PP/s (bold: trend)')
leg(ly + 66, DA_C, 'draft acceptance (lower panel)', True)

# callouts
s.append('<text x="%.1f" y="%.1f" font-size="12" font-weight="700" fill="%s" text-anchor="middle">%.1f t/s</text>' % (sx(0) + 40, syT(49.79) + 4, TG_C, 49.79))
s.append('<text x="%.1f" y="%.1f" font-size="12" font-weight="700" fill="%s" text-anchor="end">%.1f t/s</text>' % (sx(118272) - 4, syT(17.53) - 10, TG_C, 17.53))

s.append('</svg>')
open(OUT, 'w', encoding='utf-8').write('\n'.join(s))
print('wrote', OUT, len('\n'.join(s)), 'bytes')
print('session tasks=%d  PP segments=%d  TG pts=%d  DA pts=%d' % (len(ctx), len(pp_series), len(tg_pts), len(da_pts)))
print('PP trend pts=%d  TG trend pts=%d  DA trend pts=%d' % (len(pp_trend), len(tg_trend), len(da_trend)))
print('x_reload=%.0f  x_eject=%.0f' % (x_reload, x_eject))
