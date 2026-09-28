"""Regenerate original, lightweight SVG illustrations; no external media."""
from pathlib import Path
import math

DATA = Path(__file__).resolve().parents[1] / "course/data"

def svg(name, title, body, height=250):
    text = f'<svg xmlns="http://www.w3.org/2000/svg" width="720" height="{height}" viewBox="0 0 720 {height}" role="img" aria-label="{title}"><rect width="720" height="{height}" fill="#f8fafc"/><g font-family="sans-serif" fill="#15233b"><text x="24" y="32" font-size="20" font-weight="bold">{title}</text>{body}</g></svg>'
    (DATA / name).write_text(text + '\n')

body=''
for i in range(6):
    for j in range(6):
        color='#2563eb' if j <= i else '#dbe2ec'
        body += f'<rect x="{45+30*j}" y="{65+25*i}" width="26" height="21" rx="3" fill="{color}"/>'
body += '<text x="270" y="95" font-size="16">Blue: visible prefix keys</text><text x="270" y="125" font-size="16">Grey: future positions masked</text><text x="270" y="165" font-size="15">Change a suffix → earlier logits stay fixed.</text><text x="45" y="235" font-size="13">Rows: query positions • Columns: key positions</text>'
svg('causal-mask.svg','Causal attention: test the information boundary',body,260)
body=''
colors=['#ef4444','#22c55e','#3b82f6']
for c,color in enumerate(colors):
    for side in range(2):
        i=c*2+side; x=24+i*114
        body+=f'<rect x="{x}" y="65" width="100" height="100" fill="#111827"/>'
        body+=f'<rect x="{x+side*50}" y="90" width="50" height="50" fill="{color}"/>'
        label=['red','green','blue'][c]+[' left',' right'][side]
        body+=f'<text x="{x}" y="190" font-size="13">{label}</text>'
body+='<text x="24" y="225" font-size="14">Train on five combinations; hold out blue-right. Does alignment compose?</text>'
svg('paired-shapes.svg','Image-text alignment in a controlled world',body)
body=''
for panel,values in enumerate([[3,7,5],[7,3,5]]):
    x=40+panel*350
    body+=f'<text x="{x}" y="66" font-size="15">{["Original: highest B","Counterfactual: highest A"][panel]}</text>'
    body+=f'<path d="M {x} 215 H {x+260}" stroke="#475569"/>'
    for j,v in enumerate(values):
        bx=x+20+j*80
        body+=f'<rect x="{bx}" y="{215-v*18}" width="45" height="{v*18}" fill="{colors[j]}"/><text x="{bx+17}" y="238" font-size="15">{"ABC"[j]}</text>'
svg('chart-intervention.svg','Keep the question fixed; change the visual evidence',body,265)
points=' '.join(f'{30+i*2:.1f},{130-55*math.sin(2*math.pi*4*i/320):.1f}' for i in range(321))
body=f'<line x1="30" y1="130" x2="670" y2="130" stroke="#cbd5e1"/><polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="2"/><rect x="150" y="60" width="160" height="140" fill="#f59e0b" fill-opacity=".12" stroke="#d97706"/><text x="150" y="223" font-size="14">Window → Fourier transform → magnitude → log scale</text>'
svg('audio-window.svg','Illustration: overlapping windows analyze local frequencies',body)
body=''
for k,frame in enumerate([0,4,7,12,20]):
    x=24+k*138
    body+=f'<rect x="{x}" y="70" width="120" height="70" fill="#111827"/><rect x="{x+frame*4}" y="98" width="16" height="16" fill="white"/><text x="{x}" y="163" font-size="14">Frame {frame}</text>'
    if frame==7: body+=f'<rect x="{x}" y="70" width="14" height="14" fill="#ef4444"/>'
body+='<text x="24" y="200" font-size="14">Stride 4 at offset 0 misses the red event at frame 7.</text><text x="24" y="225" font-size="14">Reverse the sequence to test whether an answer depends on order.</text>'
svg('video-sampling.svg','Sampling can preserve motion while missing a brief event',body)
print('Generated five original SVG teaching illustrations.')
