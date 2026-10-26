"""Live dashboard: stdlib HTTP server + one HTML page with inline JS and canvas.

Serves http://127.0.0.1:8765/ and a JSON endpoint /api/state that the page
polls once a second. No external scripts or fonts are loaded.
"""

from __future__ import annotations

import http.server
import threading
from typing import Optional

import numpy as np

from .state import RunState, json_text
from .telemetry import Sampler, bin_series

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Heatwave laptop</title>
<style>
:root{--bg:#10151c;--panel:#18202b;--ink:#e6edf5;--mute:#8fa1b5;--line:#2a3646;
--pow:#4cc3ff;--tmp:#ff9f43;--band1:rgba(120,170,255,.16);--band2:rgba(120,170,255,.30);
--naive:#c0c8d2;--ckpt:#9b8cff;--lim:#ff5d73;--ok:#4cd08a}
@media (prefers-color-scheme: light){:root{--bg:#f3f5f8;--panel:#fff;--ink:#16202c;--mute:#5a6b7e;--line:#d5dce5;
--pow:#0b78c4;--tmp:#d96a00;--band1:rgba(30,90,200,.13);--band2:rgba(30,90,200,.25);--naive:#6b7683;--ckpt:#6a58d6;--lim:#d1243f;--ok:#14894f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px system-ui,Segoe UI,sans-serif}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;padding:12px 16px}
h1{font-size:18px;margin:0 12px 0 0}
.badge{font-weight:800;letter-spacing:.08em;border-radius:8px;padding:6px 14px}
#mode{font-size:34px;padding:8px 20px}
.dry{background:#ffcc33;color:#201800}.live{background:#ff5d73;color:#fff}
#sim{background:#7a5cff;color:#fff;font-size:18px}
.stats{display:flex;flex-wrap:wrap;gap:10px;padding:0 16px 8px}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px 14px;min-width:120px}
.stat b{display:block;font-size:22px}.stat span{color:var(--mute);font-size:12px}
main{padding:0 16px 16px;display:grid;gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px}
canvas{width:100%;height:260px;display:block}
.legend{color:var(--mute);font-size:12px;padding:4px 8px}
#note{padding:0 16px 16px;color:var(--mute);font-size:13px;white-space:pre-wrap}
</style></head><body>
<header><h1>Heatwave laptop GPU</h1><div id="mode" class="badge dry">...</div><div id="sim" class="badge" style="display:none">SIMULATED DATA</div></header>
<div class="stats" id="stats"></div>
<main>
<div class="card"><canvas id="cp"></canvas><div class="legend">Power (W). Shaded: forecast 1-99% and 10-90%. Solid: Timestral median. Dashed grey: seasonal-naive median. Vertical lines: checkpoints (solid = logged, dotted = scheduled). Red line: power budget.</div></div>
<div class="card"><canvas id="ct"></canvas><div class="legend">Temperature (C). Same encoding. Red line: temperature setpoint.</div></div>
</main>
<div id="note"></div>
<script>
const $=id=>document.getElementById(id);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let S=null;
function fmt(v,d=1){return (v===null||v===undefined||isNaN(v))?'n/a':Number(v).toFixed(d)}
function stat(label,val,unit){return `<div class="stat"><b>${val}</b><span>${label}${unit?' ('+unit+')':''}</span></div>`}
function draw(cv,key,color,setpoint){
  const dpr=window.devicePixelRatio||1,w=cv.clientWidth,h=cv.clientHeight;
  cv.width=w*dpr;cv.height=h*dpr;const g=cv.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);
  g.clearRect(0,0,w,h);
  const L=46,R=10,T=8,B=22,pw=w-L-R,ph=h-T-B;
  const hist=S.history, f=S.forecast;
  const win=S.window_s, H=f?f.horizon:0;
  const x=t=>L+((t+win)/(win+H))*pw;  // t = seconds relative to now (<=0 history)
  let ys=[];hist[key].forEach(v=>{if(v!==null)ys.push(v)});
  if(f){for(const m of Object.values(f.methods)){const q=m[key];if(q){q.q01.forEach(v=>ys.push(v));q.q99.forEach(v=>ys.push(v))}}}
  if(setpoint!==null&&setpoint!==undefined)ys.push(setpoint);
  if(!ys.length){g.fillStyle=css('--mute');g.fillText('waiting for data...',L+10,T+20);return}
  let lo=Math.min(...ys),hi=Math.max(...ys);const pad=(hi-lo||1)*.1;lo-=pad;hi+=pad;
  const y=v=>T+ph-((v-lo)/(hi-lo))*ph;
  g.strokeStyle=css('--line');g.fillStyle=css('--mute');g.lineWidth=1;g.font='11px system-ui';
  for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4,yy=y(v);g.beginPath();g.moveTo(L,yy);g.lineTo(L+pw,yy);g.stroke();g.fillText(v.toFixed(0),4,yy+4)}
  for(let t=-win;t<=H;t+=60){const xx=x(t);g.fillText((t>=0?'+':'')+t+'s',xx-10,h-6)}
  // now line
  g.strokeStyle=css('--mute');g.setLineDash([2,3]);g.beginPath();g.moveTo(x(0),T);g.lineTo(x(0),T+ph);g.stroke();g.setLineDash([]);
  // checkpoints
  g.strokeStyle=css('--ckpt');
  S.ckpt_logged.forEach(t=>{const r=t-S.now_s;if(r>=-win){g.beginPath();g.moveTo(x(r),T);g.lineTo(x(r),T+ph);g.stroke()}});
  g.setLineDash([2,3]);
  S.ckpt_scheduled.forEach(t=>{const r=t-S.now_s;if(r<=H&&r>=0){g.beginPath();g.moveTo(x(r),T);g.lineTo(x(r),T+ph);g.stroke()}});g.setLineDash([]);
  // forecast bands + medians (Timestral first)
  if(f){
    const off=f.origin_s-S.now_s;
    for(const [name,m] of Object.entries(f.methods)){
      const q=m[key];if(!q)continue;const isT=name.startsWith('timestral');
      if(isT){
        for(const [a,b,c] of [['q01','q99','--band1'],['q10','q90','--band2']]){
          g.fillStyle=css(c);g.beginPath();
          q[a].forEach((v,i)=>{const xx=x(off+i+.5);i?g.lineTo(xx,y(v)):g.moveTo(xx,y(v))});
          for(let i=q[b].length-1;i>=0;i--)g.lineTo(x(off+i+.5),y(q[b][i]));
          g.closePath();g.fill();
        }
      }
      g.strokeStyle=isT?color:css('--naive');g.lineWidth=isT?2:1.5;g.setLineDash(isT?[]:[6,4]);g.beginPath();
      q.q50.forEach((v,i)=>{const xx=x(off+i+.5);i?g.lineTo(xx,y(v)):g.moveTo(xx,y(v))});g.stroke();g.setLineDash([]);g.lineWidth=1;
    }
  }
  // history
  g.strokeStyle=color;g.lineWidth=2;g.beginPath();let first=true;
  hist[key].forEach((v,i)=>{if(v===null){first=true;return}const t=hist.t[i]-S.now_s,xx=x(t);first?g.moveTo(xx,y(v)):g.lineTo(xx,y(v));first=false});
  g.stroke();g.lineWidth=1;
  if(setpoint!==null&&setpoint!==undefined){g.strokeStyle=css('--lim');g.setLineDash([8,4]);g.beginPath();g.moveTo(L,y(setpoint));g.lineTo(L+pw,y(setpoint));g.stroke();g.setLineDash([])}
}
function render(){
  if(!S)return;
  const c=S.control||{},m=S.meta;
  const dry=c.dry_run!==false;
  const mode=$('mode');mode.textContent=dry?'DRY RUN':'LIVE CONTROL';mode.className='badge '+(dry?'dry':'live');
  $('sim').style.display=m.data_kind==='simulated'?'block':'none';
  const n=S.latest||{};
  $('stats').innerHTML=stat('GPU power',fmt(n.power_w),'W')+stat('GPU temp',fmt(n.temp_c),'C')+stat('SM clock',fmt(n.sm_clock_mhz,0),'MHz')
   +stat('Controller limit',c.current===undefined?'n/a':fmt(c.current,0)+' '+(c.unit||''),c.mode?c.mode+' mode':'')
   +stat('Enforced power limit',fmt(n.enforced_limit_w),'W')
   +stat('Forecast quantile',c.quantile===undefined?'n/a':c.quantile,'')
   +stat('Elapsed',fmt(S.now_s,0),'s');
  const last=c.last||{};
  $('note').textContent=(c.mode_reason?('Mode: '+c.mode_reason+'\n'):'')+(last.reason?('Last decision: '+(last.status||'')+' - '+last.reason+(last.command?(' ['+last.command+']'):'')+'\n'):'')
   +(c.last_command?('Last command: '+c.last_command+'\n'):'')+(m.data_kind==='simulated'?'All values on this page are SIMULATED (fake backend).\n':'');
  draw($('cp'),'power_w',css('--pow'),c.power_budget_w===undefined?null:c.power_budget_w);
  draw($('ct'),'temp_c',css('--tmp'),c.temp_limit_c===undefined?null:c.temp_limit_c);
}
async function poll(){
  try{const r=await fetch('/api/state');S=await r.json();render()}catch(e){$('mode').textContent='DISCONNECTED';$('mode').className='badge dry'}
}
window.addEventListener('resize',render);poll();setInterval(poll,1000);
</script></body></html>
"""

_QUANTS = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}


def build_api_state(state: RunState, sampler: Sampler, window_s: int = 300) -> dict:
    """The JSON the page polls. ``now_s`` and every time are sampler seconds."""
    snap = sampler.window(window_s)
    now_s = snap[-1].t_mono_s if snap else 0.0
    t = np.array([s.t_mono_s for s in snap])
    hist = {"t": [], "power_w": [], "temp_c": []}
    if len(snap):
        t0 = int(np.floor(t[0]))
        n_bins = int(np.floor(now_s)) - t0 + 1
        for key in ("power_w", "temp_c"):
            hist[key] = bin_series(t - t0, np.array([s.values[key] for s in snap]), n_bins).tolist()
        hist["t"] = [t0 + i + 0.5 for i in range(n_bins)]
    latest = snap[-1].values if snap else {}
    forecast = None
    rec = state.get_forecast()
    if rec is not None:
        methods = {}
        for name, fc in rec.forecasts.items():
            methods[name] = {
                key: {qn: fc.quantile(q)[0, v].tolist() for qn, q in _QUANTS.items()}
                for v, key in enumerate(("power_w", "temp_c"))
            }
        forecast = {"origin_s": rec.origin_s, "horizon": rec.horizon, "methods": methods}
    logged, scheduled = [], []
    if state.tracker is not None:
        state.tracker.refresh()
        logged = list(state.tracker.starts)
        known = set(round(s, 3) for s in logged)
        horizon = rec.horizon if rec is not None else 60
        scheduled = [s for s, _ in state.tracker.intervals(now_s + horizon) if round(s, 3) not in known]
    return {
        "meta": state.meta, "now_s": now_s, "window_s": window_s, "history": hist,
        "latest": latest, "forecast": forecast, "control": state.get_control(),
        "ckpt_logged": logged, "ckpt_scheduled": scheduled,
    }


class DashboardServer:
    def __init__(self, state: RunState, sampler: Sampler, port: int = 8765,
                 host: str = "127.0.0.1") -> None:
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # keep the console quiet
                pass

            def _send(self, code: int, ctype: str, body: bytes) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                path = self.path.split("?")[0]
                if path in ("/", "/index.html"):
                    self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
                elif path == "/api/state":
                    body = json_text(build_api_state(outer.state, outer.sampler)).encode("utf-8")
                    self._send(200, "application/json", body)
                else:
                    self._send(404, "text/plain", b"not found")

        self.state = state
        self.sampler = sampler
        self.httpd = http.server.ThreadingHTTPServer((host, port), Handler)
        self.port = self.httpd.server_address[1]
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = _Serve(self.httpd)
        self._thread.start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


class _Serve(threading.Thread):
    def __init__(self, httpd) -> None:
        super().__init__(name="dashboard", daemon=True)
        self.httpd = httpd

    def run(self) -> None:
        self.httpd.serve_forever(poll_interval=0.2)
