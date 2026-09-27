import os,pickle,json
from pathlib import Path
import numpy as np,pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
ROOT=Path(__file__).resolve().parent
LIVE=Path(os.getenv('LIVE_DATA_FILE',ROOT/'rainfall_tel_hr_uttarakhand_uk_2021_2025.csv'))
FEATURES=['rainfall_1h_mm','rainfall_3h_mm','rainfall_6h_mm','rainfall_12h_mm','rainfall_24h_mm','rainfall_72h_mm']
app=FastAPI(title='FLASHGUARD')
class Req(BaseModel): station:str

def load_live():
 d=pd.read_csv(LIVE); d.columns=[str(c).strip() for c in d.columns]
 tc=next((c for c in d.columns if c.lower() in ['timestamp','datetime','date_time','date','data acquisition time']),None); rc=next((c for c in d.columns if 'rainfall' in c.lower()),None); sc=next((c for c in d.columns if c.lower() in ['station','station name','station_name']),None)
 d['timestamp']=pd.to_datetime(d[tc],dayfirst=True,errors='coerce'); d['rainfall_mm']=pd.to_numeric(d[rc],errors='coerce'); d['station_clean']=d[sc].astype(str).str.strip() d.loc[d.rainfall_mm<0,'rainfall_mm']=np.nan; d['station_clean']=d[sc].astype(str).str.strip(); d=d.dropna(subset=['timestamp']).sort_values(['station_clean','timestamp']).drop_duplicates(['station_clean','timestamp'],keep='last')
 g=d.groupby('station_clean',group_keys=False)
 for h in [1,3,6,12,24,72]: d[f'rainfall_{h}h_mm']=g.rainfall_mm.rolling(h,min_periods=max(1,h//2)).sum().reset_index(level=0,drop=True)
 return d

DATA=load_live()
MODEL=None; METRICS={'status':'not_trained'}
mp=ROOT/'flashguard_model.pkl'; met=ROOT/'model_metrics.json'
if mp.exists():
 with open(mp,'rb') as f: MODEL=pickle.load(f)
if met.exists(): METRICS=json.loads(met.read_text())

def fallback(r):
 s=0
 for col,a,b in [('rainfall_1h_mm',25,50),('rainfall_3h_mm',50,100),('rainfall_24h_mm',100,200)]:
  v=r[col]
  if pd.notna(v): s += 2 if v>=b else 1 if v>=a else 0
 return ('CRITICAL' if s>=5 else 'HIGH' if s>=3 else 'MODERATE' if s>=1 else 'LOW'),s
@app.get('/',response_class=HTMLResponse)
def home(): return '<h1>🌧️ FLASHGUARD</h1><p>Actual-data trained model API</p><p><a href="/api/metrics">Model metrics</a> · <a href="/api/stations">Stations</a></p>'
@app.get('/api/health')
def health(): return {'status':'ok','model_trained':MODEL is not None}
@app.get('/api/metrics')
def metrics(): return METRICS
@app.get('/api/stations')
def stations(): return sorted(DATA.station_clean.unique().tolist())
@app.post('/api/predict')
def predict(req:Req):
 sub=DATA[DATA.station_clean.str.lower()==req.station.strip().lower()]
 if sub.empty:return {'error':'station not found'}
 r=sub.iloc[-1]
 if MODEL is not None:
  x=pd.DataFrame([[r[f] for f in FEATURES]],columns=FEATURES).fillna(0); p=float(MODEL['model'].predict_proba(x)[0,1]); risk='CRITICAL' if p>=.8 else 'HIGH' if p>=.6 else 'MODERATE' if p>=.35 else 'LOW'
  return {'station':r.station_clean,'timestamp':str(r.timestamp),'model':MODEL['model_name'],'flood_probability_percent':round(p*100,2),'risk':risk,'features':{f:float(r[f]) if pd.notna(r[f]) else None for f in FEATURES}}
 risk,score=fallback(r); return {'station':r.station_clean,'timestamp':str(r.timestamp),'model':'fallback','flood_probability_percent':None,'risk':risk,'risk_score':score,'message':'Run train_model.py with the official 1991-2020 rainfall data to enable ML.'}
