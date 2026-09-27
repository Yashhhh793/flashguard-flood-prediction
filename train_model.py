import re, json, pickle, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

ROOT=Path(__file__).resolve().parent
RAIN_URL='https://nwdp.nwic.gov.in/dataset/00f9007e-9b5a-4ce0-87e0-cee4ccd1a8e5/resource/89c87736-d23c-45e7-b4a6-a9ab8c0c4ac8/download/rainfall_tel_hr_uttarakhand_uk_1991_2020.csv'
RAIN_FILE=ROOT/'rainfall_tel_hr_uttarakhand_uk_1991_2020.csv'
EVENT_FILE=ROOT/'historical_flash_flood_events_uttarakhand.csv'
FEATURES=['rainfall_1h_mm','rainfall_3h_mm','rainfall_6h_mm','rainfall_12h_mm','rainfall_24h_mm','rainfall_72h_mm']


def download_rainfall():
    if RAIN_FILE.exists() and RAIN_FILE.stat().st_size>1000:
        return
    print('Downloading official NWIC Uttarakhand 1991-2020 telemetry rainfall...')
    urllib.request.urlretrieve(RAIN_URL,RAIN_FILE)


def prepare_rainfall():
    df=pd.read_csv(RAIN_FILE)
    df.columns=[str(c).strip() for c in df.columns]
    time_col=next(c for c in df.columns if c.lower() in ['timestamp','datetime','date_time','date'])
    rain_col=next(c for c in df.columns if 'rainfall' in c.lower())
    station_col=next(c for c in df.columns if c.lower() in ['station','station name','station_name'])
    district_col=next((c for c in df.columns if 'district' in c.lower()),None)
    df['timestamp']=pd.to_datetime(df[time_col],dayfirst=True,errors='coerce')
    df['rainfall_mm']=pd.to_numeric(df[rain_col],errors='coerce')
    df.loc[df.rainfall_mm<0,'rainfall_mm']=np.nan
    df['station_clean']=df[station_col].astype(str).str.strip()
    if district_col: df['district']=df[district_col].astype(str).str.strip()
    else: df['district']=''
    df=df.dropna(subset=['timestamp']).sort_values(['station_clean','timestamp'])
    df=df.drop_duplicates(['station_clean','timestamp'],keep='last')
    g=df.groupby('station_clean',group_keys=False)
    for h in [1,3,6,12,24,72]:
        df[f'rainfall_{h}h_mm']=g.rainfall_mm.rolling(h,min_periods=max(1,h//2)).sum().reset_index(level=0,drop=True)
    return df


def make_labels(df):
    events=pd.read_csv(EVENT_FILE,parse_dates=['date'])
    events=events.dropna(subset=['date'])
    events=events[events['district'].astype(str).str.strip()!=''].copy()
    events['district_norm']=events.district.str.lower().str.replace(r'[^a-z ]','',regex=True).str.replace(r'\s+',' ',regex=True).str.strip()
    df['district_norm']=df.district.astype(str).str.lower().str.replace(r'[^a-z ]','',regex=True).str.replace(r'\s+',' ',regex=True).str.strip()

    # Leakage-safe target: at time T, label 1 if a documented flash-flood event
    # occurs in the same district during (T, T+24h].
    event_dates={}
    for _,e in events.iterrows():
        event_dates.setdefault(e.district_norm,set()).add(e.date.normalize())

    def label_row(ts,dist):
        if not dist or dist not in event_dates: return 0
        d=ts.normalize()
        return int(any(d < ed <= d+pd.Timedelta(hours=24) for ed in event_dates[dist]))
    df['flood_label']=[label_row(ts,dist) for ts,dist in zip(df.timestamp,df.district_norm)]
    return df


def train(df):
    w=df.dropna(subset=FEATURES).sort_values('timestamp').copy()
    # Remove rows after the final event source date only if desired; keep the full 1991-2020 record.
    # Chronological holdout.
    cut=int(len(w)*0.80)
    tr=w.iloc[:cut]; te=w.iloc[cut:]
    Xtr,ytr=tr[FEATURES],tr.flood_label.astype(int)
    Xte,yte=te[FEATURES],te.flood_label.astype(int)
    if ytr.nunique()<2 or yte.nunique()<2:
        raise RuntimeError(f'Chronological split lacks both classes: train={ytr.value_counts().to_dict()}, test={yte.value_counts().to_dict()}')
    models={
      'random_forest':RandomForestClassifier(n_estimators=400,max_depth=10,min_samples_leaf=2,class_weight='balanced',random_state=42,n_jobs=-1),
      'gradient_boosting':GradientBoostingClassifier(n_estimators=250,learning_rate=.05,max_depth=3,random_state=42)
    }
    results={}; best=None; best_auc=-1
    for name,m in models.items():
        m.fit(Xtr,ytr); p=m.predict(Xte); prob=m.predict_proba(Xte)[:,1]
        auc=roc_auc_score(yte,prob)
        results[name]={'accuracy':float(accuracy_score(yte,p)),'roc_auc':float(auc),'confusion_matrix':confusion_matrix(yte,p).tolist(),'test_positive':int(yte.sum())}
        if auc>best_auc: best_auc=auc; best=(name,m)
    name,model=best
    bundle={'model':model,'features':FEATURES,'model_name':name,'label_definition':'1 = documented flash-flood event in same district within next 24 hours; district-matched from historical inventory'}
    with open(ROOT/'flashguard_model.pkl','wb') as f: pickle.dump(bundle,f)
    metrics={'status':'trained','label_method':'district_date_forward_24h','events_used':int(len(pd.read_csv(EVENT_FILE).query("district != ''"))), 'rows':int(len(w)),'train_samples':int(len(tr)),'test_samples':int(len(te)),'positive_train':int(ytr.sum()),'positive_test':int(yte.sum()),'models':results,'selected_model':name,'warning':'Labels are district/date matched because the historical inventory does not provide station coordinates/timestamps.'}
    (ROOT/'model_metrics.json').write_text(json.dumps(metrics,indent=2))
    # Save training table for inspection.
    w.to_csv(ROOT/'flashguard_training_dataset.csv',index=False)
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':
    download_rainfall(); df=prepare_rainfall(); df=make_labels(df); train(df)
