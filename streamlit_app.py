# Kalshi Temperature Model - Stable TX Hotfix
import math,re,requests,streamlit as st,pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
st.set_page_config(page_title='Kalshi Temperature Model - Stable TX Hotfix',layout='wide')
st.title('Kalshi Temperature Model - Stable TX Hotfix')
CITIES={'Phoenix':{'lat':33.4342,'lon':-112.0116,'tz':'America/Phoenix','bias':0.5,'afternoon_bump':0.3},'Las Vegas':{'lat':36.0840,'lon':-115.1537,'tz':'America/Los_Angeles','bias':0.4,'afternoon_bump':0.3},'Los Angeles':{'lat':33.9416,'lon':-118.4085,'tz':'America/Los_Angeles','bias':-0.6,'afternoon_bump':0.0},'Dallas':{'lat':32.8998,'lon':-97.0403,'tz':'America/Chicago','bias':0.4,'afternoon_bump':0.3},'Austin':{'lat':30.1945,'lon':-97.6699,'tz':'America/Chicago','bias':0.3,'afternoon_bump':0.2},'Houston':{'lat':29.9902,'lon':-95.3368,'tz':'America/Chicago','bias':0.3,'afternoon_bump':0.2},'Atlanta':{'lat':33.6407,'lon':-84.4277,'tz':'America/New_York','bias':0.2,'afternoon_bump':0.2},'NYC':{'lat':40.7829,'lon':-73.9654,'tz':'America/New_York','bias':0.6,'afternoon_bump':0.1},'Miami':{'lat':25.7959,'lon':-80.2870,'tz':'America/New_York','bias':0.2,'afternoon_bump':0.1}}
LADDERS={'Phoenix':'78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above','Las Vegas':'73 or below | 74-75 | 76-77 | 78-79 | 80-81 | 82 or above','Los Angeles':'65 or below | 66-67 | 68-69 | 70-71 | 72-73 | 74 or above','Dallas':'78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above','Austin':'78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above','Houston':'79 or below | 80-81 | 82-83 | 84-85 | 86-87 | 88 or above','Atlanta':'74 or below | 75-76 | 77-78 | 79-80 | 81-82 | 83 or above','NYC':'62 or below | 63-64 | 65-66 | 67-68 | 69-70 | 71 or above','Miami':'79 or below | 80-81 | 82-83 | 84-85 | 86-87 | 88 or above'}
W={'ICON':0.35,'OpenMeteo':0.30,'GFS':0.20,'NWS':0.15}
OUTLIER_HALF=3.0; OUTLIER_REMOVE=4.5; SIGMA_MIN=1.25; SIGMA_MAX=2.0; SPREAD_SAFETY_THRESHOLD=5.0; SPREAD_SAFETY_MULTIPLIER=0.4
def safe_get(url,params=None):
    try:
        r=requests.get(url,params=params,timeout=10); r.raise_for_status(); return r.json()
    except: return None
def median(vals):
    s=sorted(vals); n=len(s)
    if n==0:return None
    return s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2
def compute_weights(f):
    vals=[v for v in f.values() if v is not None]; med=median(vals)
    if med is None:return {k:0 for k in f}
    out={}
    for k,v in f.items():
        if v is None: out[k]=0; continue
        d=abs(v-med); w=W.get(k,0)
        if d>OUTLIER_REMOVE:w=0
        elif d>OUTLIER_HALF:w*=0.5
        out[k]=w
    return out
def consensus(f,w):
    num=den=0.0
    for k,v in f.items():
        if v is None: continue
        num+=v*w.get(k,0); den+=w.get(k,0)
    return None if den==0 else num/den
def solar_adjust(cloud,hour,city):
    if hour<9 or hour>17 or cloud is None:return 0.0
    adj=1.0 if cloud<10 else 0.6 if cloud<30 else 0.2 if cloud<50 else -0.5
    if city in {'Phoenix','Las Vegas'}:
        if cloud<20: adj+=0.4
        elif cloud<40: adj+=0.2
    return adj
def expected_curve(hour):
    curve={6:0.55,7:0.60,8:0.65,9:0.70,10:0.75,11:0.80,12:0.85,13:0.90,14:0.95,15:0.98,16:1.00,17:1.00}
    return curve.get(hour,1.0 if hour>17 else 0.50)
def raw_traj(current,high,hour):
    if current is None or high is None or hour<8 or hour>16:return 0.0
    diff=current-high*expected_curve(hour); adj=diff*0.35
    return max(min(adj,2.0),-2.0)
def valve(traj,spread): return traj*SPREAD_SAFETY_MULTIPLIER if spread>SPREAD_SAFETY_THRESHOLD else traj
def texas_cloud_cap(city,cloud,hour,current,proj):
    if city not in {'Dallas','Austin','Houston'} or current is None or proj is None or cloud is None: return proj,0.0
    reduction=0.0
    if hour>=13 and cloud>=70:
        cap=min(proj,current+4.0); reduction=proj-cap; proj=cap
    elif hour>=14 and cloud>=50:
        cap=min(proj,current+5.0); reduction=proj-cap; proj=cap
    return proj,reduction
def cdf(x,mu,s): return 0.5*(1+math.erf((x-mu)/(s*math.sqrt(2))))
def parse_ladder(text):
    out=[]
    for p in [q.strip() for q in text.split('|') if q.strip()]:
        nums=[int(x) for x in re.findall(r'\d+',p)]; lower=p.lower()
        if 'below' in lower and nums: out.append((p,None,nums[0]))
        elif 'above' in lower and nums: out.append((p,nums[0],None))
        elif len(nums)>=2: out.append((p,nums[0],nums[1]))
    return out
def probs(mu,sigma,brackets):
    rows=[]
    for lab,lo,hi in brackets:
        p=cdf(hi+0.5,mu,sigma) if lo is None else 1-cdf(lo-0.5,mu,sigma) if hi is None else cdf(hi+0.5,mu,sigma)-cdf(lo-0.5,mu,sigma)
        rows.append((lab,p))
    rows.sort(key=lambda x:x[1],reverse=True); return rows
city=st.selectbox('City',list(CITIES.keys())); p=CITIES[city]; hour=datetime.now(ZoneInfo(p['tz'])).hour
ladder_text=st.text_input('Kalshi ladder',LADDERS[city])
om=safe_get('https://api.open-meteo.com/v1/forecast',{'latitude':p['lat'],'longitude':p['lon'],'daily':'temperature_2m_max','current':'temperature_2m,cloud_cover','temperature_unit':'fahrenheit','timezone':'auto'})
gfs=safe_get('https://api.open-meteo.com/v1/forecast',{'latitude':p['lat'],'longitude':p['lon'],'daily':'temperature_2m_max','models':'gfs_seamless','temperature_unit':'fahrenheit','timezone':'auto'})
icon=safe_get('https://api.open-meteo.com/v1/forecast',{'latitude':p['lat'],'longitude':p['lon'],'daily':'temperature_2m_max','models':'icon_seamless','temperature_unit':'fahrenheit','timezone':'auto'})
nws=safe_get(f"https://api.weather.gov/points/{p['lat']},{p['lon']}")
nws_high=None
if nws and 'properties' in nws and nws['properties'].get('forecast'):
    fc=safe_get(nws['properties']['forecast'])
    if fc and 'properties' in fc and 'periods' in fc['properties']:
        for per in fc['properties']['periods']:
            if per.get('isDaytime'): nws_high=per.get('temperature'); break
f={'ICON':icon['daily']['temperature_2m_max'][0] if icon and 'daily' in icon else None,'OpenMeteo':om['daily']['temperature_2m_max'][0] if om and 'daily' in om else None,'GFS':gfs['daily']['temperature_2m_max'][0] if gfs and 'daily' in gfs else None,'NWS':nws_high}
weights=compute_weights(f); cons=consensus(f,weights)
cloud=current=None
if om and 'current' in om:
    cloud=om['current'].get('cloud_cover'); current=om['current'].get('temperature_2m')
vals=[v for v in f.values() if v is not None]; spread=(max(vals)-min(vals)) if len(vals)>=2 else 0.0; sigma=min(max(1.3+spread*0.25,SIGMA_MIN),SIGMA_MAX)
rtraj=traj=cloud_cap=0.0
if cons is not None:
    cons+=p['bias']; cons+=solar_adjust(cloud,hour,city); rtraj=raw_traj(current,cons,hour); traj=valve(rtraj,spread); cons+=traj
    if hour>=13: cons+=p.get('afternoon_bump',0.0)
    cons,cloud_cap=texas_cloud_cap(city,cloud,hour,current,cons)
st.subheader('Forecast Sources'); st.write(f)
st.subheader('Weights'); st.write(weights)
st.subheader('Consensus High'); st.write(round(cons,2) if cons is not None else 'N/A')
st.subheader('Current Station Temp'); st.write(current)
st.subheader('Cloud Cover'); st.write(cloud)
st.subheader('Raw Trajectory Adjustment'); st.write(round(rtraj,2) if cons is not None else 'N/A')
st.subheader('Trajectory Adjustment'); st.write(round(traj,2) if cons is not None else 'N/A')
st.subheader('Texas Cloud Cap Reduction'); st.write(round(cloud_cap,2) if cons is not None else 'N/A')
st.subheader('Forecast Spread'); st.write(round(spread,2))
st.subheader('Sigma'); st.write(round(sigma,2))
st.subheader('Spread Safety Valve'); st.write('ON' if spread>SPREAD_SAFETY_THRESHOLD else 'OFF')
if cons is not None:
    df=pd.DataFrame(probs(cons,sigma,parse_ladder(ladder_text)),columns=['Bracket','Model Probability'])
    df['Model Probability']=df['Model Probability'].apply(lambda x:f'{x*100:.1f}%')
    st.subheader('Kalshi Bracket Probabilities'); st.dataframe(df,use_container_width=True)
st.caption('Stable TX Hotfix â stable baseline + Texas cloud suppression')
