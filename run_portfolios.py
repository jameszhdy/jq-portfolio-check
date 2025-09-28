# run_portfolios.py
# 说明：从环境变量读取 JQDATA_USER, JQDATA_PASS, SERVERCHAN_KEY
# 运行：python run_portfolios.py

import os
import sys
from datetime import datetime, date
import pandas as pd
import numpy as np
import requests
import yaml
# 聚宽 jqdatasdk
from jqdatasdk import auth, finance, query

ROLL_WINDOW = 180
DAYS_LOOKBACK = max(ROLL_WINDOW * 3, 560)

def load_config(path='portfolios.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg.get('portfolios', [])

def jq_auth():
    user = os.environ.get('JQDATA_USER')
    pwd = os.environ.get('JQDATA_PASS')
    if not user or not pwd:
        raise RuntimeError("未找到聚宽账号，请设置 JQDATA_USER / JQDATA_PASS 环境变量或 Secrets")
    auth(user, pwd)
    print("[INFO] 已登录 jqdatasdk")

def daterange_year_chunks(start_date, end_date):
    sd = pd.Timestamp(start_date).date()
    ed = pd.Timestamp(end_date).date()
    year = sd.year
    while year <= ed.year:
        cs = date(year, 1, 1) if year > sd.year else sd
        ce = date(year, 12, 31)
        if ce > ed:
            ce = ed
        yield cs, ce
        year += 1

def fetch_fund_nav(codes, start_date=None, end_date=None, prefer_acc_nav=True):
    if start_date is None:
        start_date = (pd.Timestamp(datetime.today()) - pd.Timedelta(days=DAYS_LOOKBACK)).date()
    if end_date is None:
        end_date = pd.Timestamp(datetime.today()).date()
    frames = []
    for cs, ce in daterange_year_chunks(start_date, end_date):
        q = query(finance.FUND_NET_VALUE).filter(
            finance.FUND_NET_VALUE.code.in_(codes),
            finance.FUND_NET_VALUE.day >= cs,
            finance.FUND_NET_VALUE.day <= ce
        )
        df_part = finance.run_query(q)
        if df_part is not None and len(df_part) > 0:
            frames.append(df_part)
    if not frames:
        raise RuntimeError(f"未取到基金净值数据: {codes}")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=['code', 'day']).sort_values(['day', 'code'])
    cols = set(df.columns)
    acc_candidates = ['sum_value', 'acc_net_value', 'accumulative_net_value']
    net_col = None
    for c in acc_candidates:
        if c in cols and prefer_acc_nav:
            net_col = c
            break
    if net_col is None:
        if 'net_value' in cols:
            net_col = 'net_value'
        else:
            raise RuntimeError("未找到净值列")
    nav_wide = df.pivot_table(index='day', columns='code', values=net_col, aggfunc='last')
    nav_wide = nav_wide.sort_index().dropna(how='all')
    nav_wide = nav_wide.reindex(columns=codes)
    return nav_wide

def check_portfolio(port_cfg):
    name = port_cfg.get('name', 'noname')
    codes = port_cfg.get('codes', [])
    weights = port_cfg.get('weights')
    if not codes:
        return {'name': name, 'status': 'no_codes'}
    nav = fetch_fund_nav(codes)
    # 尝试只使用所有代码同时有值的行（交集）
    nav_aligned = nav.dropna(how='any')
    if nav_aligned.empty:
        return {'name': name, 'status': 'no_data'}
    nav_norm = nav_aligned / nav_aligned.iloc[0]
    if weights:
        w = pd.Series(weights, dtype=float)
        if abs(w.sum() - 1.0) > 1e-8:
            w = w / w.sum()
    else:
        w = pd.Series({c: 1.0/len(codes) for c in codes})
    portfolio_nv = (nav_norm * w).sum(axis=1)
    portfolio_nv = portfolio_nv / portfolio_nv.iloc[0]
    ma = portfolio_nv.rolling(window=ROLL_WINDOW, min_periods=ROLL_WINDOW).mean()
    mask_valid = ma.notna() & portfolio_nv.notna()
    if not mask_valid.any():
        return {'name': name, 'status': 'insufficient_data'}
    last_date = mask_valid[mask_valid].index[-1]
    latest_nv = float(portfolio_nv.loc[last_date])
    latest_ma = float(ma.loc[last_date])
    status = 'below' if latest_nv < latest_ma else 'above'
    return {
        'name': name,
        'status': status,
        'date': str(pd.Timestamp(last_date).date()),
        'latest_nv': latest_nv,
        'latest_ma': latest_ma
    }

def push_message_serverchan(sendkey, title, content):
    if not sendkey:
        raise RuntimeError("未提供 SERVERCHAN_KEY（请设置环境变量）")
    url = f"https://sctapi.ftqq.com/{sendkey}.send"
    payload = {
        "title": title,
        "desp": content
    }
    try:
        r = requests.post(url, data=payload, timeout=10)
        print("[PUSH] 返回:", r.text)
    except Exception as e:
        print("[PUSH] 错误:", e)

def main():
    try:
        jq_auth()
    except Exception as e:
        print("[ERROR] 聚宽登录失败：", e)
        sys.exit(1)
    portfolios = load_config('portfolios.yaml')
    results = []
    for p in portfolios:
        try:
            res = check_portfolio(p)
            results.append(res)
            print("[INFO]", res)
        except Exception as e:
            print("[WARN] 处理组合失败", p.get('name'), e)
    below = [r for r in results if r.get('status') == 'below']
    sendkey = os.environ.get('SERVERCHAN_KEY')
    if below:
        lines = []
        for r in below:
            lines.append(f"组合: {r['name']}")
            lines.append(f"日期: {r['date']}")
            lines.append(f"最新净值: {r['latest_nv']:.6f}，MA{ROLL_WINDOW}: {r['latest_ma']:.6f}")
            lines.append("---")
        content = "\n".join(lines)
        push_message_serverchan(sendkey, f"[告警] {len(below)}个组合低于MA{ROLL_WINDOW}", content)
    else:
        print("[INFO] 没有组合低于 MA")

if __name__ == '__main__':
    main()
