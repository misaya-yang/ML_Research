"""Minimal independent replay of Pro's cold-IMH examples. Python 3 stdlib only.

Run: python3 experiments/finite_state/pro_cold_core.py
Prints CSV; compare to results/pro/pro_cold_core.csv. Risk and classification
use exact rational arithmetic; decimals (including the square-root upper
bound) are display only. Original model and bounds: Pro, 2026-11-22.
"""
from fractions import Fraction as F
from math import sqrt
import csv
import sys


def moments(p, q, h, t, b):
    w = [a/z for a, z in zip(p, q)]
    f = [z-sum(a*y for a, y in zip(p, h)) for z in h]
    v = sum(a*z*z for a, z in zip(p, f))
    k = []
    for i, x in enumerate(w):
        row = [z*min(1, y/x) for y, z in zip(w, q)]
        row[i] += 1-sum(row)
        k.append(row)

    def step(x):
        return [sum(a*z for a, z in zip(row, x)) for row in k]

    u = f[:]
    s = [x*x for x in f]
    for _ in range(t-1):
        ku, ks = step(u), step(s)
        s = [x*x+2*x*y+z for x, y, z in zip(f, ku, ks)]
        u = [x+y for x, y in zip(f, ku)]
    for _ in range(b):
        s = step(s)
    return sum(a*z for a, z in zip(q, s))/t**2/v, w, f, v


def paired(t, p0, h):
    p = (p0, p0, 2*p0, 1-4*p0)
    q = (p0/t, p0, p0+p0/t, 1-2*p0-2*p0/t)
    risk, w, f, v = moments(p, q, h, t, 0)
    a = sum(y*min(1, x) for x, y in zip(w, q))
    a2 = sum(y*min(1, x/2) for x, y in zip(w, q))
    tail = p0*f[0]**2/(1-p0)
    s = tail + (v-tail)*min(1, (2/a2-1)/t)
    u = sum(y*max(0, 1-x)*z*z for x, y, z in zip(w, q, f))
    hbound = u*(3-2*a)/(a**3*t**2)
    chi = sum(y*(min(1, 1/x)/a-1)**2 for x, y in zip(w, p))
    base, rad = (s/a+hbound)/v, 4*chi*s*hbound/v**2
    assert risk <= base or (risk-base)**2 <= rad
    gate = F(1, 12)
    return risk, float(base)+sqrt(float(rad)), base < gate and rad < (gate-base)**2


def main():
    rows=[]
    for k in (8, 16):
        for name, h in [('high-and-unit',(1,1,0,0)),('medium',(0,0,1,0))]:
            t = k*k
            r, u, cert = paired(t, F(1,k), h)
            rows.append([name,t,0,f'{float(r):.12f}',f'{u:.12f}',str(cert).lower(),str(r<F(1,12)).lower()])
    t=64
    p0=F(1,t**3)
    r,*_ = moments((p0,1-p0),(F(1,2),F(1,2)),(1,0),t,0)
    assert r>F(9*t,32)
    rows.append(['low-initial',t,0,f'{float(r):.12f}','','',''])
    p0=F(1,8); q0=p0/t
    for b in (0, 64):
        r,*_ = moments((p0,1-p0),(q0,1-q0),(1,0),t,b)
        assert r>F(1,64)*F(1,3)
        rows.append(['rare-high',t,b,f'{float(r):.12f}','','',''])
    wr=csv.writer(sys.stdout,lineterminator='\n')
    wr.writerow(['model','T','burn','risk_over_variance','pro_upper_over_variance','certified_at_1_over_12','actually_safe_at_1_over_12'])
    wr.writerows(rows)


if __name__=='__main__':
    main()
