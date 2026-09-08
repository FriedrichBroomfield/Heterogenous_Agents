import numpy as np, json
from mundo import Cfg, Pob, un_ciclo, reproducir, medir_verdad


def corre(marcas, ciclos=350, seed=0, gasto=1.5, var_tam=0.5):
    cfg = Cfg(); cfg.marcas=marcas; cfg.gasto_pelea=gasto; cfg.ciclos=ciclos
    rng = np.random.default_rng(seed); pob = Pob(cfg, rng)
    pob.tam = np.clip(1+rng.normal(0,var_tam,pob.n()),0.3,3.0)
    fruta = np.full(cfg.L, cfg.K_fruta)
    reg = dict(escalada=[],peleas=[],n=[],p0=[],w_cre=[],w_tes=[],olvido=[],tam=[],
               err_info=[],err_arb=[],prej_arb=[],real_arb=[],prej_info=[],
               real_info=[],div_nidos=[],conf=[])
    for c in range(ciclos):
        fruta = un_ciclo(pob, fruta, rng, reg)
        reproducir(pob, rng)
        if pob.n() < 10: break
        reg["n"].append(pob.n())
        for r,arr in (("p0",pob.p0),("w_cre",pob.w_cre),("w_tes",pob.w_tes),
                      ("olvido",pob.olvido),("tam",pob.tam)): reg[r].append(float(arr.mean()))
        if marcas:
            V = medir_verdad(pob); cre = pob.a/(pob.a+pob.b); m = np.nanmean(cre,axis=0)
            reg["err_info"].append(float(np.nanmean(np.abs(m[0]-V[0]))))
            reg["err_arb"].append(float(np.nanmean(np.abs(m[1]-V[1]))))
            reg["prej_info"].append(float(np.nanstd(m[0])));  reg["real_info"].append(float(np.nanstd(V[0])))
            reg["prej_arb"].append(float(np.nanstd(m[1])));   reg["real_arb"].append(float(np.nanstd(V[1])))
            nd=np.array([np.nanmean(cre[pob.nido==q,1],axis=0) for q in range(cfg.n_nidos) if (pob.nido==q).sum()>2])
            reg["div_nidos"].append(float(np.nanmean(np.nanstd(nd,axis=0))) if len(nd)>1 else np.nan)
            reg["conf"].append(float((pob.a+pob.b).mean()))
    return cfg,reg,pob

if __name__=="__main__":
    out={}
    for nom,mk in (("sin_marcas",False),("con_marcas",True)):
        R=[]
        for s in range(3):
            cfg,reg,pob = corre(mk,seed=s); R.append(reg)
        out[nom]=R
        u=slice(-60,None)
        print(f"{nom}: escalada={np.mean([np.mean(r['escalada'][u]) for r in R]):.3f}"
              f"  n={np.mean([r['n'][-1] for r in R]):.0f}"
              f"  w_cre={np.mean([np.mean(r['w_cre'][u]) for r in R]):.3f}"
              f"  w_tes={np.mean([np.mean(r['w_tes'][u]) for r in R]):.3f}"
              f"  olvido={np.mean([np.mean(r['olvido'][u]) for r in R]):.3f}")
        if mk:
            for k in ("err_info","err_arb","prej_info","real_info","prej_arb","real_arb","div_nidos"):
                print(f"   {k:11s} = {np.mean([np.mean(r[k][u]) for r in R]):.4f}")
    json.dump(out, open("prejuicio.json","w"))
