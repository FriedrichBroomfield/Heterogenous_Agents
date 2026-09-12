import numpy as np, json
from mundo import Cfg, Pob, un_ciclo, reproducir, medir_verdad


def corre(marcas, ciclos=350, seed=0, gasto=1.5, var_tam=0.5, disp_nido=0.0):
    cfg = Cfg(); cfg.marcas=marcas; cfg.gasto_pelea=gasto; cfg.ciclos=ciclos
    cfg.disp_nido = disp_nido
    rng = np.random.default_rng(seed); pob = Pob(cfg, rng)
    pob.tam = np.clip(1+rng.normal(0,var_tam,pob.n()),0.3,3.0)
    fruta = np.full(cfg.L, cfg.K_fruta)
    reg = dict(escalada=[],peleas=[],n=[],p0=[],w_cre=[],w_tes=[],olvido=[],tam=[],
               err_info=[],err_arb=[],prej_arb=[],real_arb=[],prej_info=[],
               real_info=[],div_nidos=[],conf=[],repro_info=[],repro_arb=[],
               sd_p0=[],frac_halcon=[],frac_paloma=[],
               sd_w_cre=[],sd_w_tes=[],sd_olvido=[],sd_tam=[])
    for c in range(ciclos):
        fruta = un_ciclo(pob, fruta, rng, reg)

        # diferencial de seleccion: tasa de reproduccion REALIZADA por clase
        # de marca, no la creencia sobre ella. Replica el criterio de
        # reproducir() (coste_vivir/edad ya aplicados, N_max no) porque hay
        # que medirlo ANTES de que reproducir() reordene los arreglos.
        e_ef = pob.e - cfg.coste_vivir
        edad_ef = pob.edad + 1
        va_a_repro = (e_ef > 0) & (edad_ef <= cfg.edad_max) & (e_ef > cfg.e_repro)
        cl, tg = pob.clase_tam(), pob.tag
        def _tasa(mk):
            return np.array([va_a_repro[mk == k].mean() if (mk == k).any() else np.nan
                              for k in range(cfg.n_val)])
        ri, ra = float(np.nanstd(_tasa(cl))), float(np.nanstd(_tasa(tg)))

        reproducir(pob, rng, reg)
        if pob.n() < 10: break
        reg["n"].append(pob.n())
        reg["repro_info"].append(ri); reg["repro_arb"].append(ra)
        # dispersion de rasgos DENTRO de la corrida (no entre semillas): la
        # misma media puede ser una poblacion homogenea o una partida en dos
        # extremos que promedian a lo mismo. La media sola no distingue una
        # de la otra para NINGUN rasgo heredable, no solo p0.
        for r,arr in (("p0",pob.p0),("w_cre",pob.w_cre),("w_tes",pob.w_tes),
                      ("olvido",pob.olvido),("tam",pob.tam)):
            reg[r].append(float(arr.mean()))
            reg["sd_" + r].append(float(arr.std(ddof=1)) if pob.n() > 1 else 0.0)
        # el equilibrio mixto de Halcon-Paloma en particular admite dos
        # soluciones con la MISMA media de p0: todos juegan la estrategia
        # mixta p*, o la poblacion se divide en halcones y palomas puros que
        # promedian a p*. sd_p0 (ya cubierto arriba) mas las fracciones en
        # los extremos distinguen cual de las dos es.
        reg["frac_halcon"].append(float((pob.p0 > 0.8).mean()))
        reg["frac_paloma"].append(float((pob.p0 < 0.2).mean()))
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
