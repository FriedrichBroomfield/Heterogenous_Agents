"""Validacion Halcon-Paloma: la tasa de escalada debe seguir a min(1, V/C)."""
import numpy as np, json
from mundo import Cfg, Pob, un_ciclo, reproducir

def corre(gasto, ciclos=150, seed=0, marcas=False, var_tam=0.3):
    cfg = Cfg(); cfg.marcas = marcas; cfg.gasto_pelea = gasto
    rng = np.random.default_rng(seed)
    pob = Pob(cfg, rng)
    pob.tam = np.clip(1 + rng.normal(0, var_tam, pob.n()), 0.3, 3.0)
    fruta = np.full(cfg.L, cfg.K_fruta)
    reg = dict(escalada=[], peleas=[])
    for c in range(ciclos):
        fruta = un_ciclo(pob, fruta, rng, reg)
        reproducir(pob, rng)
        if pob.n() < 10: break
    return cfg, reg, pob

if __name__ == "__main__":
    res = []
    for gasto in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        obs = []
        for s in range(3):
            cfg, reg, pob = corre(gasto, seed=s)
            if len(reg["escalada"]) > 60:
                obs.append(np.mean(reg["escalada"][-50:]))
        C = 2*gasto; pred = min(1.0, cfg.V_bocado / C)
        res.append(dict(C=C, pred=pred, obs=float(np.mean(obs)), sd=float(np.std(obs)),
                        n=len(obs)))
        print(f"  C={C:4.1f}  V/C={cfg.V_bocado/C:5.3f}  prediccion={pred:5.3f}  "
              f"observado={np.mean(obs):5.3f} +- {np.std(obs):.3f}  (n={len(obs)})")
    json.dump(res, open("validacion.json","w"))
