"""Corrida exploratoria del canal de senal de amenaza (mundo.py, cfg.senales).

Pregunta: ¿la senal termina siendo honesta, enganosa, o pura ruido? Se sigue
la MISMA logica que prejuicio.py y tipos.py -- nada de promediar
trayectorias entre semillas que puedan quedar en regimenes distintos, medir
la honestidad sobre acciones reales (no sobre lo que los agentes creen), y
reportar semilla por semilla antes que un unico numero agregado.

PRELIMINAR: esta es la primera corrida sobre el mecanismo recien construido,
no una validacion contra una referencia analitica (no hay una para este
juego en particular, a diferencia de Halcon-Paloma). El numero de semillas
y ciclos default es un punto de partida razonable (misma escala que
prejuicio.py/tipos.py), no un umbral con el que se decidio de antemano que
alcanza para concluir algo.
"""
import numpy as np, json, argparse
from mundo import Cfg, Pob, un_ciclo, reproducir


def corre(ciclos=2000, seed=0, gasto=1.5, var_tam=0.5, marcas=True, disp_nido=0.0):
    cfg = Cfg(); cfg.marcas = marcas; cfg.senales = True; cfg.gasto_pelea = gasto
    cfg.ciclos = ciclos; cfg.disp_nido = disp_nido
    rng = np.random.default_rng(seed)
    pob = Pob(cfg, rng)
    pob.tam = np.clip(1 + rng.normal(0, var_tam, pob.n()), 0.3, 3.0)
    fruta = np.full(cfg.L, cfg.K_fruta)
    reg = dict(escalada=[], peleas=[], n=[], p_esc=[], p_ced=[], w_senal=[],
               sd_p_esc=[], sd_p_ced=[], sd_w_senal=[])
    for c in range(ciclos):
        fruta = un_ciclo(pob, fruta, rng, reg)
        reproducir(pob, rng, reg)
        if pob.n() < 10:
            break
        reg["n"].append(pob.n())
        reg["p_esc"].append(float(pob.p_senal_si_escala.mean()))
        reg["p_ced"].append(float(pob.p_senal_si_cede.mean()))
        reg["w_senal"].append(float(pob.w_senal.mean()))
        reg["sd_p_esc"].append(float(pob.p_senal_si_escala.std(ddof=1)) if pob.n() > 1 else 0.0)
        reg["sd_p_ced"].append(float(pob.p_senal_si_cede.std(ddof=1)) if pob.n() > 1 else 0.0)
        reg["sd_w_senal"].append(float(pob.w_senal.std(ddof=1)) if pob.n() > 1 else 0.0)
    return cfg, reg, pob


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, default=6)
    ap.add_argument("--ciclos", type=int, default=2000)
    ap.add_argument("--disp_nido", type=float, default=0.0)
    A = ap.parse_args()

    out = []
    u = slice(-100, None)
    for s in range(A.semillas):
        cfg, reg, pob = corre(ciclos=A.ciclos, seed=s, disp_nido=A.disp_nido)
        out.append(reg)
        m = np.array(reg["mostro"][u]); me = np.array(reg["mostro_escalo"][u])
        nm = np.array(reg["no_mostro"][u]); nme = np.array(reg["no_mostro_escalo"][u])
        p_mostro = me.sum() / max(1, m.sum())
        p_no = nme.sum() / max(1, nm.sum())
        print(f"semilla {s}: n_final={reg['n'][-1]:4d} ciclos={len(reg['n']):4d}  "
              f"p_esc={np.mean(reg['p_esc'][u]):.3f}(sd={np.mean(reg['sd_p_esc'][u]):.3f})  "
              f"p_ced={np.mean(reg['p_ced'][u]):.3f}(sd={np.mean(reg['sd_p_ced'][u]):.3f})  "
              f"w_senal={np.mean(reg['w_senal'][u]):.3f}(sd={np.mean(reg['sd_w_senal'][u]):.3f})  "
              f"P(escalo|mostro)={p_mostro:.3f}  P(escalo|no_mostro)={p_no:.3f}", flush=True)
        # checkpoint por semilla: una corrida larga no deberia perder todo si
        # el proceso muere a mitad de camino.
        json.dump(out, open("senales.json", "w"))
    print("escrito senales.json")
