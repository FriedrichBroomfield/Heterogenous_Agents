"""tipos.py -- 4 combinaciones de rasgos fijadas al inicio: ve si el aprendizaje
las ajusta o las hace desaparecer.

Diseno 2x2: propension a escalar (alta/baja) x cuanto pesa el testimonio del
nido (alto/bajo). Es la cruza directa de los dos rasgos que estan en el
centro de los problemas abiertos 1 y 2 de la etapa 1 (w_testimonio no
converge y varia entre semillas).

A diferencia de prejuicio.py, aca no se parte de rasgos uniformes al azar:
la poblacion inicial se reparte en 4 grupos iguales, cada uno con un vector
de rasgos fijo. La MUTACION SIGUE ACTUANDO sobre los descendientes (misma
regla que el resto del modelo, mut=0.05 -- no es una tasa chica) -- lo que
se seguiria no es que el rasgo se quede fijo, sino que fraccion de la
poblacion desciende de cada arquetipo fundador. Si una combinacion es
mejor, su fraccion deberia crecer con el tiempo; si el rasgo no importa
para la fitness, las 4 fracciones deberian solo flotar por deriva (random
walk en el simplex, sin tendencia neta).

OJO -- "gano" es sobre LINAJE, no sobre el valor del rasgo. Como la
mutacion sigue activa, un linaje que gana demograficamente puede derivar
lejos del valor fundador (ya sabemos por el experimento de invasion de p0
que el optimo real esta en ~0.35-0.40, no en 0.15 ni 0.85 -- los valores
fundadores de este archivo). Por eso reg["p0_por_tipo"] y
reg["w_tes_por_tipo"] rastrean el valor medio del rasgo DENTRO de cada
linaje, no solo su fraccion de poblacion: sin esto no se puede distinguir
"el arquetipo fundador es mejor" de "un linaje sobrevivio y evoluciono
hacia otra cosa".
"""
import numpy as np, json, sys, argparse
from mundo import Cfg, Pob, un_ciclo, reproducir

# nombre                     p0    w_cre  w_tes  olvido
ARQUETIPOS = [
    ("halcon_testimoniante",  0.85, 1.2,   0.4,   0.9),
    ("halcon_solitario",      0.85, 1.2,   0.02,  0.9),
    ("paloma_testimoniante",  0.15, 1.2,   0.4,   0.9),
    ("paloma_solitaria",      0.15, 1.2,   0.02,  0.9),
]


def corre(ciclos=1200, seed=0, gasto=1.5, var_tam=0.5, marcas=True, disp_nido=0.0,
          corte_dominancia=False, umbral_dom=1.0, ventana_dom=200):
    """corte_dominancia: si True, corta la corrida apenas un arquetipo se
    mantiene por encima de umbral_dom durante ventana_dom ciclos seguidos
    -- ahorra ciclos cuando la pregunta es solo "quien gana", pero trunca
    la deriva post-fijacion de p0_por_tipo/w_tes_por_tipo. Por eso NO es el
    default: para ver convergencia de disposiciones de aprendizaje hace
    falta la corrida completa.

    umbral_dom=1.0 (fijacion exacta, no un umbral laxo) es deliberado: tipo
    se hereda SIN mutacion (a diferencia de p0/w_tes/etc.), asi que una vez
    que un arquetipo llega a 0 individuos es IRREVERSIBLE -- no hay como
    reintroducirlo. Llegar a 1.0 ya es un evento permanente; ventana_dom
    es solo margen de seguridad contra un conteo transitorio raro, no
    porque 1.0 pueda revertirse."""
    cfg = Cfg(); cfg.marcas = marcas; cfg.gasto_pelea = gasto; cfg.ciclos = ciclos
    cfg.disp_nido = disp_nido
    rng = np.random.default_rng(seed)
    pob = Pob(cfg, rng)
    n, k = pob.n(), len(ARQUETIPOS)

    # reparto EXACTO en 4 grupos iguales (no rng.integers, que da ruido
    # multinomial en la condicion inicial que no queremos mezclar con la
    # dinamica que estamos midiendo).
    grupo = np.tile(np.arange(k), n // k + 1)[:n]
    rng.shuffle(grupo)
    pob.tipo = grupo.copy()
    for i, (_, p0, w_cre, w_tes, olvido) in enumerate(ARQUETIPOS):
        sel = grupo == i
        pob.p0[sel], pob.w_cre[sel] = p0, w_cre
        pob.w_tes[sel], pob.olvido[sel] = w_tes, olvido
    pob.tam = np.clip(1 + rng.normal(0, var_tam, n), 0.3, 3.0)

    fruta = np.full(cfg.L, cfg.K_fruta)
    reg = dict(escalada=[], peleas=[], n=[], frac=[[] for _ in range(k)],
               p0_por_tipo=[[] for _ in range(k)], w_tes_por_tipo=[[] for _ in range(k)],
               corte_ciclo=None)
    racha_tipo, racha_len = None, 0
    for c in range(ciclos):
        fruta = un_ciclo(pob, fruta, rng, reg)
        reproducir(pob, rng, reg)
        if pob.n() < 10:
            break
        reg["n"].append(pob.n())
        fracs_c = np.zeros(k)
        for i in range(k):
            sel = pob.tipo == i
            f = float(sel.mean())
            fracs_c[i] = f
            reg["frac"][i].append(f)
            reg["p0_por_tipo"][i].append(float(pob.p0[sel].mean()) if sel.any() else float("nan"))
            reg["w_tes_por_tipo"][i].append(float(pob.w_tes[sel].mean()) if sel.any() else float("nan"))

        if corte_dominancia:
            top = int(np.argmax(fracs_c))
            if fracs_c[top] >= umbral_dom and top == racha_tipo:
                racha_len += 1
            else:
                racha_tipo, racha_len = top, 1
            if racha_len >= ventana_dom:
                reg["corte_ciclo"] = c
                break
    return cfg, reg, pob


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, default=6)
    ap.add_argument("--ciclos", type=int, default=1200)
    ap.add_argument("--disp_nido", type=float, default=0.0)
    ap.add_argument("--corte_dominancia", action="store_true",
                     help="cortar apenas un arquetipo domina >100 ciclos seguidos "
                          "-- ahorra tiempo pero trunca la deriva post-fijacion, "
                          "no usar si el objetivo es ver convergencia de aprendizaje")
    A = ap.parse_args()

    out = []
    for s in range(A.semillas):
        cfg, reg, pob = corre(ciclos=A.ciclos, seed=s, disp_nido=A.disp_nido,
                               corte_dominancia=A.corte_dominancia)
        out.append(reg)
        finales = "  ".join(f"{nom}={reg['frac'][i][-1]:.3f}"
                             for i, (nom, *_) in enumerate(ARQUETIPOS))
        corte = f"  CORTE en ciclo {reg['corte_ciclo']}" if reg["corte_ciclo"] is not None else ""
        print(f"semilla {s}: n_final={reg['n'][-1]:4d} ciclos={len(reg['n']):4d}  {finales}{corte}")
    json.dump(out, open("tipos.json", "w"))
    print("escrito tipos.json")
