"""
tipos.py -- 4 combinaciones de rasgos fijadas al inicio: ve si el aprendizaje
las ajusta o las hace desaparecer.

Diseno 2x2: propension a escalar (alta/baja) x cuanto pesa el testimonio del
nido (alto/bajo). Es la cruza directa de los dos rasgos que estan en el
centro de los problemas abiertos 1 y 2 de la etapa 1 (w_testimonio no
converge y varia entre semillas).

A diferencia de prejuicio.py, aca no se parte de rasgos uniformes al azar:
la poblacion inicial se reparte en 4 grupos iguales, cada uno con un vector
de rasgos fijo. La MUTACION SIGUE ACTUANDO sobre los descendientes (misma
regla que el resto del modelo) -- lo que se seguiria no es que el rasgo se
quede fijo, sino que fraccion de la poblacion desciende de cada arquetipo
fundador. Si una combinacion es mejor, su fraccion deberia crecer con el
tiempo; si el rasgo no importa para la fitness, las 4 fracciones deberian
solo flotar por deriva (random walk en el simplex, sin tendencia neta).
"""
import numpy as np, json, sys
from mundo import Cfg, Pob, un_ciclo, reproducir

# nombre                     p0    w_cre  w_tes  olvido
ARQUETIPOS = [
    ("halcon_testimoniante",  0.85, 1.2,   0.4,   0.9),
    ("halcon_solitario",      0.85, 1.2,   0.02,  0.9),
    ("paloma_testimoniante",  0.15, 1.2,   0.4,   0.9),
    ("paloma_solitaria",      0.15, 1.2,   0.02,  0.9),
]


def corre(ciclos=1200, seed=0, gasto=1.5, var_tam=0.5, marcas=True):
    cfg = Cfg(); cfg.marcas = marcas; cfg.gasto_pelea = gasto; cfg.ciclos = ciclos
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
    reg = dict(escalada=[], peleas=[], n=[], frac=[[] for _ in range(k)])
    for c in range(ciclos):
        fruta = un_ciclo(pob, fruta, rng, reg)
        reproducir(pob, rng, reg)
        if pob.n() < 10:
            break
        reg["n"].append(pob.n())
        for i in range(k):
            reg["frac"][i].append(float((pob.tipo == i).mean()))
    return cfg, reg, pob


if __name__ == "__main__":
    semillas = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    ciclos = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
    out = []
    for s in range(semillas):
        cfg, reg, pob = corre(ciclos=ciclos, seed=s)
        out.append(reg)
        finales = "  ".join(f"{nom}={reg['frac'][i][-1]:.3f}"
                             for i, (nom, *_) in enumerate(ARQUETIPOS))
        print(f"semilla {s}: n_final={reg['n'][-1]:4d} ciclos={len(reg['n']):4d}  {finales}")
    json.dump(out, open("tipos.json", "w"))
    print("escrito tipos.json")
