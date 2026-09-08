"""Panel 2: como cambia N y por que. Panel 3: arquetipos (tipos.py).

Responde tres preguntas que panel1.png no contesta:
  - crece la poblacion, se estanca, o esta pegada al tope N_max?
  - cuando el tope actua, frena a TODOS por igual (no es una poda
    dirigida a una minoria -- la seleccion pasa antes, por energia)
  - de las muertes, cuantas son "naturales" (edad_max) y cuantas por
    energia negativa (inanicion / gasto de pelea)?
Y si existe tipos.json (python tipos.py), una figura aparte con la
fraccion de poblacion de cada arquetipo por ciclo, semilla por semilla
-- sin promediar, porque la pregunta ahi es justamente si distintas
semillas quedan con ganadores distintos.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figuras import suave, COL


def _media(S, key):
    arrs = [np.array(r[key], float) for r in S if r.get(key)]
    if not arrs:
        return np.array([])
    L = min(len(a) for a in arrs)
    return suave(np.nanmean(np.stack([a[:L] for a in arrs]), axis=0))


def poblacion(path="prejuicio.json", salida="panel2.png"):
    S = json.load(open(path))["con_marcas"]
    n = len(S)
    if not S or "murio_edad" not in S[0]:
        print("prejuicio.json no tiene murio_edad/n_nacidos/tope_nmax -- "
              "correr de nuevo con el mundo.py actualizado.")
        return
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                         "grid.alpha": .25, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    a = ax[0]
    for i, r in enumerate(S):
        a.plot(r["n"], color=COL[i % len(COL)], lw=1, alpha=.5)
    a.plot(_media(S, "n"), color="#111111", lw=2.3, label="media")
    a.axhline(900, ls="--", c="#7f8c8d", lw=1, label="N_max")
    a.set_xlabel("ciclo"); a.set_ylabel("N"); a.set_title("1. Tamaño de la población")
    a.legend(frameon=False, fontsize=7)

    a = ax[1]
    W = 30
    rolls = []
    for i, r in enumerate(S):
        t = np.array(r["tope_nmax"], float)
        if len(t) < W:
            continue
        roll = np.convolve(t, np.ones(W) / W, mode="valid")
        a.plot(roll, color=COL[i % len(COL)], lw=1, alpha=.5)
        rolls.append(roll)
    if rolls:
        L = min(len(x) for x in rolls)
        media = np.nanmean(np.stack([x[:L] for x in rolls]), axis=0)
        a.plot(media, color="#111111", lw=2.3, label="media")
    a.set_ylim(-.03, 1.03)
    a.set_xlabel("ciclo"); a.set_ylabel(f"fracción de ciclos con tope activo (ventana {W})")
    a.set_title("2. ¿El techo N_max frena nacimientos?\nsi actúa, frena a TODOS por igual, no es selectivo")
    a.legend(frameon=False, fontsize=7)

    a = ax[2]
    fracs = []
    for r in S:
        me = np.array(r["murio_edad"], float)
        mh = np.array(r["murio_energia"], float)
        tot = me + mh
        f = np.divide(me, tot, out=np.full_like(tot, np.nan), where=tot > 0)
        fracs.append(f)
    L = min(len(f) for f in fracs)
    M = np.stack([f[:L] for f in fracs])
    f_edad = suave(np.nanmean(M, axis=0))
    x = np.arange(len(f_edad))
    a.fill_between(x, 0, f_edad, color="#2980b9", alpha=.55, label="murió por edad (natural)")
    a.fill_between(x, f_edad, 1, color="#c0392b", alpha=.55, label="murió por energía negativa")
    a.set_ylim(0, 1)
    a.set_xlabel("ciclo"); a.set_ylabel("fracción de las muertes")
    a.set_title("3. Causa de muerte\npromedio entre semillas")
    a.legend(frameon=False, fontsize=7, loc="upper right")

    fig.suptitle(f"Dinámica poblacional · {n} semillas", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(salida, bbox_inches="tight")
    print(f"escrito {salida}")


def tipos_fig(path="tipos.json", salida="panel3.png"):
    if not os.path.exists(path):
        print(f"{path} no existe -- correr 'python tipos.py' primero. Se omite panel3.")
        return
    R = json.load(open(path))
    n = len(R)
    from tipos import ARQUETIPOS
    nombres = [a[0] for a in ARQUETIPOS]
    colores = ["#c0392b", "#e67e22", "#2980b9", "#27ae60"]

    plt.rcParams.update({"figure.dpi": 130, "font.size": 8, "axes.grid": True,
                         "grid.alpha": .25, "axes.spines.top": False,
                         "axes.spines.right": False})
    cols = min(3, n)
    filas = (n + cols - 1) // cols
    fig, axs = plt.subplots(filas, cols, figsize=(4.6 * cols, 3.6 * filas), squeeze=False)

    for i, r in enumerate(R):
        a = axs[i // cols][i % cols]
        frac = np.array(r["frac"], float)  # (4, ciclos)
        x = np.arange(frac.shape[1])
        base = np.zeros(frac.shape[1])
        for k in range(len(nombres)):
            a.fill_between(x, base, base + frac[k], color=colores[k], alpha=.75,
                            label=nombres[k] if i == 0 else None)
            base = base + frac[k]
        a.set_ylim(0, 1)
        a.set_title(f"semilla {i}  (n_final={r['n'][-1]})")
        a.set_xlabel("ciclo")
        if i % cols == 0:
            a.set_ylabel("fracción de la población")
    for j in range(n, filas * cols):
        axs[j // cols][j % cols].axis("off")

    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               fontsize=8, bbox_to_anchor=(.5, 1.04))
    fig.suptitle("Arquetipos fundadores: fracción de la población por ciclo\n"
                  "(sin promediar entre semillas -- la pregunta es si el ganador es el mismo)",
                  fontsize=10, y=1.11)
    fig.tight_layout()
    fig.savefig(salida, bbox_inches="tight")
    print(f"escrito {salida}")


if __name__ == "__main__":
    poblacion()
    tipos_fig()
