"""Panel de la etapa 1. Lineas finas por semilla + media agregada encima.

Las semillas terminan en regimenes distintos, asi que una banda min-max que
las mezcla no mide incertidumbre real. Por eso la media agregada (linea
gruesa, banda ESTRECHA = SEM entre semillas) responde una pregunta DISTINTA
de la que responden las lineas finas: si la media converge o sigue
moviendose, no si todas las semillas convergen al MISMO lugar. Esto ultimo
se audita con el CV del tramo final entre semillas -- si es alto, se marca
en el propio grafico en vez de esconderlo.

Para p0, w_cre, w_tes y olvido hay ademas una banda ANCHA = dispersion
ENTRE INDIVIDUOS dentro de cada corrida (sd_*, no confundir con la banda
angosta de arriba). La misma media puede ser una poblacion convergida a un
valor o una partida en dos extremos -- sin esta banda no se distingue.

Recorte deliberado (revision del panel, ver conversacion): se sacaron los
paneles de escalada-en-el-tiempo y costo-de-convivencia -- son derivados de
la misma serie que el panel de validacion y no discriminan nada por marca,
asi que no aportan a la pregunta de fondo (prejuicio informativo/arbitrario,
disposiciones de aprendizaje). El chequeo de "N_max frena nacimientos" se
saco de figuras2.py por la misma razon: es una pregunta binaria que no
necesita un panel entero, ver el chequeo de texto en analiza.py.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = ["#c0392b", "#2980b9", "#27ae60", "#8e44ad", "#d35400", "#16a085",
       "#7f8c8d", "#2c3e50", "#e67e22", "#1abc9c", "#9b59b6", "#34495e"]


def suave(x, k=15):
    x = np.array(x, float)
    return x if len(x) < k else np.convolve(x, np.ones(k) / k, mode="valid")


def _series(S, key, transform=None):
    out = []
    for r in S:
        if key not in r or not r[key]:
            continue
        x = np.array(r[key], float)
        if transform is not None:
            x = transform(r)
        out.append(x)
    return out


def cv_final(S, key, transform=None, u=slice(-60, None)):
    """Coeficiente de variacion entre semillas del valor final (ultimos 60
    ciclos). Es lo que decide si la media de abajo describe algo real."""
    vals = []
    for x in _series(S, key, transform):
        if len(x):
            vals.append(float(np.nanmean(x[u])))
    v = np.array(vals, float)
    if len(v) < 2 or v.mean() == 0 or np.isnan(v.mean()):
        return np.nan
    return float(v.std(ddof=1) / abs(v.mean()))


def overlay_media(ax, S, key, transform=None, color="#111111", k=15, label="media",
                   avisar_cv=True, y_cv=.03, sd_key=None):
    """Linea gruesa = media entre semillas por ciclo (suavizada); banda
    ANGOSTA = SEM entre semillas (incertidumbre de la media agregada). Si se
    pasa sd_key, se agrega ademas una banda ANCHA = dispersion tipica ENTRE
    INDIVIDUOS dentro de una corrida (media entre semillas de sd_*) -- es
    una pregunta distinta: no "cuanto se mueve la media" sino "que tan
    partida esta la poblacion detras de esa media".

    avisar_cv es INDEPENDIENTE de label a proposito: antes, los paneles con
    dos series pasaban label=None para no duplicar la leyenda, y eso
    apagaba tambien el aviso de CV -- asi que paneles como "Error de
    creencia" o "Diferencial de seleccion" nunca mostraban el aviso, aunque
    los datos lo ameritaran. Ahora cada serie se audita, tenga o no label."""
    arrs = _series(S, key, transform)
    if len(arrs) < 2:
        return
    L = min(len(a) for a in arrs)
    if L < 2:
        return
    M = np.stack([a[:L] for a in arrs])
    media = suave(np.nanmean(M, axis=0), k)
    sem = suave(np.nanstd(M, axis=0, ddof=1) / np.sqrt(M.shape[0]), k)
    x = np.arange(len(media))

    if sd_key is not None:
        sds = _series(S, sd_key)
        if len(sds) >= 2:
            Ld = min(L, min(len(s) for s in sds))
            sd_media = suave(np.nanmean(np.stack([s[:Ld] for s in sds]), axis=0), k)
            Lb = min(len(media), len(sd_media))
            xb = np.arange(Lb)
            ax.fill_between(xb, media[:Lb] - sd_media[:Lb], media[:Lb] + sd_media[:Lb],
                             color=color, alpha=.08, lw=0, zorder=2)

    ax.plot(x, media, color=color, lw=2.3, zorder=5, label=label)
    ax.fill_between(x, media - sem, media + sem, color=color, alpha=.15, zorder=4, lw=0)
    if not avisar_cv:
        return
    cv = cv_final(S, key, transform)
    if not np.isnan(cv) and cv > .25:
        ax.text(.02, y_cv, f"CV entre semillas={cv:.2f} (tramo final): las semillas\n"
                           "no coinciden -- la media es una tendencia, no un valor unico",
                transform=ax.transAxes, fontsize=6.2, color=color, va="bottom")


def main(salida="panel1.png"):
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                         "grid.alpha": .25, "axes.spines.top": False,
                         "axes.spines.right": False})
    val = json.load(open("validacion.json"))
    S = json.load(open("prejuicio.json"))["con_marcas"]
    n = len(S)
    fig, axs = plt.subplots(3, 3, figsize=(13.5, 10.8))
    ax = [axs[i // 3][i % 3] for i in range(9)]
    leg = dict(frameon=False, fontsize=7, ncol=min(4, n))

    a = ax[0]
    C = [r["C"] for r in val]
    a.plot(C, [r["pred"] for r in val], "k--", lw=1.6, label="predicción V/C")
    a.errorbar(C, [r["obs"] for r in val], yerr=[r["sd"] for r in val],
               fmt="o-", color="#c0392b", ms=5, lw=1.4, label="simulación")
    a.set_xlabel("costo de pelea C"); a.set_ylabel("tasa de escalada")
    a.set_title("1. Validación: Halcón-Paloma (dirección y orden de magnitud)")
    a.legend(frameon=False, fontsize=8)
    a.text(.03, .06, "Desplazamiento sistemático conocido (+0.04 a +0.09):\n"
                      "confirmado por Montecarlo (z=+14 a +17) e invasión\n"
                      "independiente con mutación apagada. No es un bug\n"
                      "abierto -- ver nota completa en todo.py.",
           transform=a.transAxes, fontsize=6.2, color="#7f5300", va="bottom")

    a = ax[1]
    for r in S:
        a.plot(suave(r["err_info"]), color="#27ae60", lw=.7, alpha=.35)
        a.plot(suave(r["err_arb"]), color="#e67e22", lw=.7, alpha=.35)
    overlay_media(a, S, "err_info", color="#145a32", label=None, y_cv=.03)
    overlay_media(a, S, "err_arb", color="#a04000", label=None, y_cv=.14)
    a.plot([], [], color="#27ae60", label="informativa")
    a.plot([], [], color="#e67e22", label="arbitraria")
    a.set_xlabel("ciclo"); a.set_ylabel("|creencia − realidad|")
    a.set_title("2. Error de creencia"); a.legend(frameon=False, fontsize=8)

    dif_info = lambda r: np.array(r["prej_info"], float) - np.array(r["real_info"], float)
    a = ax[2]
    for i, r in enumerate(S):
        a.plot(suave(dif_info(r)), color=COL[i % len(COL)], lw=1, alpha=.5,
               label=f"s{i}" if i < 4 else None)
    overlay_media(a, S, "prej_info", transform=dif_info)
    a.axhline(0, ls="--", c="k", lw=1.3)
    a.set_xlabel("ciclo"); a.set_ylabel("discriminan − real")
    a.set_title("3. Marca INFORMATIVA\nbajo cero = sub-usan la señal válida")
    a.legend(**leg)

    # La razon tiene denominador chico y explota. Se grafica la DIFERENCIA,
    # que es la misma pregunta sin la inestabilidad.
    dif_arb = lambda r: np.array(r["prej_arb"], float) - np.array(r["real_arb"], float)
    a = ax[3]
    for i, r in enumerate(S):
        a.plot(suave(dif_arb(r)), color=COL[i % len(COL)], lw=1, alpha=.5,
               label=f"s{i}" if i < 4 else None)
    overlay_media(a, S, "prej_arb", transform=dif_arb)
    a.axhline(0, ls="--", c="k", lw=1.3)
    a.set_xlabel("ciclo"); a.set_ylabel("discriminan − real")
    a.set_title("4. Marca ARBITRARIA\nsobre cero = prejuicio injustificado")
    a.legend(**leg)

    # Disposiciones: separado por rasgo, no todos en el mismo eje. w_cre
    # opera en una escala distinta (puede pasar de 1.0) a w_tes/olvido
    # (acotados en [0,1]); mezclarlos aplastaba a los dos ultimos. Banda
    # ancha = dispersion intra-poblacional (sd_w_cre) -- ver docstring de
    # overlay_media.
    a = ax[4]
    for i, r in enumerate(S):
        a.plot(suave(r["w_cre"]), color=COL[i % len(COL)], lw=1, alpha=.5,
               label=f"s{i}" if i < 4 else None)
    overlay_media(a, S, "w_cre", sd_key="sd_w_cre")
    a.set_xlabel("ciclo"); a.set_ylabel("peso de la creencia")
    a.set_title("5. Peso de la creencia\nbanda ancha = dispersión intra-población")
    a.legend(**leg)

    a = ax[5]
    for r in S:
        a.plot(suave(r["w_tes"]), color="#2c3e50", lw=.7, alpha=.35)
        a.plot(suave(r["olvido"]), color="#d35400", lw=.7, alpha=.35)
    overlay_media(a, S, "w_tes", color="#0b1f33", label=None, sd_key="sd_w_tes", y_cv=.03)
    overlay_media(a, S, "olvido", color="#7a2600", label=None, sd_key="sd_olvido", y_cv=.14)
    a.plot([], [], color="#2c3e50", label="peso del testimonio")
    a.plot([], [], color="#d35400", label="tasa de olvido")
    a.set_xlabel("ciclo"); a.set_ylabel("valor medio del rasgo")
    a.set_title("6. Testimonio y olvido\nbanda ancha = dispersión intra-población")
    a.legend(frameon=False, fontsize=8)

    # Diferencial de seleccion: no es creencia ni verdad de la pelea
    # individual, es quien efectivamente se reprodujo mas por clase.
    a = ax[6]
    for r in S:
        a.plot(suave(r["repro_info"]), color="#27ae60", lw=.7, alpha=.35)
        a.plot(suave(r["repro_arb"]), color="#e67e22", lw=.7, alpha=.35)
    overlay_media(a, S, "repro_info", color="#145a32", label=None, y_cv=.03)
    overlay_media(a, S, "repro_arb", color="#a04000", label=None, y_cv=.14)
    a.plot([], [], color="#27ae60", label="informativa")
    a.plot([], [], color="#e67e22", label="arbitraria")
    a.set_xlabel("ciclo"); a.set_ylabel("dispersión tasa de reproducción")
    a.set_title("7. Diferencial de selección\npor clase de marca (realizado, no creído)")
    a.legend(frameon=False, fontsize=8)

    ax[7].axis("off")
    ax[8].axis("off")

    fig.suptitle(f"Recolección de fruta · etapa 1 · {n} semillas", fontsize=11, y=.995)
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(salida, bbox_inches="tight")
    print(f"escrito {salida}")


if __name__ == "__main__":
    main()
