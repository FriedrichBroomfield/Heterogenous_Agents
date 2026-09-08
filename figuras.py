"""Panel de la etapa 1. Lineas finas por semilla + media agregada encima.

Las semillas terminan en regimenes distintos, asi que una banda min-max que
las mezcla no mide incertidumbre real. Por eso la media agregada (linea
gruesa, banda = SEM entre semillas) responde una pregunta DISTINTA de la que
responden las lineas finas: si la media converge o sigue moviendose, no si
todas las semillas convergen al MISMO lugar. Esto ultimo se audita con el CV
del tramo final entre semillas -- si es alto, se marca en el propio grafico
en vez de esconderlo (la regla de no promediar regimenes distintos sigue
valiendo para lo que esa media *significa*, no para si se puede calcular).
El sombreado de "prejuicio injustificado" de la version anterior solo se
pintaba donde la media favorecia la conclusion; eso sigue sin hacerse aca.
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


def overlay_media(ax, S, key, transform=None, color="#111111", k=15, label="media"):
    """Linea gruesa = media entre semillas por ciclo (suavizada); banda =
    SEM entre semillas. Responde 'la tendencia agregada se aplana o sigue
    moviendose', que es una pregunta distinta de si las semillas coinciden.
    Si no coinciden (CV alto), lo dice en el propio panel."""
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
    ax.plot(x, media, color=color, lw=2.3, zorder=5, label=label)
    ax.fill_between(x, media - sem, media + sem, color=color, alpha=.15, zorder=4, lw=0)
    if label is None:
        return  # panel de dos series: el aviso de CV va una sola vez, ver caller
    cv = cv_final(S, key, transform)
    if not np.isnan(cv) and cv > .25:
        ax.text(.02, .03, f"CV entre semillas={cv:.2f} (tramo final): las semillas\n"
                           "no coinciden -- la media es una tendencia, no un valor unico",
                transform=ax.transAxes, fontsize=6.2, color="#c0392b", va="bottom")


def main(salida="panel1.png"):
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                         "grid.alpha": .25, "axes.spines.top": False,
                         "axes.spines.right": False})
    val = json.load(open("validacion.json"))
    S = json.load(open("prejuicio.json"))["con_marcas"]
    n = len(S)
    fig, ax = plt.subplots(3, 3, figsize=(13.5, 10.8))
    leg = dict(frameon=False, fontsize=7, ncol=min(4, n))

    a = ax[0, 0]
    C = [r["C"] for r in val]
    a.plot(C, [r["pred"] for r in val], "k--", lw=1.6, label="predicción V/C")
    a.errorbar(C, [r["obs"] for r in val], yerr=[r["sd"] for r in val],
               fmt="o-", color="#c0392b", ms=5, lw=1.4, label="simulación")
    a.set_xlabel("costo de pelea C"); a.set_ylabel("tasa de escalada")
    a.set_title("1. Validación: Halcón-Paloma"); a.legend(frameon=False, fontsize=8)

    a = ax[0, 1]
    for i, r in enumerate(S):
        a.plot(suave(r["escalada"]), color=COL[i % len(COL)], lw=1, alpha=.5,
               label=f"s{i}" if i < 4 else None)
    overlay_media(a, S, "escalada")
    a.axhline(1 / 3, ls="--", c="k", lw=1)
    a.text(5, .345, "V/C = 0.33", fontsize=7)
    a.set_xlabel("ciclo"); a.set_ylabel("tasa de escalada")
    a.set_title(f"2. Escalada ({n} semillas)"); a.legend(**leg)

    a = ax[0, 2]
    for r in S:
        a.plot(suave(r["err_info"]), color="#27ae60", lw=.7, alpha=.35)
        a.plot(suave(r["err_arb"]), color="#e67e22", lw=.7, alpha=.35)
    overlay_media(a, S, "err_info", color="#145a32", label=None)
    overlay_media(a, S, "err_arb", color="#a04000", label=None)
    a.plot([], [], color="#27ae60", label="informativa")
    a.plot([], [], color="#e67e22", label="arbitraria")
    a.set_xlabel("ciclo"); a.set_ylabel("|creencia − realidad|")
    a.set_title("3. Error de creencia"); a.legend(frameon=False, fontsize=8)

    dif_info = lambda r: np.array(r["prej_info"], float) - np.array(r["real_info"], float)
    a = ax[1, 0]
    for i, r in enumerate(S):
        a.plot(suave(dif_info(r)), color=COL[i % len(COL)], lw=1, alpha=.5,
               label=f"s{i}" if i < 4 else None)
    overlay_media(a, S, "prej_info", transform=dif_info)
    a.axhline(0, ls="--", c="k", lw=1.3)
    a.set_xlabel("ciclo"); a.set_ylabel("discriminan − real")
    a.set_title("4. Marca INFORMATIVA\nbajo cero = sub-usan la señal válida")
    a.legend(**leg)

    # La razon tiene denominador chico y explota. Se grafica la DIFERENCIA,
    # que es la misma pregunta sin la inestabilidad.
    dif_arb = lambda r: np.array(r["prej_arb"], float) - np.array(r["real_arb"], float)
    a = ax[1, 1]
    for i, r in enumerate(S):
        a.plot(suave(dif_arb(r)), color=COL[i % len(COL)], lw=1, alpha=.5,
               label=f"s{i}" if i < 4 else None)
    overlay_media(a, S, "prej_arb", transform=dif_arb)
    a.axhline(0, ls="--", c="k", lw=1.3)
    a.set_xlabel("ciclo"); a.set_ylabel("discriminan − real")
    a.set_title("5. Marca ARBITRARIA\nsobre cero = prejuicio injustificado")
    a.legend(**leg)

    # Escalar es individualmente mejor pero, si todos lo hacen, mas
    # encuentros se resuelven a pelea: energia perdida por los dos lados
    # e injuria, no solo fruta repartida distinto. Esto es el costo a la
    # convivencia que "tasa de escalada" (panel 2) no muestra.
    a = ax[1, 2]
    for i, r in enumerate(S):
        a.plot(suave(r["peleas"]), color=COL[i % len(COL)], lw=1, alpha=.5,
               label=f"s{i}" if i < 4 else None)
    overlay_media(a, S, "peleas")
    a.set_xlabel("ciclo"); a.set_ylabel("fracción de encuentros")
    a.set_title("6. Costo de convivencia\nencuentros que terminan en pelea")
    a.legend(**leg)

    # Disposiciones: separado por rasgo, no todos en el mismo eje. w_cre
    # opera en una escala distinta (puede pasar de 1.0) a w_tes/olvido
    # (acotados en [0,1]); mezclarlos aplastaba a los dos ultimos.
    a = ax[2, 0]
    for i, r in enumerate(S):
        a.plot(suave(r["w_cre"]), color=COL[i % len(COL)], lw=1, alpha=.5,
               label=f"s{i}" if i < 4 else None)
    overlay_media(a, S, "w_cre")
    a.set_xlabel("ciclo"); a.set_ylabel("peso de la creencia")
    a.set_title("7. Peso de la creencia"); a.legend(**leg)

    a = ax[2, 1]
    for r in S:
        a.plot(suave(r["w_tes"]), color="#2c3e50", lw=.7, alpha=.35)
        a.plot(suave(r["olvido"]), color="#d35400", lw=.7, alpha=.35)
    overlay_media(a, S, "w_tes", color="#0b1f33", label=None)
    overlay_media(a, S, "olvido", color="#7a2600", label=None)
    a.plot([], [], color="#2c3e50", label="peso del testimonio")
    a.plot([], [], color="#d35400", label="tasa de olvido")
    a.set_xlabel("ciclo"); a.set_ylabel("valor medio del rasgo")
    a.set_title("8. Testimonio y olvido"); a.legend(frameon=False, fontsize=8)

    # Diferencial de seleccion: no es creencia ni verdad de la pelea
    # individual, es quien efectivamente se reprodujo mas por clase.
    a = ax[2, 2]
    for r in S:
        a.plot(suave(r["repro_info"]), color="#27ae60", lw=.7, alpha=.35)
        a.plot(suave(r["repro_arb"]), color="#e67e22", lw=.7, alpha=.35)
    overlay_media(a, S, "repro_info", color="#145a32", label=None)
    overlay_media(a, S, "repro_arb", color="#a04000", label=None)
    a.plot([], [], color="#27ae60", label="informativa")
    a.plot([], [], color="#e67e22", label="arbitraria")
    a.set_xlabel("ciclo"); a.set_ylabel("dispersión tasa de reproducción")
    a.set_title("9. Diferencial de selección\npor clase de marca (realizado, no creído)")
    a.legend(frameon=False, fontsize=8)

    fig.suptitle(f"Recolección de fruta · etapa 1 · {n} semillas", fontsize=11, y=.995)
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(salida, bbox_inches="tight")
    print(f"escrito {salida}")


if __name__ == "__main__":
    main()
