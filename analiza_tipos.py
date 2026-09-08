"""Analiza tipos.json: quien gana, semilla por semilla -- CONTEO categorico,
no promedio de fracciones. Ademas: fraccion promedio final entre MUCHAS
semillas contra el neutral (eso si es un promedio valido, ver mas abajo), y
deriva del valor del rasgo dentro de cada linaje (porque la mutacion sigue
activa: "gano" es sobre linaje, no sobre el valor con el que arranco).

Por que promediar fracciones aca es distinto del error que evita el conteo
de arriba: promediar TRAYECTORIAS entre semillas con ganadores distintos
implica una coexistencia que ninguna semilla muestra -- eso es lo que
evita el conteo categorico. Promediar el RESULTADO FINAL entre muchas
semillas para estimar "que fraccion de universos replicados terminan con
este arquetipo dominante" es otra cosa: es un estimador de probabilidad de
fijacion (misma logica que el experimento de invasion de p0 en mundo.py),
y se interpreta contra el neutral (1/k si ningun arquetipo tuviera ventaja
real), no como "el estado tipico de una poblacion".
"""
import json, sys
import numpy as np
from tipos import ARQUETIPOS

UMBRAL_GANADOR = 0.85  # fraccion promedio (ultimos 30 ciclos) para contar "gano"


def main(path="tipos.json"):
    R = json.load(open(path))
    nombres = [a[0] for a in ARQUETIPOS]
    p0_fundador = [a[1] for a in ARQUETIPOS]
    wtes_fundador = [a[3] for a in ARQUETIPOS]
    k = len(nombres)
    u = 30
    n = len(R)

    ganadores, finales_todas = [], []
    print(f"POR SEMILLA (fraccion promedio, ultimos {u} ciclos)\n")
    for i, r in enumerate(R):
        frac = np.array(r["frac"], float)  # (k, ciclos)
        finales = np.nanmean(frac[:, -u:], axis=1)
        finales_todas.append(finales)
        linea = "  ".join(f"{nombres[j]}={finales[j]:.3f}" for j in range(k))
        top = int(np.argmax(finales))
        gano = nombres[top] if finales[top] >= UMBRAL_GANADOR else None
        ganadores.append(gano)
        etiqueta = f"gana {nombres[top]}" if gano else "sin ganador claro"
        corte = f"  [corte en ciclo {r['corte_ciclo']}]" if r.get("corte_ciclo") is not None else ""
        print(f"  semilla {i}: {linea}   -> {etiqueta}{corte}")

    print(f"\nCONTEO DE GANADORES ({n} semillas, umbral={UMBRAL_GANADOR:.2f})")
    print("  CONTEO categorico, no promedio de fracciones -- ver docstring del modulo.")
    for nom in nombres:
        c = ganadores.count(nom)
        print(f"  {nom:22s} gano outright en {c}/{n} semillas")
    indecisas = ganadores.count(None)
    if indecisas:
        print(f"  {'(sin ganador claro)':22s} {indecisas}/{n} semillas")
    if len(set(g for g in ganadores if g is not None)) > 1:
        print("  >1 arquetipo distinto gana segun la semilla: el resultado depende")
        print("  del regimen, no hay un ganador universal. No promediar esto.")

    print(f"\nFRACCION FINAL PROMEDIO ENTRE SEMILLAS vs. NEUTRAL (1/{k}={1/k:.3f})")
    print("  esto SI es un promedio valido: estima prob. de fijacion, no un")
    print("  estado tipico. Muy por encima del neutral = ventaja real; cerca")
    print("  del neutral = el arquetipo se comporta como si fuera indiferente.")
    if n < 8:
        print(f"  *** n={n} semillas es poco para esto -- {UMBRAL_GANADOR} es ruidoso con")
        print("      menos de ~15-20 semillas. Tratar como orientativo, no concluyente. ***")
    M = np.array(finales_todas)  # (semillas, k)
    for j, nom in enumerate(nombres):
        media = np.nanmean(M[:, j])
        sem = np.nanstd(M[:, j], ddof=1) / np.sqrt(n) if n > 1 else float("nan")
        z = (media - 1 / k) / sem if sem > 1e-9 else float("nan")
        flag = "  <-- ventaja real" if not np.isnan(z) and z > 2 else \
               ("  <-- desventaja real" if not np.isnan(z) and z < -2 else "")
        print(f"  {nom:22s} media={media:.3f}  z vs neutral={z:+.2f}{flag}")

    print("\nDERIVA DEL RASGO DENTRO DEL LINAJE (valor fundador vs. valor actual)")
    print("  'gano' es sobre linaje (pob.tipo), no sobre el valor del rasgo -- la")
    print("  mutacion (mut=0.05) sigue activa. Un linaje puede ganar demograficamente")
    print("  y derivar lejos de con que arranco. Si |deriva| es grande, el arquetipo")
    print("  fundador ya no describe a quienes lo componen.")
    for i, r in enumerate(R):
        if "p0_por_tipo" not in r:
            print(f"  semilla {i}: (sin p0_por_tipo, json viejo -- correr tipos.py de nuevo)")
            continue
        print(f"  semilla {i}:")
        for j, nom in enumerate(nombres):
            p0s = np.array(r["p0_por_tipo"][j], float)
            wts = np.array(r["w_tes_por_tipo"][j], float)
            if np.all(np.isnan(p0s[-u:])):
                continue  # linaje extinto -- nada que reportar
            p0_final = np.nanmean(p0s[-u:])
            wt_final = np.nanmean(wts[-u:])
            print(f"    {nom:22s} p0: {p0_fundador[j]:.2f} -> {p0_final:.2f}  "
                  f"({p0_final - p0_fundador[j]:+.2f})   "
                  f"w_tes: {wtes_fundador[j]:.2f} -> {wt_final:.2f} "
                  f"({wt_final - wtes_fundador[j]:+.2f})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tipos.json")
