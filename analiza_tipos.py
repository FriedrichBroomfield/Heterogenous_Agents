"""Analiza tipos.json: quien gana, semilla por semilla -- CONTEO categorico,
no promedio de fracciones.

Promediar aca reproduce el mismo error que el proyecto ya evita en
prejuicio.py: si en una semilla gana "paloma_testimoniante" y en otra gana
"paloma_solitaria", la fraccion promedio "50/50" describe un regimen que no
existe en NINGUNA semilla real -- exactamente lo que el docstring de
figuras2.py ya advierte para el panel, pero que antes no tenia ningun
resumen de texto, solo 6+ subgraficos para mirar a ojo uno por uno.
"""
import json, sys
import numpy as np
from tipos import ARQUETIPOS

UMBRAL_GANADOR = 0.85  # fraccion promedio (ultimos 30 ciclos) para contar "gano"


def main(path="tipos.json"):
    R = json.load(open(path))
    nombres = [a[0] for a in ARQUETIPOS]
    k = len(nombres)
    u = 30

    ganadores = []
    print(f"POR SEMILLA (fraccion promedio, ultimos {u} ciclos)\n")
    for i, r in enumerate(R):
        frac = np.array(r["frac"], float)  # (k, ciclos)
        finales = np.nanmean(frac[:, -u:], axis=1)
        linea = "  ".join(f"{nombres[j]}={finales[j]:.3f}" for j in range(k))
        top = int(np.argmax(finales))
        gano = nombres[top] if finales[top] >= UMBRAL_GANADOR else None
        ganadores.append(gano)
        etiqueta = f"gana {nombres[top]}" if gano else "sin ganador claro"
        print(f"  semilla {i}: {linea}   -> {etiqueta}")

    print(f"\nCONTEO DE GANADORES ({len(R)} semillas, umbral={UMBRAL_GANADOR:.2f})")
    print("  CONTEO categorico, no promedio de fracciones -- ver docstring del modulo.")
    for nom in nombres:
        c = ganadores.count(nom)
        print(f"  {nom:22s} gano outright en {c}/{len(R)} semillas")
    indecisas = ganadores.count(None)
    if indecisas:
        print(f"  {'(sin ganador claro)':22s} {indecisas}/{len(R)} semillas")

    if len(set(g for g in ganadores if g is not None)) > 1:
        print("\n  >1 arquetipo distinto gana segun la semilla: el resultado depende")
        print("  del regimen, no hay un ganador universal. No promediar esto.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tipos.json")
