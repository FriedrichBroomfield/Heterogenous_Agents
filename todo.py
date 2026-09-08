#!/usr/bin/env python3
"""
todo.py -- corre la etapa 1 completa de punta a punta.

ETAPAS
  1  validacion   barre el costo de pelea y compara contra Halcon-Paloma.
                  Si la escalada no sigue a V/C, hay un bug y se corta ahi.
  2  prejuicio    corre N semillas con y sin marcas, midiendo por ciclo la
                  discriminacion sobre cada marca y la diferencia real.
  3  analisis     tabla POR SEMILLA (no promediando: las semillas terminan
                  en regimenes distintos) mas los tres estadisticos que
                  deciden el experimento.
  4  figuras      panel1.png, una linea por semilla.

USO
  python todo.py --test           # ~2 min, solo verifica que corre
  python todo.py                  # 12 semillas x 1200 ciclos, ~25-35 min
  python todo.py --semillas 24 --ciclos 2000
  python todo.py --saltar 1       # reusa validacion.json existente

Escribe validacion.json, prejuicio.json y panel1.png.
Pegame la salida de consola completa y el panel.
"""
import argparse, json, sys, time
import numpy as np


def barra(frac, ancho=30):
    n = int(max(0., min(1., frac)) * ancho)
    return "#" * n + "." * (ancho - n)


def reloj(s):
    s = int(s)
    return f"{s//60:02d}:{s%60:02d}"


def etapa_validacion(rapido):
    import validar
    print("\n" + "=" * 68)
    print("ETAPA 1/4  VALIDACION contra Halcon-Paloma")
    print("  la tasa de escalada debe seguir a min(1, V/C).")
    print("  Si se despega, hay un bug y no vale la pena seguir.")
    print("=" * 68, flush=True)
    gastos = (0.75, 1.5, 3.0) if rapido else (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    semillas = 2 if rapido else 3
    res, t0 = [], time.time()
    for j, g in enumerate(gastos):
        obs = []
        for s in range(semillas):
            cfg, reg, pob = validar.corre(g, ciclos=120 if rapido else 150, seed=s)
            if len(reg["escalada"]) > 60:
                obs.append(np.mean(reg["escalada"][-50:]))
        C = 2 * g
        pred = min(1.0, cfg.V_bocado / C)
        o = float(np.mean(obs)) if obs else float("nan")
        res.append(dict(C=C, pred=pred, obs=o,
                        sd=float(np.std(obs)) if obs else 0.0, n=len(obs)))
        print(f"  C={C:4.1f}  predicción={pred:5.3f}  observado={o:5.3f} "
              f"±{np.std(obs) if obs else 0:.3f}   [{reloj(time.time()-t0)}]",
              flush=True)
    json.dump(res, open("validacion.json", "w"))
    v = [r for r in res if r["pred"] < 0.99]
    err = float(np.mean([abs(r["obs"] - r["pred"]) for r in v])) if v else float("nan")
    print(f"\n  error medio en la zona de equilibrio mixto: {err:.3f}")
    if err > 0.12:
        print("  *** LA VALIDACION FALLA. No sigas: revisá el modelo. ***")
        return False
    print("  validación OK\n")
    return True


def etapa_prejuicio(semillas, ciclos, disp_nido=0.0):
    import prejuicio
    print("=" * 68)
    print(f"ETAPA 2/4  PREJUICIO  ({semillas} semillas x {ciclos} ciclos"
          + (f", disp_nido={disp_nido}" if disp_nido else "") + ")")
    print("  marca informativa = clase de tamaño, predice de verdad quién gana.")
    print("  marca arbitraria  = etiqueta heredable sin efecto causal, pero")
    print("                      correlacionada con el tamaño por linaje.")
    print("=" * 68, flush=True)
    out, t0 = {}, time.time()
    total = 2 * semillas
    hecho = 0
    for nom, mk in (("sin_marcas", False), ("con_marcas", True)):
        R = []
        for s in range(semillas):
            t1 = time.time()
            cfg, reg, pob = prejuicio.corre(mk, ciclos=ciclos, seed=s, disp_nido=disp_nido)
            R.append(reg)
            hecho += 1
            eta = (time.time() - t0) / hecho * (total - hecho)
            extra = ""
            if mk and reg["prej_arb"]:
                u = slice(-60, None)
                pa = np.nanmean(reg["prej_arb"][u])
                ra = np.nanmean(reg["real_arb"][u])
                extra = f"  prej-real={pa-ra:+.4f}"
            print(f"  [{barra(hecho/total)}] {nom} s{s}  n={reg['n'][-1]:3d}"
                  f"{extra}   {reloj(time.time()-t1)}  resta ~{reloj(eta)}",
                  flush=True)
        out[nom] = R
    json.dump(out, open("prejuicio.json", "w"))
    print(f"\n  escrito prejuicio.json   [{reloj(time.time()-t0)}]\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="~2 min, solo verifica")
    ap.add_argument("--semillas", type=int, default=None)
    ap.add_argument("--ciclos", type=int, default=None)
    ap.add_argument("--disp_nido", type=float, default=0.0,
                     help="frac. de crias que se van a un nido al azar (0=nido es linaje puro)")
    ap.add_argument("--saltar", default="", help="etapas a saltar, ej: 1")
    A = ap.parse_args()

    if A.test:
        semillas, ciclos, rapido = 3, 150, True
    else:
        semillas, ciclos, rapido = 12, 1200, False
    if A.semillas:
        semillas = A.semillas
    if A.ciclos:
        ciclos = A.ciclos
    saltar = {int(x) for x in A.saltar.split(",") if x.strip()}

    t0 = time.time()
    print(f"etapa 1 completa · {semillas} semillas × {ciclos} ciclos"
          + ("  [MODO TEST]" if A.test else ""))

    if 1 not in saltar:
        if not etapa_validacion(rapido):
            sys.exit(1)
    if 2 not in saltar:
        etapa_prejuicio(semillas, ciclos, disp_nido=A.disp_nido)

    if 3 not in saltar:
        print("=" * 68)
        print("ETAPA 3/4  ANALISIS")
        print("=" * 68, flush=True)
        import analiza
        analiza.main("prejuicio.json")

    if 4 not in saltar:
        print("\n" + "=" * 68)
        print("ETAPA 4/4  FIGURAS")
        print("=" * 68, flush=True)
        import figuras
        figuras.main("panel1.png")

    print("\n" + "=" * 68)
    print(f"listo en {reloj(time.time()-t0)}")
    print("archivos: validacion.json, prejuicio.json, panel1.png")
    print("=" * 68)
    print("""
QUE MIRAR, en orden:

  1. z de la diferencia arbitraria. Si |z| < 2, no hay prejuicio
     injustificado y el resultado que veniamos persiguiendo no existe.

  2. cuantas semillas sub-usan la marca informativa. Si es la mayoria con
     z negativo, ese si es el resultado: desperdician la senal valida.

  3. convergencia. Si algun rasgo "sigue moviendose", los ciclos no
     alcanzan y las cifras de arriba son provisorias.

  4. correlacion peso del testimonio vs error informativo. Con 12 semillas
     empieza a significar algo; mide el doble conteo por el nido.
""")


if __name__ == "__main__":
    main()
