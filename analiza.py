"""Analiza prejuicio.json POR SEMILLA, no promediando.

Motivo: las semillas terminan en regimenes distintos. Promediarlas describe
una poblacion que no existe.
"""
import json, sys
import numpy as np


def main(path="prejuicio.json"):
    R = json.load(open(path))
    S = R["con_marcas"]
    S0 = R.get("sin_marcas", [])
    u = slice(-60, None)

    def m(x):
        a = np.array(x, float)
        return float(np.nanmean(a[u])) if len(a) else float("nan")

    print("POR SEMILLA (ultimos 60 ciclos)\n")
    print(f"{'':>4} {'n':>5} {'w_tes':>7} {'w_cre':>7} {'olvido':>7} {'conf':>7} "
          f"{'err_inf':>8} {'err_arb':>8} {'prej_arb':>9} {'real_arb':>9} "
          f"{'razon':>6} {'dif_arb':>8}")
    raz, dif, wt, ei = [], [], [], []
    for i, r in enumerate(S):
        ra, pa = m(r["real_arb"]), m(r["prej_arb"])
        raz.append(pa / ra if ra > 1e-6 else np.nan)
        dif.append(pa - ra)
        wt.append(m(r["w_tes"])); ei.append(m(r["err_info"]))
        print(f"{i:>4} {r['n'][-1]:>5} {wt[-1]:>7.3f} {m(r['w_cre']):>7.3f} "
              f"{m(r['olvido']):>7.3f} {m(r['conf']):>7.1f} {ei[-1]:>8.4f} "
              f"{m(r['err_arb']):>8.4f} {pa:>9.4f} {ra:>9.4f} "
              f"{raz[-1]:>6.2f} {dif[-1]:>+8.4f}")

    raz, dif = np.array(raz, float), np.array(dif, float)
    print(f"\nSOBRE-DISCRIMINACION ARBITRARIA")
    print(f"  razon (prej/real):  media {np.nanmean(raz):.2f}  "
          f"mediana {np.nanmedian(raz):.2f}  rango {np.nanmin(raz):.2f}-{np.nanmax(raz):.2f}")
    print(f"  semillas con razon > 1: {int(np.nansum(raz > 1))}/{len(raz)}")
    # ddof=1: dif.std() con default (ddof=0) subestima el error estandar en
    # un factor exacto sqrt(n/(n-1)), que infla z. Con n=3 eso es 1.22x --
    # justo la diferencia entre "justo debajo del umbral" y no.
    print(f"  diferencia (prej-real): media {dif.mean():+.4f} +- {dif.std(ddof=1)/np.sqrt(len(dif)):.4f}")
    print("  [la razon tiene denominador chico y es inestable; la diferencia no]")
    z = dif.mean() / (dif.std(ddof=1) / np.sqrt(len(dif)) + 1e-12)
    print(f"  z de la diferencia = {z:+.2f}  -> "
          f"{'distinta de cero' if abs(z) > 2 else 'NO distinta de cero'}")

    print("\nMARCA INFORMATIVA (discriminan - real; bajo cero = sub-usan)")
    d_inf = np.array([m(r["prej_info"]) - m(r["real_info"]) for r in S])
    for i, d in enumerate(d_inf):
        print(f"  semilla {i}: {d:+.4f}")
    z = d_inf.mean() / (d_inf.std(ddof=1) / np.sqrt(len(d_inf)) + 1e-12)
    print(f"  media {d_inf.mean():+.4f}  z = {z:+.2f}  "
          f"({int((d_inf < 0).sum())}/{len(d_inf)} semillas sub-usan)")

    print("\nDIFERENCIAL DE SELECCION (dispersion de tasa de reproduccion REALIZADA por clase)")
    print("  esto no es creencia ni verdad de la pelea: es quien efectivamente")
    print("  se reprodujo mas, por clase de marca, en cada ciclo.")
    d_ri = np.array([m(r["repro_info"]) for r in S])
    d_ra = np.array([m(r["repro_arb"]) for r in S])
    print(f"  con marcas   informativa: media {np.nanmean(d_ri):.4f}  arbitraria: media {np.nanmean(d_ra):.4f}")
    if S0:
        d_ri0 = np.array([m(r["repro_info"]) for r in S0])
        d_ra0 = np.array([m(r["repro_arb"]) for r in S0])
        print(f"  sin marcas   informativa: media {np.nanmean(d_ri0):.4f}  arbitraria: media {np.nanmean(d_ra0):.4f}")
        print("  [sin marcas = nulo: la disparidad que sobreviva sin comportamiento basado")
        print("   en marcas es puramente estructural (tamano real, correlacion de linaje);")
        print("   lo que con_marcas agrega por encima de esto es lo que el comportamiento causa]")
    else:
        print("  [no hay corrida sin_marcas en este json para comparar contra el nulo]")

    print("\nDOBLE CONTEO: peso del testimonio vs error sobre marca informativa")
    wt, ei = np.array(wt), np.array(ei)
    if len(wt) > 2 and wt.std() > 1e-9:
        r_p = float(np.corrcoef(wt, ei)[0, 1])
        print(f"  correlacion = {r_p:+.2f} sobre {len(wt)} semillas  -> "
              f"{'mas testimonio, peor calibracion' if r_p > 0.5 else 'sin senal clara'}")
        print("  [con menos de ~10 semillas esta correlacion no significa nada]")

    print("\nDIVERGENCIA ENTRE SEMILLAS (coef. de variacion)")
    for k in ("w_tes", "w_cre", "olvido", "conf", "err_info"):
        v = np.array([m(r[k]) for r in S])
        cv = v.std(ddof=1) / abs(v.mean()) if v.mean() else np.nan
        flag = "  <-- ALTA: promediar semillas no describe a ninguna" if cv > .25 else ""
        print(f"  {k:9s} media={v.mean():7.3f}  cv={cv:.2f}{flag}")

    print("\nCONVERGENCIA (primer tercio vs ultimo tercio)")
    algo = False
    for k in ("w_tes", "w_cre", "olvido"):
        mov = 0
        for r in S:
            a = np.array(r[k], float); t = len(a) // 3
            if abs(a[-t:].mean() - a[:t].mean()) > 0.08:
                mov += 1
        if mov:
            algo = True
            print(f"  {k:9s} sigue moviendose en {mov}/{len(S)} semillas")
    if not algo:
        print("  todos los rasgos estables -> hay optimo interior")

    print("\nESTABILIDAD DEL SESGO (prej-real, primer tercio vs ultimo tercio)")
    print("  si el signo cambia entre tercios, el z de arriba describe un punto")
    print("  de paso, no una conclusion -- no reportarlo como establecido.")
    for nom, pk, rk in (("informativa", "prej_info", "real_info"),
                         ("arbitraria", "prej_arb", "real_arb")):
        prim, ult = [], []
        for r in S:
            d = np.array(r[pk], float) - np.array(r[rk], float)
            t = len(d) // 3
            if t == 0:
                continue
            prim.append(np.nanmean(d[:t])); ult.append(np.nanmean(d[-t:]))
        if not prim:
            continue
        mp, mu = float(np.nanmean(prim)), float(np.nanmean(ult))
        invertido = np.sign(mp) != np.sign(mu) and mp != 0 and mu != 0
        print(f"  {nom:11s} primer tercio {mp:+.4f}  ultimo tercio {mu:+.4f}"
              + ("  <-- CAMBIO DE SIGNO" if invertido else ""))

    print("\nESTRUCTURA DE NIDOS")
    for i, r in enumerate(S):
        d = np.array(r["div_nidos"], float)
        nn = int(np.isnan(d).sum())
        print(f"  semilla {i}: div={np.nanmean(d[u]):.4f}  "
              f"ciclos con <2 nidos poblados: {nn}/{len(d)}"
              + ("  <-- linaje unico por tramos" if nn > 10 else ""))

    print("\nDISPERSION DENTRO DE LA POBLACION (misma media, poblacion homogenea o partida?)")
    print("  sd de cada rasgo ENTRE INDIVIDUOS de la misma corrida, ultimos 60 ciclos.")
    print("  sd baja y estable = poblacion convergida a un valor. sd que no baja = el")
    print("  rasgo sigue partido entre distintos tipos de agente, aunque la media sea fija.")
    for nom, grupo in (("sin_marcas", S0), ("con_marcas", S)):
        if not grupo:
            continue
        print(f"  {nom}:")
        for i, r in enumerate(grupo):
            faltan = [k for k in ("sd_w_cre", "sd_w_tes", "sd_olvido", "sd_tam") if k not in r or not r[k]]
            if faltan:
                print(f"    semilla {i}: (sin {','.join(faltan)}, json viejo -- correr de nuevo)")
                continue
            print(f"    semilla {i}: sd_w_cre={m(r['sd_w_cre']):.3f}  sd_w_tes={m(r['sd_w_tes']):.3f}  "
                  f"sd_olvido={m(r['sd_olvido']):.3f}  sd_tam={m(r['sd_tam']):.3f}")

    print("\nDIVERGENCIA DE p0 DENTRO DE LA CORRIDA (halcon puro vs paloma pura vs mixta)")
    print("  el equilibrio mixto de Halcon-Paloma admite dos soluciones con la misma")
    print("  media: todos jugando la estrategia mixta p*, o la poblacion partida en")
    print("  halcones y palomas puros que promedian a p*. Esto mide cual de las dos.")
    print("  *** DESCRIPTIVO: falta el nulo (ver PROMPT_TRASPASO / nota del revisor).")
    print("      sd_p0 alto NO prueba polimorfismo mantenido por seleccion -- podria")
    print("      ser deriva+mutacion sin nada que lo sostenga. El test que corresponde")
    print("      (regla 6) es invasibilidad mutua, no esto. ***")
    for nom, grupo in (("sin_marcas", S0), ("con_marcas", S)):
        if not grupo:
            continue
        print(f"  {nom}:")
        for i, r in enumerate(grupo):
            if "sd_p0" not in r or not r["sd_p0"]:
                print(f"    semilla {i}: (sin datos, json viejo -- correr de nuevo)")
                continue
            sd = m(r["sd_p0"]); fh = m(r["frac_halcon"]); fp = m(r["frac_paloma"])
            bimodal = (fh + fp) > 0.3
            print(f"    semilla {i}: p0_media={m(r['p0']):.3f}  sd_p0={sd:.3f}  "
                  f"halcon(p0>.8)={fh:.2f}  paloma(p0<.2)={fp:.2f}"
                  + ("  <-- bimodal: no es una unica estrategia mixta" if bimodal else ""))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "prejuicio.json")
