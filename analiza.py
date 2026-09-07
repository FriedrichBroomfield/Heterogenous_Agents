"""Analiza prejuicio.json POR SEMILLA, no promediando.

Motivo: las semillas terminan en regimenes distintos. Promediarlas describe
una poblacion que no existe.
"""
import json, sys
import numpy as np


def main(path="prejuicio.json"):
    R = json.load(open(path))
    S = R["con_marcas"]
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
    print(f"  diferencia (prej-real): media {dif.mean():+.4f} +- {dif.std()/np.sqrt(len(dif)):.4f}")
    print("  [la razon tiene denominador chico y es inestable; la diferencia no]")
    z = dif.mean() / (dif.std() / np.sqrt(len(dif)) + 1e-12)
    print(f"  z de la diferencia = {z:+.2f}  -> "
          f"{'distinta de cero' if abs(z) > 2 else 'NO distinta de cero'}")

    print("\nMARCA INFORMATIVA (discriminan - real; bajo cero = sub-usan)")
    d_inf = np.array([m(r["prej_info"]) - m(r["real_info"]) for r in S])
    for i, d in enumerate(d_inf):
        print(f"  semilla {i}: {d:+.4f}")
    z = d_inf.mean() / (d_inf.std() / np.sqrt(len(d_inf)) + 1e-12)
    print(f"  media {d_inf.mean():+.4f}  z = {z:+.2f}  "
          f"({int((d_inf < 0).sum())}/{len(d_inf)} semillas sub-usan)")

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
        cv = v.std() / abs(v.mean()) if v.mean() else np.nan
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

    print("\nESTRUCTURA DE NIDOS")
    for i, r in enumerate(S):
        d = np.array(r["div_nidos"], float)
        nn = int(np.isnan(d).sum())
        print(f"  semilla {i}: div={np.nanmean(d[u]):.4f}  "
              f"ciclos con <2 nidos poblados: {nn}/{len(d)}"
              + ("  <-- linaje unico por tramos" if nn > 10 else ""))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "prejuicio.json")
