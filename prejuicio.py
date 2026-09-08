"""
ETAPA 1b -- prejuicio: corridas con y sin marcas.

Este archivo no estaba escrito. analiza.py y figuras.py ya existian y
fijan el contrato de que campos debe tener reg -- eso es lo que uso como
especificacion. Las definiciones exactas de "prej" y "real" no estaban
fijadas en ningun lado y las decido aca. Quedan documentadas para que se
puedan objetar.

DECISIONES DE MEDICION (no las inventa mundo.py, las pongo yo):

  prej_X, real_X (X = info | arb): "cuanto discrimina la poblacion por
    esta marca" se mide como DISPERSION entre valores, no como un punto:
      real_X  = max_k(medir_verdad[d,k]) - min_k(medir_verdad[d,k])
      prej_X  = max_k(creencia_media[d,k]) - min_k(creencia_media[d,k])
    donde creencia_media[d,k] = promedio en la poblacion de a/(a+b) para
    esa dimension y valor. Si la marca no tuviera ningun poder predictivo
    real, real_X ~ 0; si la poblacion tampoco discrimina, prej_X ~ 0.
    prej-real > 0 en la marca arbitraria = discriminan MAS de lo que esa
    marca predice de verdad -> prejuicio no justificado.
    prej-real < 0 en la marca informativa = discriminan MENOS de lo que
    esa marca SI predice -> desperdician señal valida.

  err_X: error de calibracion, |creencia_media - real| promediado sobre
    los valores de marca que tienen realidad definida (excluye clases con
    <2 individuos, igual que medir_verdad). Es un numero distinto de
    prej-real: se puede discriminar la cantidad correcta pero en el
    sentido equivocado y el error de calibracion no lo vería iguial que
    la dispersion; por eso van por separado.

  conf: promedio poblacional de (alfa+beta-2) sobre las 2x3 celdas de
    creencia. Resta el prior Beta(1,1) para medir evidencia ACUMULADA,
    no conteo bruto. Sube con testimonio, baja con olvido. Es lo que
    analiza.py llama "confianza".

  div_nidos: diversidad efectiva de linajes = 1/sum(p_nido^2) (inversa de
    Simpson), NaN si menos de 2 nidos tienen algun individuo. Se eligio
    esta forma y no "nidos poblados" en bruto porque es independiente del
    tamano poblacional (regla metodologica: diversidad final no debe
    seguir a N). No es un test de invasibilidad mutua -- eso, si hace
    falta, se mide aparte reintroduciendo un linaje extinto.

  sin_marcas: corre con cfg.marcas=False. Ahi decidir() ignora las marcas
    y el bloque de creencias en un_ciclo nunca se ejecuta, asi que a,b
    quedan en el prior (0.5) todo el tiempo y prej_info=prej_arb=0 por
    construccion. ES EL NULO: si con marcas encendidas prej-real no se
    despega de lo que da aca, no hay nada que reportar.
"""
import numpy as np
from mundo import Cfg, corre_ciclos, medir_verdad


def _spread(x):
    x = x[~np.isnan(x)]
    return float(x.max() - x.min()) if len(x) >= 2 else float("nan")


def _err(creencia, real, d):
    m = ~np.isnan(real[d])
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(creencia[d, m] - real[d, m])))


def _medir(pob, reg):
    cfg = pob.cfg
    real = medir_verdad(pob)                      # (2, n_val)
    p = pob.a / (pob.a + pob.b)                    # (n, 2, n_val)
    creencia = p.mean(axis=0)                      # (2, n_val)
    conf = float((pob.a + pob.b - 2).mean())

    reg.setdefault("prej_info", []).append(_spread(creencia[0].copy()))
    reg.setdefault("real_info", []).append(_spread(real[0].copy()))
    reg.setdefault("prej_arb", []).append(_spread(creencia[1].copy()))
    reg.setdefault("real_arb", []).append(_spread(real[1].copy()))
    reg.setdefault("err_info", []).append(_err(creencia, real, 0))
    reg.setdefault("err_arb", []).append(_err(creencia, real, 1))
    reg.setdefault("w_tes", []).append(float(pob.w_tes.mean()))
    reg.setdefault("w_cre", []).append(float(pob.w_cre.mean()))
    reg.setdefault("olvido", []).append(float(pob.olvido.mean()))
    reg.setdefault("conf", []).append(conf)

    conteos = np.bincount(pob.nido, minlength=cfg.n_nidos)
    poblados = int((conteos > 0).sum())
    if poblados < 2:
        div = float("nan")
    else:
        pn = conteos / conteos.sum()
        div = float(1.0 / np.sum(pn ** 2))
    reg.setdefault("div_nidos", []).append(div)


def corre(marcas, ciclos=1200, seed=0):
    cfg = Cfg()
    cfg.marcas = marcas
    reg, pob = corre_ciclos(cfg, seed, ciclos, post_ciclo=_medir)
    return cfg, reg, pob


if __name__ == "__main__":
    for mk in (False, True):
        cfg, reg, pob = corre(mk, ciclos=150, seed=0)
        u = slice(-30, None)
        print(f"marcas={mk}  n_final={reg['n'][-1] if reg['n'] else 0}  "
              f"prej_arb={np.nanmean(reg['prej_arb'][u]):+.4f}  "
              f"real_arb={np.nanmean(reg['real_arb'][u]):+.4f}")
