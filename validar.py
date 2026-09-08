"""
ETAPA 1a -- validacion contra Halcon-Paloma.

Corre el mundo SIN marcas (cfg.marcas = False): las decisiones de escalar
salen solo de p0, el rasgo heredable de propension base. Sin marcas no hay
creencias que aprender, asi que esto aisla la pregunta de fondo: la
seleccion sobre p0, bajo la regla fisica de disputa de mundo.py, converge
a la tasa de escalada que predice Halcon-Paloma (p* = V/C)?

Si NO converge, el problema esta en la mecanica de disputa/reproduccion,
no en marcas ni en creencias -- por eso esto se corre antes que nada.
"""
import numpy as np
from mundo import Cfg, corre_ciclos


def corre(gasto_pelea, ciclos=150, seed=0, var_tam=None):
    cfg = Cfg()
    cfg.marcas = False
    cfg.gasto_pelea = gasto_pelea
    if var_tam is not None:
        cfg.var_tam = var_tam
    reg, pob = corre_ciclos(cfg, seed, ciclos)
    return cfg, reg, pob


if __name__ == "__main__":
    # barrido rapido de costo, para mirar sin pasar por todo.py
    for g in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        cfg, reg, pob = corre(g, ciclos=150, seed=0)
        C = 2 * g
        pred = min(1.0, cfg.V_bocado / C)
        obs = float(np.mean(reg["escalada"][-50:])) if len(reg["escalada"]) > 50 else float("nan")
        print(f"C={C:4.1f}  pred={pred:.3f}  obs={obs:.3f}  n_final={reg['n'][-1] if reg['n'] else 0}")
