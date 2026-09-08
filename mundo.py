"""
ETAPA 1 -- forrajeo, disputa, marcas y aprendizaje bayesiano.

Sin senales ni depredadores todavia. El objetivo de esta etapa es doble:
  (a) VALIDAR contra Halcon-Paloma: sin marcas, la tasa de escalada debe
      converger a V/C. Si no lo hace, hay un bug y nos enteramos ahora.
  (b) Con marcas encendidas, medir cuanto del prejuicio esta justificado.

REGLAS (ninguna matriz de pagos; todo sale de una regla fisica)
  Mundo: L parches en anillo, fruta con regeneracion logistica.
  Ciclo: T pasos de forrajeo, vuelta al nido, reproduccion, muerte por edad.
  Disputa: si dos agentes coinciden en un parche, cada uno decide ceder o
    escalar. Nadie ve la decision del otro.
      ceden ambos    -> reparten la fruta, gasto cero
      uno escala     -> el que escala se lleva todo PERO paga el gasto igual
      escalan ambos  -> pelea. Gana con prob. S_i/(S_i+S_j). Ambos pagan
                        energia y arriesgan lesion (baja el tamano efectivo).
    El costo no es un parametro llamado "costo": es energia efectivamente
    perdida. Si no puedo cambiarlo sin cambiar algo fisico, es endogeno.

  Prediccion Halcon-Paloma: p* = V / C, con C = 2 * gasto_pelea.

MARCAS (vector de dos dimensiones, observables)
  marca 0  INFORMATIVA: clase de tamano. Predice de verdad quien gana.
  marca 1  ARBITRARIA: etiqueta heredable sin ningun efecto causal.
  Ojo: como la etiqueta se hereda y los linajes difieren en tamano, la
  etiqueta CORRELACIONA con la capacidad de pelea por estructura de linaje,
  sin relacion causal. El prejuicio sera a veces correcto. Medimos por
  separado lo justificado y lo inventado.

CREENCIAS
  Beta(a,b) sobre "gano si peleo contra alguien con marca de valor k",
  una por dimension y valor. Decision por muestreo de Thompson: la
  exploracion sale de la incertidumbre, no de un parametro aparte.
  Solo PELEAR informa: ceder no dice si habrias ganado. La asimetria es real.

  Testimonio: al volver al nido se comparte lo observado, con peso reducido
  y heredable. Se comparten EVENTOS, no creencias: nadie copia la creencia
  de otro, lo que acota el doble conteo a cuantos testigos hubo.

  Olvido: cada ciclo las creencias decaen hacia el prior. La expectativa se
  conserva y la CONFIANZA cae, asi que sin evidencia nueva el agente vuelve
  a explorar. Es lo que impide que una tradicion falsa se congele.

HERENCIA
  Asexual. Nido = linaje. La cria hereda rasgos con mutacion y ademas las
  creencias del progenitor: lo que asumimos del mundo al crecer es lo que
  nos contaron.
"""
import numpy as np

# ----------------------------------------------------------------- parametros

class Cfg:
    L = 40                 # parches
    T = 40                 # pasos por ciclo
    K_fruta = 6.0          # capacidad por parche
    g_fruta = 0.6          # regeneracion logistica
    V_bocado = 1.0         # fruta extraible por evento

    gasto_escalar = 0.05   # energia por escalar (se paga aunque no haya pelea)
    gasto_pelea = 0.25     # energia adicional si ambos escalan
    p_lesion = 0.15        # prob. de lesion por pelea
    lesion = 0.25          # fraccion de tamano perdida
    recup = 0.05           # recuperacion de lesion por ciclo

    coste_vivir = 0.25     # por ciclo
    e_repro = 6.0          # umbral de reproduccion
    e_ini = 3.0
    edad_max = 6           # ciclos

    N0 = 300
    N_max = 900
    n_nidos = 30

    mut = 0.05             # sd de mutacion de rasgos
    n_val = 3              # valores por marca
    disp_nido = 0.0        # frac. de crias que se van a un nido al azar en
                            # vez de heredar el del progenitor. 0.0 = nido
                            # es linaje puro (default, sin cambio de
                            # comportamiento). >0 es una perilla de
                            # diagnostico: si el colapso a un solo linaje es
                            # deriva por ausencia total de dispersion, un
                            # poco de dispersion deberia bastar para
                            # mantener varios nidos poblados.

    marcas = True          # False -> validacion Halcon-Paloma
    ciclos = 400

    @property
    def C(self):
        """Costo en el sentido de Halcon-Paloma: ambos escalan ->
        pago esperado V/2 - gasto_pelea = (V - C)/2, luego C = 2*gasto.
        OJO: esto NO es la prediccion de equilibrio completa. El escalador
        tambien paga gasto_escalar cuando el rival cede (mundo.py linea
        ~219), asi que la condicion de indiferencia real da
        p* = (V - 2*gasto_escalar) / C, no V/C. Ver validar.py / todo.py."""
        return 2 * self.gasto_pelea


RASGOS = ["p0", "w_creencia", "w_testimonio", "olvido", "tamano"]


# ----------------------------------------------------------------- poblacion

class Pob:
    def __init__(self, cfg, rng):
        n = cfg.N0
        self.cfg, self.rng = cfg, rng
        self.p0 = rng.uniform(0.1, 0.9, n)
        self.w_cre = rng.uniform(0.0, 1.0, n)
        self.w_tes = rng.uniform(0.0, 0.5, n)
        self.olvido = rng.uniform(0.85, 0.99, n)
        self.tam = rng.uniform(0.7, 1.3, n)
        self.lesionado = np.zeros(n)
        self.tag = rng.integers(0, cfg.n_val, n)          # marca arbitraria
        self.tipo = np.zeros(n, int)                      # arquetipo fundador (tipos.py)
        self.nido = rng.integers(0, cfg.n_nidos, n)
        self.e = np.full(n, cfg.e_ini)
        self.edad = np.zeros(n, int)
        self.pos = rng.integers(0, cfg.L, n)
        # creencias: [ind, dimension, valor] -> (alfa, beta)
        self.a = np.ones((n, 2, cfg.n_val))
        self.b = np.ones((n, 2, cfg.n_val))

    def n(self):
        return len(self.e)

    def tam_ef(self):
        return self.tam * (1 - self.lesionado)

    def clase_tam(self):
        """Marca informativa: clase de tamano observable."""
        q = np.quantile(self.tam_ef(), [1 / 3, 2 / 3])
        return np.digitize(self.tam_ef(), q)


# ----------------------------------------------------------------- ciclo

def marcas_de(pob, idx):
    return np.stack([pob.clase_tam()[idx], pob.tag[idx]], axis=1)


def decidir(pob, yo, marca_op, rng):
    """Muestreo de Thompson sobre las marcas del oponente."""
    cfg = pob.cfg
    if not cfg.marcas:
        return rng.random(len(yo)) < np.clip(pob.p0[yo], 0, 1)
    q = np.zeros(len(yo))
    for d in range(2):
        k = marca_op[:, d]
        a = pob.a[yo, d, k]
        b = pob.b[yo, d, k]
        q += rng.beta(a, b)
    q /= 2.0
    p = np.clip(pob.p0[yo] + pob.w_cre[yo] * (q - 0.5), 0.0, 1.0)
    return rng.random(len(yo)) < p


def un_ciclo(pob, fruta, rng, reg):
    cfg = pob.cfg
    n = pob.n()
    # observaciones del ciclo: (ind, dim, valor, gano)
    obs_a = np.zeros((n, 2, cfg.n_val))
    obs_b = np.zeros((n, 2, cfg.n_val))
    # observaciones del nido (testimonio): eventos, no creencias
    nid_a = np.zeros((cfg.n_nidos, 2, cfg.n_val))
    nid_b = np.zeros((cfg.n_nidos, 2, cfg.n_val))

    n_esc = n_enc = n_pelea = 0

    for t in range(cfg.T):
        # movimiento simple: paso aleatorio (el forrajeo dirigido es etapa 2)
        mv = rng.random(n) < 0.5
        pob.pos[mv] = (pob.pos[mv] + rng.choice([-1, 1], mv.sum())) % cfg.L

        orden = rng.permutation(n)
        pos_o = pob.pos[orden]
        idx_por_parche = {}
        for i, p in zip(orden, pos_o):
            idx_por_parche.setdefault(int(p), []).append(i)

        for p, grupo in idx_por_parche.items():
            if fruta[p] < cfg.V_bocado:
                continue
            rng.shuffle(grupo)
            # los pares se disputan; un impar cosecha sin disputa
            for j in range(0, len(grupo) - 1, 2):
                i1, i2 = grupo[j], grupo[j + 1]
                if fruta[p] < cfg.V_bocado:
                    break
                n_enc += 1
                m1 = marcas_de(pob, np.array([i2]))
                m2 = marcas_de(pob, np.array([i1]))
                e1 = decidir(pob, np.array([i1]), m1, rng)[0]
                e2 = decidir(pob, np.array([i2]), m2, rng)[0]
                n_esc += int(e1) + int(e2)
                V = cfg.V_bocado
                fruta[p] -= V

                if e1 and e2:
                    n_pelea += 1
                    s1, s2 = pob.tam_ef()[i1], pob.tam_ef()[i2]
                    gana1 = rng.random() < s1 / (s1 + s2)
                    pob.e[i1] += V if gana1 else 0.0
                    pob.e[i2] += 0.0 if gana1 else V
                    pob.e[i1] -= cfg.gasto_escalar + cfg.gasto_pelea
                    pob.e[i2] -= cfg.gasto_escalar + cfg.gasto_pelea
                    for who in (i1, i2):
                        if rng.random() < cfg.p_lesion:
                            pob.lesionado[who] = min(0.6, pob.lesionado[who] + cfg.lesion)
                    # SOLO pelear informa. Ceder no dice si habrias ganado.
                    for yo, op, gano in ((i1, i2, gana1), (i2, i1, not gana1)):
                        mk = [pob.clase_tam()[op], pob.tag[op]]
                        for d in range(2):
                            if gano:
                                obs_a[yo, d, mk[d]] += 1
                                nid_a[pob.nido[yo], d, mk[d]] += 1
                            else:
                                obs_b[yo, d, mk[d]] += 1
                                nid_b[pob.nido[yo], d, mk[d]] += 1
                elif e1 or e2:
                    gan, ced = (i1, i2) if e1 else (i2, i1)
                    pob.e[gan] += V - cfg.gasto_escalar
                else:
                    pob.e[i1] += V / 2
                    pob.e[i2] += V / 2
            if len(grupo) % 2 == 1 and fruta[p] >= cfg.V_bocado:
                fruta[p] -= cfg.V_bocado
                pob.e[grupo[-1]] += cfg.V_bocado

        fruta += cfg.g_fruta * fruta * (1 - fruta / cfg.K_fruta)

    # ---- vuelta al nido: testimonio con peso reducido -------------------
    if cfg.marcas:
        for i in range(n):
            nd = pob.nido[i]
            wt = pob.w_tes[i]
            pob.a[i] += obs_a[i] + wt * (nid_a[nd] - obs_a[i])
            pob.b[i] += obs_b[i] + wt * (nid_b[nd] - obs_b[i])
        # olvido: la expectativa se conserva, la confianza cae
        d = pob.olvido[:, None, None]
        pob.a = 1 + (pob.a - 1) * d
        pob.b = 1 + (pob.b - 1) * d

    reg["escalada"].append(n_esc / max(1, 2 * n_enc))
    reg["peleas"].append(n_pelea / max(1, n_enc))
    return fruta


def reproducir(pob, rng, reg=None):
    """reg, si se pasa, recibe diagnosticos de POR QUE cambia N: cuantos
    mueren por edad vs por energia (excluyentes: edad manda), cuantos nacen
    realmente vs cuantos podrian haber nacido, y si el tope N_max bloqueo
    nacimientos ese ciclo. El tope, cuando actua, frena TODO nacimiento por
    igual -- no es una poda dirigida a ninguna minoria; la seleccion sobre
    quien llega a reproducirse ocurre antes, via el umbral de energia."""
    cfg = pob.cfg
    pob.e -= cfg.coste_vivir
    pob.edad += 1
    pob.lesionado = np.maximum(0.0, pob.lesionado - cfg.recup)

    vive = (pob.e > 0) & (pob.edad <= cfg.edad_max)
    if reg is not None:
        murio_edad = (~vive) & (pob.edad > cfg.edad_max)
        murio_energia = (~vive) & (pob.edad <= cfg.edad_max)
        reg.setdefault("murio_edad", []).append(int(murio_edad.sum()))
        reg.setdefault("murio_energia", []).append(int(murio_energia.sum()))

    hijos_pot = np.flatnonzero(vive & (pob.e > cfg.e_repro))
    if len(hijos_pot) and vive.sum() < cfg.N_max:
        cupo = max(0, cfg.N_max - int(vive.sum()))
        hijos = hijos_pot[:cupo]
        tope = len(hijos_pot) > cupo
    else:
        hijos = np.array([], int)
        tope = len(hijos_pot) > 0

    if reg is not None:
        reg.setdefault("n_nacidos", []).append(int(len(hijos)))
        reg.setdefault("n_nacidos_potenciales", []).append(int(len(hijos_pot)))
        reg.setdefault("tope_nmax", []).append(bool(tope))

    def sel(arr):
        return np.concatenate([arr[vive], arr[hijos]])

    if len(hijos):
        pob.e[hijos] /= 2.0
    nh = len(hijos)

    nuevo = {}
    for r, arr in (("p0", pob.p0), ("w_cre", pob.w_cre), ("w_tes", pob.w_tes),
                   ("olvido", pob.olvido), ("tam", pob.tam)):
        h = arr[hijos] + rng.normal(0, cfg.mut, nh) if nh else np.array([])
        nuevo[r] = np.concatenate([arr[vive], h])
    pob.p0 = np.clip(nuevo["p0"], 0, 1)
    pob.w_cre = np.clip(nuevo["w_cre"], 0, 3)
    pob.w_tes = np.clip(nuevo["w_tes"], 0, 1)
    pob.olvido = np.clip(nuevo["olvido"], 0.5, 0.999)
    pob.tam = np.clip(nuevo["tam"], 0.3, 3.0)

    # etiqueta arbitraria: se hereda, muta poco, NO afecta nada
    tag_h = pob.tag[hijos].copy() if nh else np.array([], int)
    if nh:
        m = rng.random(nh) < 0.02
        tag_h[m] = rng.integers(0, cfg.n_val, m.sum())
    pob.tag = np.concatenate([pob.tag[vive], tag_h]).astype(int)
    # tipo (arquetipo fundador): hereda sin mutar, para poder rastrear que
    # fraccion de la poblacion desciende de cada combinacion inicial.
    tipo_h = pob.tipo[hijos].copy() if nh else np.array([], int)
    pob.tipo = np.concatenate([pob.tipo[vive], tipo_h]).astype(int)
    # dispersion: por default la cria hereda el nido del progenitor sin
    # excepcion (nido=linaje). disp_nido>0 reasigna una fraccion al azar.
    nido_h = pob.nido[hijos].copy() if nh else np.array([], int)
    if nh and cfg.disp_nido > 0:
        disp = rng.random(nh) < cfg.disp_nido
        nido_h[disp] = rng.integers(0, cfg.n_nidos, int(disp.sum()))
    pob.nido = np.concatenate([pob.nido[vive], nido_h]).astype(int)
    pob.e = sel(pob.e)
    pob.edad = np.concatenate([pob.edad[vive], np.zeros(nh, int)])
    pob.lesionado = np.concatenate([pob.lesionado[vive], np.zeros(nh)])
    pob.pos = sel(pob.pos).astype(int)
    # la cria hereda las CREENCIAS del progenitor
    pob.a = np.concatenate([pob.a[vive], pob.a[hijos]])
    pob.b = np.concatenate([pob.b[vive], pob.b[hijos]])


def medir_verdad(pob):
    """Tasa de victoria REALIZADA por valor de cada marca, en la poblacion.

    Para la marca arbitraria la verdad NO es 0.5: como la etiqueta se hereda
    y los linajes difieren en tamano, aparece correlacion espuria con la
    capacidad de pelea sin ninguna relacion causal. Eso es lo que hace que el
    prejuicio sea a veces correcto. Comparar la creencia contra ESTO, y no
    contra 0.5, es lo que separa lo justificado de lo inventado.
    """
    cfg = pob.cfg
    cl, tg, s = pob.clase_tam(), pob.tag, pob.tam_ef()
    out = np.full((2, cfg.n_val), np.nan)
    for d, mk in enumerate((cl, tg)):
        for k in range(cfg.n_val):
            sel = mk == k
            if sel.sum() < 2 or (~sel).sum() < 2:
                continue
            out[d, k] = float(np.mean(s[:, None] / (s[:, None] + s[sel][None, :])))
    return out
