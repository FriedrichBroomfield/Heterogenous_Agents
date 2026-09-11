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

SENAL DE AMENAZA (opcional, cfg.senales -- Opcion A)
  Separa la INTENCION interna de la SENAL externa: cada agente resuelve
  primero (en base a p0 solamente, antes de ver nada del rival) si esta
  resuelto a escalar, y RECIEN despues decide que mostrar -- con
  p_senal_si_escala / p_senal_si_cede, dos probabilidades heredables e
  independientes entre si. Si ambas convergen al mismo valor la senal es
  puro ruido; si se separan, es honesta o enganosa segun para que lado.
  Nada obliga a que la senal diga la verdad: eso tiene que salir de la
  seleccion, no esta impuesto.

  La decision FINAL (la que de verdad ocurre) es distinta de la intencion:
  ver la senal del rival actualiza una creencia Beta sobre "si el rival
  muestra amenaza, ¿de verdad escala?" (misma maquinaria de testimonio y
  olvido que las marcas) y esa creencia empuja la decision en la direccion
  contraria a las marcas -- creer que el rival va a escalar hace MENOS
  rentable escalar tambien (dos que escalan pagan el costo completo de la
  pelea), no mas. El peso de esa creencia es w_senal, un rasgo heredable
  separado de w_creencia.

  La honestidad no se mide por lo que los agentes CREEN sino por la
  correlacion real entre mostrar la senal y escalar de verdad:
  P(escalo | mostro) vs P(escalo | no mostro), contado directamente sobre
  las acciones tomadas.
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

    depredadores = False   # False -> etapa 1, sin riesgo de depredacion
    frac_peligro = 0.15    # fraccion de parches peligrosos, sorteada por CICLO
                            # (no por paso: el peligro es estable dentro del
                            # ciclo, para que una señal de alerta mas adelante
                            # tenga algo persistente de que avisar)
    p_ataque = 0.07         # prob. de ataque por estar en un parche peligroso
                            # en un paso dado (no hace falta disputar fruta).
                            # Calibrado para que ataques/agente/ciclo ~ 0.4
                            # (T*frac_peligro*p_ataque) y muerte por
                            # depredacion ~6%/ciclo -- riesgo real, no
                            # dominante frente a edad_max=6 (~17%/ciclo en
                            # regimen estacionario). Con 0.35 (primer intento)
                            # colapsaba la poblacion de 300 a 90 en 4 ciclos.
    dano_ataque = 0.8       # energia perdida si atacan -- fisico, igual que
                            # el costo de pelea, no un parametro simbolico
    p_muerte_ataque = 0.15  # prob. de que el ataque sea fatal, dado que hubo
                            # ataque

    senales = False         # False -> sin canal de senal de amenaza (etapas
                            # previas). La intencion interna sigue existiendo
                            # siempre (es solo p0); lo que este flag agrega es
                            # el canal de SENAL y la creencia del receptor
                            # sobre ella -- ver docstring del modulo.

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
        # senal de amenaza (cfg.senales): intencion y senal son cosas
        # distintas -- ver docstring del modulo. Solo consume aleatoriedad
        # si el canal esta activo, para que una corrida con senales=False
        # sea bit a bit identica a antes de que este canal existiera.
        if cfg.senales:
            self.p_senal_si_escala = rng.uniform(0.0, 1.0, n)
            self.p_senal_si_cede = rng.uniform(0.0, 1.0, n)
            self.w_senal = rng.uniform(0.0, 1.0, n)
        else:
            self.p_senal_si_escala = np.full(n, 0.5)
            self.p_senal_si_cede = np.full(n, 0.5)
            self.w_senal = np.zeros(n)
        # creencia del receptor: [ind, senal_vista(0/1)] -> (alfa, beta) de
        # "el rival escalo, dado que mostro esta senal"
        self.a_senal = np.ones((n, 2))
        self.b_senal = np.ones((n, 2))

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


def decidir(pob, yo, marca_op, senal_op, rng):
    """Muestreo de Thompson. Las marcas leen si YO ganaria (sube el impulso
    a escalar); la senal del rival lee si EL va a escalar (baja el impulso
    -- pelear contra quien ya esta resuelto a pelear cuesta, no rinde). Son
    evidencias distintas, con pesos heredables distintos, y se combinan
    sumando/restando sobre el mismo p0 base."""
    cfg = pob.cfg
    p = pob.p0[yo]
    if cfg.marcas:
        q = np.zeros(len(yo))
        for d in range(2):
            k = marca_op[:, d]
            a = pob.a[yo, d, k]
            b = pob.b[yo, d, k]
            q += rng.beta(a, b)
        q /= 2.0
        p = p + pob.w_cre[yo] * (q - 0.5)
    if cfg.senales:
        a_s = pob.a_senal[yo, senal_op]
        b_s = pob.b_senal[yo, senal_op]
        q_s = rng.beta(a_s, b_s)
        p = p - pob.w_senal[yo] * (q_s - 0.5)
    return rng.random(len(yo)) < np.clip(p, 0.0, 1.0)


def un_ciclo(pob, fruta, rng, reg):
    cfg = pob.cfg
    n = pob.n()
    # observaciones del ciclo: (ind, dim, valor, gano)
    obs_a = np.zeros((n, 2, cfg.n_val))
    obs_b = np.zeros((n, 2, cfg.n_val))
    # observaciones del nido (testimonio): eventos, no creencias
    nid_a = np.zeros((cfg.n_nidos, 2, cfg.n_val))
    nid_b = np.zeros((cfg.n_nidos, 2, cfg.n_val))
    # observaciones de la senal del rival: [ind, senal_vista(0/1)] -> conteo
    # de "el rival escalo" / "el rival no escalo"
    obs_a_senal = np.zeros((n, 2))
    obs_b_senal = np.zeros((n, 2))
    nid_a_senal = np.zeros((cfg.n_nidos, 2))
    nid_b_senal = np.zeros((cfg.n_nidos, 2))

    n_esc = n_enc = n_pelea = 0
    n_ataques = n_muertes_depred = 0
    n_mostro = n_mostro_escalo = n_no_mostro = n_no_mostro_escalo = 0
    peligroso = rng.random(cfg.L) < cfg.frac_peligro if cfg.depredadores else None

    for t in range(cfg.T):
        # movimiento simple: paso aleatorio (el forrajeo dirigido es etapa 2)
        mv = rng.random(n) < 0.5
        pob.pos[mv] = (pob.pos[mv] + rng.choice([-1, 1], mv.sum())) % cfg.L

        orden = rng.permutation(n)
        pos_o = pob.pos[orden]
        idx_por_parche = {}
        for i, p in zip(orden, pos_o):
            idx_por_parche.setdefault(int(p), []).append(i)

        # riesgo de depredacion: por ESTAR en un parche peligroso este paso,
        # no por disputar fruta -- independiente del forrajeo de abajo. La
        # muerte se resuelve recien en reproducir() (e muy negativa), igual
        # que la muerte por energia de una pelea perdida: es consistente con
        # como el resto del modelo ya difiere toda muerte al fin del ciclo.
        if cfg.depredadores:
            for p, grupo in idx_por_parche.items():
                if not peligroso[p]:
                    continue
                for i in grupo:
                    if rng.random() < cfg.p_ataque:
                        n_ataques += 1
                        pob.e[i] -= cfg.dano_ataque
                        if rng.random() < cfg.p_muerte_ataque:
                            n_muertes_depred += 1
                            pob.e[i] = -999.0

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
                if cfg.senales:
                    # intencion: resuelta ANTES de ver nada del rival, solo
                    # con el impulso base p0 -- lo que se muestra despues es
                    # una decision aparte.
                    intencion1 = rng.random() < pob.p0[i1]
                    intencion2 = rng.random() < pob.p0[i2]
                    senal1 = rng.random() < (pob.p_senal_si_escala[i1] if intencion1
                                              else pob.p_senal_si_cede[i1])
                    senal2 = rng.random() < (pob.p_senal_si_escala[i2] if intencion2
                                              else pob.p_senal_si_cede[i2])
                    s1_ve = np.array([int(senal2)])   # lo que i1 ve del rival
                    s2_ve = np.array([int(senal1)])   # lo que i2 ve del rival
                else:
                    s1_ve = s2_ve = None
                e1 = decidir(pob, np.array([i1]), m1, s1_ve, rng)[0]
                e2 = decidir(pob, np.array([i2]), m2, s2_ve, rng)[0]
                n_esc += int(e1) + int(e2)

                if cfg.senales:
                    # honestidad: correlacion REAL entre mostrar y escalar,
                    # no lo que nadie cree.
                    for mostro, escalo in ((senal1, e1), (senal2, e2)):
                        if mostro:
                            n_mostro += 1
                            n_mostro_escalo += int(escalo)
                        else:
                            n_no_mostro += 1
                            n_no_mostro_escalo += int(escalo)
                    # creencia del receptor sobre la senal del rival: se
                    # observa la ACCION del rival (e2 para i1, e1 para i2),
                    # visible este mismo encuentro sin importar que decidi
                    # hacer yo -- a diferencia de "quien gano", que solo se
                    # sabe si de verdad se peleo.
                    k1, k2 = int(senal2), int(senal1)
                    if e2:
                        obs_a_senal[i1, k1] += 1
                    else:
                        obs_b_senal[i1, k1] += 1
                    if e1:
                        obs_a_senal[i2, k2] += 1
                    else:
                        obs_b_senal[i2, k2] += 1
                    nd1, nd2 = pob.nido[i1], pob.nido[i2]
                    if e2:
                        nid_a_senal[nd1, k1] += 1
                    else:
                        nid_b_senal[nd1, k1] += 1
                    if e1:
                        nid_a_senal[nd2, k2] += 1
                    else:
                        nid_b_senal[nd2, k2] += 1
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

    if cfg.senales:
        for i in range(n):
            nd = pob.nido[i]
            wt = pob.w_tes[i]
            pob.a_senal[i] += obs_a_senal[i] + wt * (nid_a_senal[nd] - obs_a_senal[i])
            pob.b_senal[i] += obs_b_senal[i] + wt * (nid_b_senal[nd] - obs_b_senal[i])
        d = pob.olvido[:, None]
        pob.a_senal = 1 + (pob.a_senal - 1) * d
        pob.b_senal = 1 + (pob.b_senal - 1) * d

    reg["escalada"].append(n_esc / max(1, 2 * n_enc))
    reg["peleas"].append(n_pelea / max(1, n_enc))
    if cfg.depredadores:
        reg.setdefault("ataques", []).append(n_ataques)
        reg.setdefault("muertes_depred", []).append(n_muertes_depred)
    if cfg.senales:
        reg.setdefault("mostro", []).append(n_mostro)
        reg.setdefault("mostro_escalo", []).append(n_mostro_escalo)
        reg.setdefault("no_mostro", []).append(n_no_mostro)
        reg.setdefault("no_mostro_escalo", []).append(n_no_mostro_escalo)
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

    # los rasgos de senal solo mutan si cfg.senales esta activo -- si no,
    # se transportan sin tocar (sel(), sin consumir aleatoriedad) para que
    # una corrida con senales=False sea bit a bit identica a antes de que
    # este canal existiera.
    rasgos_mut = [("p0", pob.p0), ("w_cre", pob.w_cre), ("w_tes", pob.w_tes),
                  ("olvido", pob.olvido), ("tam", pob.tam)]
    if cfg.senales:
        rasgos_mut += [("p_senal_si_escala", pob.p_senal_si_escala),
                       ("p_senal_si_cede", pob.p_senal_si_cede),
                       ("w_senal", pob.w_senal)]
    nuevo = {}
    for r, arr in rasgos_mut:
        h = arr[hijos] + rng.normal(0, cfg.mut, nh) if nh else np.array([])
        nuevo[r] = np.concatenate([arr[vive], h])
    pob.p0 = np.clip(nuevo["p0"], 0, 1)
    pob.w_cre = np.clip(nuevo["w_cre"], 0, 3)
    pob.w_tes = np.clip(nuevo["w_tes"], 0, 1)
    pob.olvido = np.clip(nuevo["olvido"], 0.5, 0.999)
    pob.tam = np.clip(nuevo["tam"], 0.3, 3.0)
    if cfg.senales:
        pob.p_senal_si_escala = np.clip(nuevo["p_senal_si_escala"], 0, 1)
        pob.p_senal_si_cede = np.clip(nuevo["p_senal_si_cede"], 0, 1)
        pob.w_senal = np.clip(nuevo["w_senal"], 0, 3)
    else:
        pob.p_senal_si_escala = sel(pob.p_senal_si_escala)
        pob.p_senal_si_cede = sel(pob.p_senal_si_cede)
        pob.w_senal = sel(pob.w_senal)
    pob.a_senal = np.concatenate([pob.a_senal[vive], pob.a_senal[hijos]])
    pob.b_senal = np.concatenate([pob.b_senal[vive], pob.b_senal[hijos]])

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
