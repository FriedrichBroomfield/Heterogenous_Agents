# Contexto para continuar

Pegá esto entero al abrir la sesión nueva, con la carpeta del proyecto disponible.

---

Estoy construyendo una simulación basada en agentes para estudiar cómo emergen
rasgos y estrategias sin imponer funciones de utilidad. La idea central: en vez
de suponer cómo valoran los individuos, defino reglas físicas y observo qué
sobrevive. Las estrategias, los costos y el significado de las señales tienen
que ser **endógenos**.

La carpeta tiene la etapa 1 funcionando. Necesito continuar desde ahí.

## Archivos

- `mundo.py` — mundo, agentes, ciclo, disputa, creencias bayesianas
- `validar.py` — barrido del costo de pelea contra Halcón-Paloma
- `prejuicio.py` — corridas con y sin marcas
- `analiza.py` — tabla por semilla y estadísticos de decisión
- `figuras.py` — panel de 6 gráficos
- `todo.py` — corre todo: `python todo.py` (12 semillas × 1200 ciclos, ~30 min)

## Qué hay construido

**Mundo.** Parches en anillo con fruta que regenera logísticamente. Ciclo de T
pasos, después vuelta al nido, reproducción asexual, muerte por edad o energía.
Nido = linaje.

**Disputa sin matriz de pagos.** Si dos agentes coinciden en un parche, cada
uno decide ceder o escalar sin ver la decisión del otro. Ceden ambos → reparten
sin gasto. Uno escala → se lleva todo pero paga el gasto igual. Escalan ambos →
pelea, gana con probabilidad proporcional al tamaño, ambos pagan energía y
arriesgan lesión. El costo no es un parámetro llamado "costo": es energía
efectivamente perdida. Prueba: si puedo cambiarlo sin cambiar algo físico, está
mal.

**Marcas observables, vector de dos dimensiones.**
- Informativa: clase de tamaño, predice de verdad quién gana.
- Arbitraria: etiqueta heredable sin efecto causal. Pero como se hereda y los
  linajes difieren en tamaño, correlaciona con la capacidad de pelea. El
  prejuicio es a veces correcto, y por eso se mide contra la correlación
  **realizada**, no contra 0.5.

**Creencias bayesianas.** Beta por marca y valor sobre "gano si peleo contra
alguien con esta marca". Decisión por muestreo de Thompson: la exploración sale
de la incertidumbre, no de un parámetro. Solo pelear informa; ceder no dice si
habrías ganado. Al volver al nido se comparte lo observado con peso reducido y
heredable, y se comparten **eventos, no creencias**, lo que acota el doble
conteo. Cada ciclo las creencias decaen hacia el prior: la expectativa se
conserva y la confianza cae, así que sin evidencia nueva el agente vuelve a
explorar. Las crías heredan las creencias del progenitor.

**Rasgos heredables:** propensión base a escalar, peso de la creencia, peso del
testimonio, tasa de olvido, tamaño, etiqueta arbitraria.

## Estado de los resultados

**Validado:** la tasa de escalada sigue a min(1, V/C) en la zona de equilibrio
mixto. Es la única referencia analítica del modelo; sin ella no distinguiríamos
un resultado de un bug. Hay un sesgo hacia arriba que crece con C, explicable
por la variación de tamaño que Halcón-Paloma clásico no tiene. Falsable
poniendo `var_tam=0.0`; no lo verifiqué.

**Candidato firme:** sub-usan la marca informativa. La dispersión de sus
creencias queda por debajo de la diferencia real en 3/3 semillas. Desperdician
la señal que sí predice.

**NO establecido:** la sobre-discriminación de la marca arbitraria. Con 3
semillas la diferencia prej−real da z = +1.99, justo debajo del umbral. La
corrida de 12 semillas decide.

**Problemas abiertos:**
1. Nada converge en 350 ciclos. El peso del testimonio se movió de 0.25 a 0.63
   entre el primer y el último tercio en una semilla.
2. Las semillas terminan en regímenes distintos (peso del testimonio 0.24 /
   0.37 / 0.65). Promediarlas describe una población que no existe.
3. Doble conteo por el nido: la semilla con más peso de testimonio tuvo el peor
   error de creencia. Con 3 puntos no significa nada.
4. En una semilla la estructura de nidos colapsó a un solo linaje por tramos.

## Reglas metodológicas, aprendidas cayendo en cada trampa

Estas importan más que el código. En un proyecto anterior perdimos semanas por
no tenerlas.

1. **Toda métrica necesita un nulo antes de interpretarse.** Nunca reportar un
   número sin saber qué da cuando no hay nada que detectar.
2. **Si la estimación puntual se mueve con n, hay un bug estructural.** No es
   ruido. Nos pasó: una métrica dio −0.19 con 10 réplicas y −3.61 con 120.
3. **Cuidado con las razones de denominador chico.** `prej/real` explotaba sin
   que la conducta cambiara. Se reemplazó por la diferencia.
4. **Nunca sombrear solo donde la conclusión se cumple.** Teníamos una figura
   que solo podía confirmar.
5. **No promediar réplicas que están en regímenes distintos.** Reportar por
   semilla y medir el coeficiente de variación.
6. **Diversidad final ≠ polimorfismo mantenido.** En el proyecto anterior una
   métrica de diversidad seguía al tamaño poblacional, que es lo que predice
   deriva. Si hace falta medir coexistencia, el criterio es invasibilidad
   mutua, no composición final.
7. **Declarar predicciones antes de correr**, incluyendo qué contaría como
   "no pasó nada".
8. **Los diagnósticos delatan los bugs.** Fracción de extinción, largo de
   serie, saturación, fallo de fijación. Mirarlos antes que los resultados.
9. **Optimizar políticas por promedio penaliza la especialización antes de que
   corra la selección.** Nos costó una vuelta entera.
10. **Una corrida más rápida de lo esperado es sospechosa**, no una ventaja.
    Suele significar salidas tempranas.

## Qué sigue

**Inmediato:** correr `python todo.py` con 12 semillas y 1200 ciclos, y decidir
con el z de la diferencia arbitraria y la convergencia de los rasgos.

**Etapa 2 — peligro.** Algunos parches tienen depredador oculto, resorteado
cada ciclo. Forrajear ahí mata con probabilidad d. El presupuesto perceptual
pasa a repartirse entre tres canales que compiten: cantidad de fruta, presencia
de depredador, señales ajenas. Referencia: teoría de forrajeo bajo riesgo.

**Etapa 3 — señales, y es la parte difícil.** Diseño ya acordado:

- Separar señal **interna** (ceder / escalar, que nadie ve) de **externa**
  (exhibición pública). La brecha entre ambas es lo que hace posible el engaño:
  un agente puede exhibir "voy a pelear" y ceder por dentro. La honestidad deja
  de ser supuesto.
- Orden temporal obligatorio: exhibición → observación → decisión interna →
  resolución física. Si la decisión interna fuera antes de exhibir, la
  exhibición podría ser un reporte fiel y no habría nada que estudiar.
- Las señales externas son **fichas vacías**: un conjunto de emisiones
  distinguibles sin contenido asignado. Empezar con tres; con muchas el espacio
  de búsqueda hace que no emerja nada por combinatoria. El genoma lleva, por
  separado, con qué probabilidad emite cada ficha en cada estado interno y
  cuánto pesa cada ficha ajena al decidir. El significado aparece solo si la
  correlación entre ficha y estado del mundo se vuelve fiable.
- **Emitir y percibir son rasgos separados que deben coincidir.** Eso crea
  biestabilidad: cada rasgo es inútil sin el otro, así que ambos son deletéreos
  cuando son raros. Dos vías de arranque, y hay que separarlas
  experimentalmente: (a) el nido, porque un mutante que emite y percibe genera
  un nido entero correlacionado por descendencia; (b) exaptación, si el canal
  perceptual ya sirve para detectar depredadores. Apagar (a) con dispersión
  alta, apagar (b) con canal dedicado. Si la comunicación emerge sin ninguna,
  hay un bug.
- **Predicción declarada, y es el mejor test del diseño.** Las dos situaciones
  usan el mismo canal, las mismas fichas, el mismo costo de emisión, y difieren
  solo en la estructura de intereses. En la disputa los intereses están
  enfrentados, así que por el argumento de cheap talk las exhibiciones deberían
  colapsar a ruido. En el aviso de peligro están parcialmente alineados por el
  parentesco del nido, así que ahí puede estabilizarse. Mismo canal, resultados
  opuestos. Si emergen exhibiciones honestas en la disputa, o hay un canal de
  costo que no vimos, o hay un bug.
- **Hace falta una perilla de dispersión.** Con nido por linaje y reproducción
  asexual el parentesco es ~1 y la regla de Hamilton se cumple por aritmética,
  lo cual no es un test. Con una fracción de crías que se van a otro nido, se
  puede calcular el parentesco del pedigrí, medir beneficio y costo realizados,
  y **predecir la dispersión exacta a la que el aviso colapsa**. Si colapsa
  donde Hamilton dice, el modelo está bien construido.

**Etapa 4 — reproducción sexual.** No es una extensión menor. La recombinación
separa emisión y percepción cada generación, y la selección no puede sostener
un rasgo compuesto que solo paga junto. Predicción: la comunicación es mucho
más difícil bajo sexual que bajo asexual, y la diferencia crece con la tasa de
recombinación. Queda pendiente decidir cómo se heredan las creencias con dos
progenitores: promediar homogeneiza y probablemente mata la diversidad de
tradiciones; elegir una al azar la mantiene con ruido; transmitir el agregado
del nido convierte la tradición en propiedad del grupo.

## Cómo quiero que trabajes

Sé un revisor duro. Buscá activamente por qué el resultado podría ser un
artefacto antes de celebrarlo. Cuando propongas un ajuste después de ver un
resultado que no gustó, decilo explícitamente y decime cómo distinguir el
ajuste legítimo del tuneo. Si una figura o una métrica solo puede confirmar,
señalalo. Y cuando corras algo, mostrame el código.
