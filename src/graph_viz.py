"""Interactive 3D transaction-graph renderer for the Streamlit demo.

`graph_3d_html` returns a self-contained HTML document that embeds
`3d-force-graph` (WebGL / Three.js) from the unpkg CDN as plain UMD scripts (the
robust path — an ES-module/import-map setup fails to load inside the Streamlit
component iframe and leaves a black canvas). It renders:

  * node SHAPE by type      - customer=sphere, merchant=box, transaction=cone
  * node COLOR by status    - fraud txn=red, genuine txn=green, others by type
  * node SIZE by degree      - hub customers/merchants stand out
  * neon GLOW                - additive-blended halo sprites (no postprocessing
                               dependency, so it always loads)
  * text LABELS              - canvas-texture sprites on hubs, merchants and the
                               new node (no external label library)
  * static EDGES             - solid links, gold + thick on hover
  * hover TOOLTIP + HIGHLIGHT - dims all but the hovered node & its neighbours
  * a NEW transaction node   - gold pulsing node the camera flies to
  * a subtle starfield background + gentle auto-rotate

Everything here is a pure string builder (no torch / no browser), so it is
cheap to unit-test and drop into `st.components.v1.html`.
"""
from __future__ import annotations

import json

# CDN scripts (UMD builds expose THREE and ForceGraph3D as globals)
_THREE_SRC = "https://unpkg.com/three@0.149.0/build/three.min.js"
_FG_SRC = "https://unpkg.com/3d-force-graph@1.70.19/dist/3d-force-graph.min.js"

_PALETTE = {
    "customer": "#38bdf8",   # sky
    "merchant": "#c084fc",   # violet
    "genuine": "#34d399",    # emerald
    "fraud": "#fb5f7a",      # rose
    "new": "#fde047",        # amber
    "dim": "#1b1e2e",
    "link": "#8fa8e8",
    "linkHi": "#fde047",
    "bg": "#05060d",
}

NEW_TXN_ID = "txn:__NEW__"
NEW_TXN_PREFIX = "txn:__NEW__"   # plural ids are this + the sequence number


def _cap_transactions(viz: dict, max_txns: int) -> dict:
    """Keep at most `max_txns` transaction nodes + the entities/links they touch.

    Injected `is_new` transactions are ALWAYS kept (they are appended last, so a
    naive head-cap would drop the very node the user just scored).
    """
    nodes = viz.get("nodes", [])
    links = viz.get("links", [])
    txns = [n for n in nodes if n.get("type") == "transaction"]
    new_txns = [t for t in txns if t.get("is_new")]
    old_txns = [t for t in txns if not t.get("is_new")]

    budget = max(0, max_txns - len(new_txns))
    keep_txn = {t["id"] for t in new_txns + old_txns[:budget]}
    kept_links = [l for l in links
                  if l.get("is_new") or l["source"] in keep_txn or l["target"] in keep_txn]
    referenced = {l["source"] for l in kept_links} | {l["target"] for l in kept_links}
    kept_nodes = [n for n in nodes
                  if n.get("is_new")
                  or (n.get("type") == "transaction" and n["id"] in keep_txn)
                  or (n.get("type") != "transaction" and n["id"] in referenced)]
    return {"nodes": kept_nodes, "links": kept_links}


def merge_new_transaction(viz: dict, record: dict, is_fraud: bool, score: float,
                          threshold: float) -> dict:
    """Return a COPY of `viz` with the user's just-scored transaction injected.

    The new transaction links to its customer/merchant, reusing those nodes if
    they already exist in the sample (so the edge connects to real data) or
    adding cold-start nodes otherwise. The new nodes/links are flagged
    ``is_new`` so the renderer highlights and pulses them.
    """
    nodes = [n for n in viz.get("nodes", []) if n.get("id") != NEW_TXN_ID]
    links = [l for l in viz.get("links", [])
             if NEW_TXN_ID not in (l.get("source"), l.get("target"))]
    ids = {n["id"] for n in nodes}

    cc = str(record.get("cc_num", "?"))
    merch = str(record.get("merchant", "?"))
    cust_id, merch_id = f"cust:{cc}", f"merch:{merch}"

    nodes.append({
        "id": NEW_TXN_ID, "type": "transaction",
        "is_fraud": int(bool(is_fraud)), "is_new": 1,
        "raw": {
            "verdict": "FRAUD" if is_fraud else "GENUINE",
            "recon_error": round(float(score), 6),
            "threshold": round(float(threshold), 6),
            "amt": record.get("amt"),
            "category": record.get("category"),
            "datetime": record.get("trans_date_trans_time"),
        },
    })
    if cust_id not in ids:
        nodes.append({"id": cust_id, "type": "customer", "is_new": 1,
                      "raw": {"cc_num": cc, "gender": record.get("gender"),
                              "city_pop": record.get("city_pop")}})
    if merch_id not in ids:
        nodes.append({"id": merch_id, "type": "merchant", "is_new": 1,
                      "raw": {"merchant": merch, "category": record.get("category")}})
    links.append({"source": cust_id, "target": NEW_TXN_ID, "relation": "makes", "is_new": 1})
    links.append({"source": merch_id, "target": NEW_TXN_ID, "relation": "sells", "is_new": 1})
    return {"nodes": nodes, "links": links}


def merge_new_transactions(viz: dict, entries: list) -> dict:
    """Return a COPY of `viz` with MANY scored transactions injected at once.

    Each entry is a dict ``{record, is_fraud, score, threshold, color, seq}``.
    Every injected transaction gets a unique id (``txn:__NEW__<seq>``) and its
    own `color` so the renderer highlights each in a distinct hue; customers and
    merchants are shared across entries (and with the sampled graph) when their
    ids match. The most-recent entry is flagged ``is_latest`` for the camera.
    """
    nodes = list(viz.get("nodes", []))
    links = list(viz.get("links", []))
    ids = {n["id"] for n in nodes}

    for e in entries:
        rec = e["record"]
        cc = str(rec.get("cc_num", "?"))
        merch = str(rec.get("merchant", "?"))
        cust_id, merch_id = f"cust:{cc}", f"merch:{merch}"
        txn_id = f"{NEW_TXN_PREFIX}{e['seq']}"

        nodes.append({
            "id": txn_id, "type": "transaction",
            "is_fraud": int(bool(e["is_fraud"])), "is_new": 1,
            "color": e.get("color"), "tag": f"TXN {e['seq']}",
            "raw": {
                "verdict": "FRAUD" if e["is_fraud"] else "GENUINE",
                "recon_error": round(float(e["score"]), 6),
                "threshold": round(float(e["threshold"]), 6),
                "amt": rec.get("amt"),
                "category": rec.get("category"),
                "datetime": rec.get("trans_date_trans_time"),
            },
        })
        ids.add(txn_id)
        if cust_id not in ids:
            nodes.append({"id": cust_id, "type": "customer", "is_new": 1,
                          "raw": {"cc_num": cc, "gender": rec.get("gender"),
                                  "city_pop": rec.get("city_pop")}})
            ids.add(cust_id)
        if merch_id not in ids:
            nodes.append({"id": merch_id, "type": "merchant", "is_new": 1,
                          "raw": {"merchant": merch, "category": rec.get("category")}})
            ids.add(merch_id)
        links.append({"source": cust_id, "target": txn_id, "relation": "makes", "is_new": 1})
        links.append({"source": merch_id, "target": txn_id, "relation": "sells", "is_new": 1})

    if entries:
        latest_id = f"{NEW_TXN_PREFIX}{entries[-1]['seq']}"
        for n in nodes:
            if n.get("id") == latest_id:
                n["is_latest"] = 1
    return {"nodes": nodes, "links": links}


def graph_3d_html(viz_graph: dict, height: int = 50, max_txns: int = 100) -> str:
    """Build the embeddable HTML for the 3D graph. Empty graph -> a notice."""
    viz = viz_graph or {}
    if not viz.get("nodes"):
        return "<div style='color:#888;padding:1rem'>No graph data in this model artifact.</div>"

    viz = _cap_transactions(viz, max_txns)
    nodes, links = viz["nodes"], viz["links"]
    n_txn = sum(1 for n in nodes if n.get("type") == "transaction")
    n_fraud = sum(1 for n in nodes if n.get("is_fraud") == 1)

    header = (
        f'{len(nodes)} nodes &middot; {n_txn} transactions ({n_fraud} fraud) &middot; '
        f'<span style="color:{_PALETTE["genuine"]}">&#9679; genuine</span> '
        f'<span style="color:{_PALETTE["fraud"]}">&#9679; fraud</span> '
        '&#9733; your scored transactions (each its own colour) &middot; '
        'sphere=customer, box=merchant, cone=transaction &middot; '
        'drag to rotate, scroll to zoom, hover to inspect'
    )
    return (_TEMPLATE
            .replace("__THREE__", _THREE_SRC)
            .replace("__FG__", _FG_SRC)
            .replace("__HEIGHT__", str(int(height)))
            .replace("__HEADER__", header)
            .replace("__PALETTE__", json.dumps(_PALETTE))
            .replace("__DATA__", json.dumps({"nodes": nodes, "links": links})))


_TEMPLATE = r"""
<div style="font:13px sans-serif;color:#ccc;margin-bottom:4px">__HEADER__</div>
<div id="graph" style="width:100%;height:__HEIGHT__px;background:#05060d;border-radius:8px;overflow:hidden"></div>
<!-- resolve the optional `import('three/webgpu')` probe so 3d-force-graph falls back to WebGL
     instead of throwing "Failed to resolve module specifier three/webgpu" -->
<script type="importmap">
{ "imports": {
  "three": "https://unpkg.com/three@0.149.0/build/three.module.js",
  "three/webgpu": "https://unpkg.com/three@0.149.0/build/three.module.js",
  "three/tsl": "https://unpkg.com/three@0.149.0/build/three.module.js"
}}
</script>
<script src="__THREE__"></script>
<script src="__FG__"></script>
<script>
const DATA = __DATA__;
const P = __PALETTE__;

// node degree (for size-by-importance and label thresholds)
const DEG = {};
DATA.nodes.forEach(n => DEG[n.id] = 0);
DATA.links.forEach(l => {
  const s = typeof l.source === "object" ? l.source.id : l.source;
  const t = typeof l.target === "object" ? l.target.id : l.target;
  DEG[s] = (DEG[s] || 0) + 1; DEG[t] = (DEG[t] || 0) + 1;
});

// only the injected TRANSACTION is the gold "your txn"; its customer/merchant
// keep their normal type colour so the three are distinguishable
function isNewTxn(node) { return node.is_new && node.type === "transaction"; }

function baseColor(node) {
  if (isNewTxn(node)) return node.color || P.new;   // each injected txn its own hue
  if (node.type === "transaction") return node.is_fraud === 1 ? P.fraud : P.genuine;
  return P[node.type] || "#888888";
}
function nodeRadius(node) {
  if (isNewTxn(node)) return 8;
  const base = node.type === "transaction" ? 3 : 4;
  return base + Math.min(7, Math.sqrt(DEG[node.id] || 0) * 1.7);
}
function labelText(node) {
  const r = node.raw || {};
  if (isNewTxn(node)) return node.tag || "YOUR TXN";
  if (node.type === "transaction") return String(r.category || "txn");
  if (node.type === "merchant") return String(r.merchant || node.id).replace(/^merch:/, "").slice(0, 18);
  return String(r.cc_num || node.id).replace(/^cust:/, "").slice(0, 12);
}
function shouldLabel(node) {
  return node.is_new || node.type === "merchant" || (DEG[node.id] || 0) >= 5;
}

// ---- procedural sprite textures (no external label/glow library) ----
let HALO_TEX = null;
function haloTexture() {
  if (HALO_TEX) return HALO_TEX;
  const c = document.createElement("canvas"); c.width = c.height = 128;
  const ctx = c.getContext("2d");
  const g = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.25, "rgba(255,255,255,0.55)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g; ctx.fillRect(0, 0, 128, 128);
  HALO_TEX = new THREE.CanvasTexture(c);
  return HALO_TEX;
}
function makeHalo(color, r) {
  const mat = new THREE.SpriteMaterial({
    map: haloTexture(), color: new THREE.Color(color), transparent: true,
    opacity: 0.6, blending: THREE.AdditiveBlending, depthWrite: false,
  });
  const sp = new THREE.Sprite(mat); const s = r * 4.6; sp.scale.set(s, s, 1);
  return sp;
}
function makeLabel(text, color, big) {
  const c = document.createElement("canvas");
  const ctx = c.getContext("2d");
  const font = 48;
  ctx.font = `bold ${font}px sans-serif`;
  const w = Math.ceil(ctx.measureText(text).width);
  c.width = w + 24; c.height = font + 24;
  ctx.font = `bold ${font}px sans-serif`;
  ctx.fillStyle = color; ctx.textBaseline = "middle";
  ctx.fillText(text, 12, c.height / 2);
  const tex = new THREE.CanvasTexture(c); tex.minFilter = THREE.LinearFilter;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
  const sp = new THREE.Sprite(mat);
  const h = big ? 9 : 5.5;
  sp.scale.set(h * c.width / c.height, h, 1);
  return sp;
}

// adjacency for hover highlighting
const NEIGHBORS = {}, LINKS_OF = {};
DATA.nodes.forEach(n => { NEIGHBORS[n.id] = new Set(); LINKS_OF[n.id] = new Set(); });
DATA.links.forEach((l, i) => {
  const s = typeof l.source === "object" ? l.source.id : l.source;
  const t = typeof l.target === "object" ? l.target.id : l.target;
  NEIGHBORS[s].add(t); NEIGHBORS[t].add(s);
  LINKS_OF[s].add(i); LINKS_OF[t].add(i);
});

let hlNodes = new Set(), hlLinks = new Set();
const meshMap = {};          // id -> {mesh, halo, node}
const pulseItems = [];       // {mesh, halo} to animate (is_new)

function makeObject(node) {
  const r = nodeRadius(node);
  let geo;
  if (node.type === "customer") geo = new THREE.SphereGeometry(r, 18, 18);
  else if (node.type === "merchant") geo = new THREE.BoxGeometry(1.7*r, 1.7*r, 1.7*r);
  else geo = new THREE.ConeGeometry(r, 2.2*r, 20);
  const col = baseColor(node);
  const mat = new THREE.MeshStandardMaterial({
    color: col, emissive: col, emissiveIntensity: isNewTxn(node) ? 1.1 : 0.75,
    roughness: 0.35, metalness: 0.15, transparent: true, opacity: 0.96,
  });
  const mesh = new THREE.Mesh(geo, mat);
  const halo = makeHalo(col, r);
  meshMap[node.id] = { mesh, halo, node };
  if (isNewTxn(node)) pulseItems.push({ mesh, halo });

  const group = new THREE.Group();
  group.add(halo); group.add(mesh);
  if (shouldLabel(node)) {
    const lbl = makeLabel(labelText(node), isNewTxn(node) ? (node.color || P.new) : "#e6ecff", isNewTxn(node));
    lbl.position.set(0, r + 7, 0);
    group.add(lbl);
  }
  return group;
}

function updateStyles() {
  const anyHover = hlNodes.size > 0;
  for (const id in meshMap) {
    const { mesh, halo, node } = meshMap[id];
    // keep the user's whole injected sub-graph (txn + its customer/merchant) lit
    const keepBright = node.is_new;
    const on = keepBright || !anyHover || hlNodes.has(id);
    const col = on ? baseColor(node) : P.dim;
    mesh.material.color.set(col);
    mesh.material.emissive.set(col);
    mesh.material.opacity = on ? 0.96 : 0.12;
    if (!isNewTxn(node)) mesh.material.emissiveIntensity = on ? 0.75 : 0.05;
    halo.material.color.set(col);
    halo.material.opacity = keepBright ? 0.75 : (on ? 0.6 : 0.06);
  }
}

function tooltip(node) {
  const raw = node.raw || {};
  const rows = Object.keys(raw).map(k =>
    `<tr><td style="color:#8a8aa0;padding-right:10px">${k}</td>`
    + `<td style="color:#eee">${raw[k]}</td></tr>`).join("");
  const tag = isNewTxn(node) ? " &#11088; your transaction"
            : (node.is_new ? " (new)" : "");
  return `<div style="font:12px sans-serif;background:#12131f;`
       + `padding:8px 10px;border-radius:6px;border:1px solid #3a3a4a">`
       + `<b style="color:${baseColor(node)}">${node.type}${tag}</b>`
       + `<table style="margin-top:4px">${rows}</table></div>`;
}

const Graph = ForceGraph3D()(document.getElementById("graph"))
  .backgroundColor(P.bg)
  .graphData(DATA)
  .nodeThreeObject(makeObject)
  .nodeLabel(tooltip)
  .linkColor(l => hlLinks.has(l) ? P.linkHi : (l.is_new ? P.new : P.link))
  .linkWidth(l => (hlLinks.has(l) || l.is_new) ? 3.0 : 1.8)
  .linkOpacity(0.9)
  .linkCurvature(0.1)
  .onNodeHover(node => {
    hlNodes = new Set(); hlLinks = new Set();
    if (node) {
      hlNodes.add(node.id);
      NEIGHBORS[node.id].forEach(id => hlNodes.add(id));
      LINKS_OF[node.id].forEach(i => hlLinks.add(Graph.graphData().links[i]));
    }
    updateStyles();
    Graph.linkColor(Graph.linkColor()).linkWidth(Graph.linkWidth());
  });

// lighting for the emissive/standard materials
Graph.scene().add(new THREE.AmbientLight(0xffffff, 0.55));
const dir = new THREE.DirectionalLight(0xffffff, 0.65); dir.position.set(1, 1, 1);
Graph.scene().add(dir);

// subtle starfield background
const starGeo = new THREE.BufferGeometry();
const STARS = 1400, pos = new Float32Array(STARS * 3);
for (let i = 0; i < STARS * 3; i++) pos[i] = (Math.random() - 0.5) * 4600;
starGeo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
Graph.scene().add(new THREE.Points(starGeo,
  new THREE.PointsMaterial({ color: 0x8aa0d0, size: 1.6, transparent: true, opacity: 0.5 })));

// gentle auto-rotate until the user grabs the view
const ctrls = Graph.controls();
ctrls.autoRotate = true; ctrls.autoRotateSpeed = 0.6;
["mousedown", "touchstart", "wheel"].forEach(ev =>
  Graph.renderer().domElement.addEventListener(ev, () => { ctrls.autoRotate = false; }));

// pulse the injected "new" node(s)
function pulse() {
  const t = performance.now() * 0.004;
  const s = 1 + 0.28 * Math.sin(t);
  pulseItems.forEach(({ mesh, halo }) => {
    mesh.scale.set(s, s, s);
    mesh.material.emissiveIntensity = 0.9 + 0.5 * (0.5 + 0.5 * Math.sin(t));
    halo.material.opacity = 0.6 + 0.35 * (0.5 + 0.5 * Math.sin(t));
  });
  requestAnimationFrame(pulse);
}
pulse();

// fly the camera to the new node once the layout settles
let focused = false;
const newNode = DATA.nodes.find(n => n.is_latest) || DATA.nodes.find(n => isNewTxn(n));
Graph.onEngineStop(() => {
  if (focused || !newNode || newNode.x === undefined) return;
  focused = true;
  const r = Math.hypot(newNode.x, newNode.y, newNode.z) || 1;
  const k = 1 + 130 / r;
  Graph.cameraPosition(
    { x: newNode.x * k, y: newNode.y * k, z: newNode.z * k + 130 }, newNode, 1600);
  ctrls.autoRotate = false;
});
</script>
"""
