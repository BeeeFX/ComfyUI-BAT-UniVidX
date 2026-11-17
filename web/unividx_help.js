/**
 * Help affordances for the 🦇 UniVidX nodes.
 *
 * Adds a "?" icon to the top-right of the node's title bar, plus a
 * right-click menu entry. Either opens a panel explaining every mode:
 * what goes in, what comes out, and when you'd pick it.
 *
 * Render-mode coverage
 * --------------------
 * The title button is drawn on the LiteGraph canvas, so it does NOT
 * appear in ComfyUI's Vue node renderer (the DOM-based "nodes 2.0"
 * mode) — that renderer's header component has no title-button hook.
 * Two other paths cover it:
 *
 *   - the right-click menu entry registered below, and
 *   - ComfyUI's own per-node help panel, which reads the markdown that
 *     unividx_wrapper_runtime/help_docs.py generates into web/docs/.
 *     That panel is render-mode independent.
 *
 * Implementation notes
 * --------------------
 * `node.addTitleButton()` is a first-class LiteGraph API in the ComfyUI
 * frontend (LGraphNode.addTitleButton -> LGraphButton). Title buttons are
 * laid out from the node's right edge leftwards ("Start flush with right
 * edge" in LGraphCanvas.drawNode), so a single button lands exactly at the
 * top right with no manual positioning.
 *
 * Button glyphs are drawn in the PrimeIcons font, so `text` must be the
 * icon's unicode codepoint rather than a CSS class name — `pi-question-circle`
 * is U+E959. The font ships with the frontend package.
 *
 * Clicks arrive via `onTitleButtonClick`, which the canvas calls before its
 * own mouse handling (so the click doesn't also drag the node).
 *
 * The right-click entry uses the node-level `getExtraMenuOptions` hook,
 * which both renderers honour: the canvas menu calls it directly, and the
 * Vue-nodes menu (useMoreOptionsMenu) builds its "litegraph" section from
 * the same `getNodeMenuOptions` call. So there is exactly one entry in
 * either mode — adding the newer extension-level `getNodeMenuItems` hook
 * as well would list it twice on the canvas.
 *
 * The mode reference itself is fetched from /bat_unividx/modes, which the
 * Python side generates from modes.py. It is deliberately NOT duplicated
 * here — an earlier version of this pack hand-mirrored the mode table into
 * JS and the two could drift apart silently. It goes through
 * `api.fetchApi` (i.e. /api/bat_unividx/modes, which ComfyUI registers
 * for every route) so it also works when ComfyUI is served under a
 * sub-path behind a reverse proxy; so do the relative imports below.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Nodes that get the help affordances. Decode is included because the
// mode reference explains what its four output slots mean per family;
// the Loader because every speed/quality knob lives on it and the
// tooltips alone can't carry the tradeoffs.
const HELP_NODES = new Set([
    "BAT_UniVidXLoader",
    "BAT_UniVidXSample",
    "BAT_UniVidXDecode",
    // The two options nodes carry the quality-for-speed trades, so the
    // explanation has to be reachable FROM them — a user who drops in
    // Cache Options shouldn't have to know the Loader's help covers it.
    "BAT_UniVidXCacheOptions",
    "BAT_UniVidXContextOptions",
]);
// PrimeIcons `pi-question-circle`. Written as an escape rather than a raw
// glyph so it survives editors and encoding round-trips intact.
const PI_QUESTION_CIRCLE = "\ue959";

let modeCache = null;

async function fetchModes() {
    if (modeCache) return modeCache;
    try {
        const resp = await api.fetchApi("/bat_unividx/modes");
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        modeCache = data.modes || [];
    } catch (err) {
        console.error("[BAT UniVidX] Could not load mode reference:", err);
        // Don't cache the failure — let the next open retry.
        return [];
    }
    return modeCache;
}

// ── Panel ──────────────────────────────────────────────────────────

function closePanel() {
    document.getElementById("bat-unividx-help")?.remove();
}

function buildRows(entries, currentMode) {
    return entries.map(e => {
        const inputs = e.inputs.length ? e.inputs.join(" + ") : "text only";
        const isCurrent = e.mode === currentMode;
        const row = document.createElement("tr");
        row.style.cssText = isCurrent
            ? "background:#2d4a2d;"
            : "border-bottom:1px solid #333;";

        const cells = [
            [e.mode + (isCurrent ? "  ←" : ""), "font-family:monospace;font-weight:bold;white-space:nowrap;"],
            [inputs, "font-family:monospace;color:#9cf;white-space:nowrap;"],
            [e.outputs.join(" + "), "font-family:monospace;color:#fc9;white-space:nowrap;"],
            [e.description, "color:#ddd;"],
        ];
        for (const [text, css] of cells) {
            const td = document.createElement("td");
            td.textContent = text;
            td.style.cssText = "padding:5px 10px;vertical-align:top;" + css;
            row.appendChild(td);
        }
        return row;
    });
}

async function openPanel(node) {
    closePanel();
    const entries = await fetchModes();
    const currentMode = node.widgets?.find(w => w.name === "mode")?.value;

    const overlay = document.createElement("div");
    overlay.id = "bat-unividx-help";
    overlay.style.cssText = `
        position:fixed; inset:0; z-index:10000;
        background:rgba(0,0,0,0.6);
        display:flex; align-items:center; justify-content:center;
        font-family:system-ui,sans-serif; font-size:13px;`;
    overlay.addEventListener("click", ev => {
        if (ev.target === overlay) closePanel();
    });

    const panel = document.createElement("div");
    panel.style.cssText = `
        background:#1e1e1e; color:#eee; border:1px solid #444;
        border-radius:8px; box-shadow:0 8px 32px rgba(0,0,0,0.6);
        max-width:min(1000px,92vw); max-height:86vh;
        display:flex; flex-direction:column;`;

    // Header
    const header = document.createElement("div");
    header.style.cssText = `
        padding:14px 18px; border-bottom:1px solid #444;
        display:flex; align-items:center; justify-content:space-between; gap:16px;`;
    const title = document.createElement("div");
    title.innerHTML = `<strong style="font-size:15px;">🦇 UniVidX — mode reference</strong>`;
    const closeBtn = document.createElement("button");
    closeBtn.textContent = "✕";
    closeBtn.style.cssText = `
        background:none; border:none; color:#aaa; font-size:18px;
        cursor:pointer; padding:0 4px; line-height:1;`;
    closeBtn.addEventListener("click", closePanel);
    header.append(title, closeBtn);

    // Body
    const body = document.createElement("div");
    body.style.cssText = "padding:14px 18px; overflow:auto;";

    if (!entries.length) {
        body.innerHTML = `<p style="color:#f99;">Couldn't load the mode list from the
            server. The naming scheme is <code>&lt;inputs&gt;2&lt;outputs&gt;</code> —
            e.g. <code>R2AIN</code> takes RGB and predicts Albedo + Irradiance + Normals.</p>`;
    } else {
        const intro = document.createElement("p");
        intro.style.cssText = "margin:0 0 12px; color:#bbb; line-height:1.5;";
        intro.innerHTML = `
            Modes are named <code>&lt;inputs&gt;2&lt;outputs&gt;</code>. The letters
            before the <code>2</code> are what you feed in; the letters after are what
            the model predicts. Picking a mode reshapes the node so it only has the
            sockets that mode uses.<br>
            <span style="color:#9cf;">Intrinsic</span>: R=RGB, A=albedo, I=irradiance,
            N=normals &nbsp;·&nbsp;
            <span style="color:#9cf;">Alpha</span>: R=composite, P=matte, F=foreground,
            B=background.<br>
            <em style="color:#999;">The mode's family must match the variant you loaded
            in the loader node.</em>`;
        body.appendChild(intro);

        for (const family of ["intrinsic", "alpha"]) {
            const group = entries.filter(e => e.family === family);
            if (!group.length) continue;

            const h = document.createElement("h3");
            h.textContent = family === "intrinsic"
                ? "Intrinsic — lighting decomposition"
                : "Alpha — matting / compositing";
            h.style.cssText = "margin:18px 0 6px; font-size:14px; color:#9cf;";
            body.appendChild(h);

            const table = document.createElement("table");
            table.style.cssText = "border-collapse:collapse; width:100%;";
            const thead = document.createElement("thead");
            const hrow = document.createElement("tr");
            for (const label of ["Mode", "Takes", "Gives", "What it's for"]) {
                const th = document.createElement("th");
                th.textContent = label;
                th.style.cssText = `
                    text-align:left; padding:5px 10px; color:#888;
                    border-bottom:1px solid #555; font-weight:600;`;
                hrow.appendChild(th);
            }
            thead.appendChild(hrow);
            table.appendChild(thead);
            const tbody = document.createElement("tbody");
            buildRows(group, currentMode).forEach(r => tbody.appendChild(r));
            table.appendChild(tbody);
            body.appendChild(table);
        }
    }

    panel.append(header, body);
    overlay.appendChild(panel);
    document.body.appendChild(overlay);
}

// Escape closes the panel.
window.addEventListener("keydown", ev => {
    if (ev.key === "Escape") closePanel();
});

// ── Extension registration ─────────────────────────────────────────

app.registerExtension({
    name: "BAT.UniVidXWrapper.Help",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!HELP_NODES.has(nodeData?.name)) return;

        // 1. Title-bar "?" button — canvas renderer only.
        //
        // Title buttons are drawn by LGraphCanvas onto the canvas. The
        // Vue node renderer builds nodes from DOM components instead,
        // and its header (NodeHeader.vue) renders only the collapse
        // toggle, title, price/status badges and pin — there's no
        // title-button hook, so this button simply won't appear there.
        // That's what the context-menu entry below and the built-in
        // help markdown (web/docs/*.md) are for.
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);

            // Guard: older frontends have no title-button API. Degrade
            // silently rather than throwing during node creation.
            if (typeof this.addTitleButton !== "function") return r;

            this.addTitleButton({
                name: "bat_unividx_help",
                text: PI_QUESTION_CIRCLE,
                fontSize: 14,
                xOffset: -6,
            });
            return r;
        };

        const onTitleButtonClick = nodeType.prototype.onTitleButtonClick;
        nodeType.prototype.onTitleButtonClick = function (button, canvas) {
            if (button?.name === "bat_unividx_help") {
                openPanel(this);
                return;
            }
            return onTitleButtonClick?.apply(this, arguments);
        };

        // 2. Right-click menu entry — works regardless of how the node
        // is rendered, so the mode reference stays reachable when the
        // title button isn't drawn.
        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
            const r = getExtraMenuOptions?.apply(this, arguments);
            if (Array.isArray(options)) {
                options.unshift({
                    content: "🦇 UniVidX mode reference",
                    callback: () => openPanel(this),
                });
            }
            return r;
        };
    },
});
