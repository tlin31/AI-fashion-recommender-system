# Architecture Diagram Style Guide
> Reference for regenerating diagrams in the style of `serving_path_diagram.html` and `learning_cycle.html`

---

## What This Style Produces

A self-contained HTML architecture diagram with:
- Colored zone blocks representing system boundaries (who owns what)
- Actor/component boxes with icons, service call pills, and ownership badges
- Decision diamonds for routing/branching logic
- Parallel columns for split paths
- Left and right side note panels
- Arrows with large black labels between blocks
- A comparison table or bottom callout summarizing key properties

---

## The Prompt Template

Use this as the base prompt when asking Claude to generate a new diagram:

```
Generate a self-contained HTML architecture flow diagram in the style of serving_path_diagram.html.

LAYOUT:
- Three-column CSS grid: [185px left notes] [1fr center flow] [185px right notes]
- White background (#f8f9fb), light card backgrounds, no dark mode
- Max-width 1260px, centered, padding 40px 48px 64px

ZONES (system boundary wrappers):
- Each major actor/subsystem gets a colored zone with a border-radius:10px rounded box
- Zone has a 2px solid border + light tinted background
- A small colored pill tag in top-left corner names the owner (e.g. "Gorse Core", "fashion-recommend")
- Colors:
    fashion-recommend → amber  (background:#fffbeb, border:#fcd34d, tag:#f59e0b)
    Gorse Core        → blue   (background:#eff6ff, border:#93c5fd, tag:#3b82f6)
    Shared Storage    → slate  (background:#f8fafc, border:#cbd5e1, tag:#6b7280)
    Gorse Worker      → green  (background:#f0fdf4, border:#86efac, tag:#10b981)
    Shared boundary   → purple (background:#f5f3ff, border:#c4b5fd, tag:#8b5cf6)

ACTOR/COMPONENT BLOCKS (inside zones):
- White background, 2px colored border, border-radius:8px, padding 11px 13px
- Contains: emoji icon (font-size:22px), bold UPPERCASE title, body text, service call pills, ownership badge
- Service call pills (inline-block, monospace font):
    GET request  → blue   background:#dbeafe color:#1e40af
    POST request → green  background:#d1fae5 color:#065f46
    Cache read   → purple background:#f5f3ff color:#5b21b6
    DB read      → amber  background:#fef3c7 color:#92400e
    gRPC call    → red    background:#fee2e2 color:#991b1b
- Ownership badges (tiny, bottom of block):
    Gorse   → background:#dbeafe color:#1e40af
    fashion → background:#fef3c7 color:#92400e

DECISION DIAMONDS:
- clip-path: polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%)
- Width ~140px, height ~68px, centered
- YES label in green (#16a34a), NO label in red (#dc2626)
- Tinted background matching the path color

ARROWS BETWEEN BLOCKS:
- font-size: 20px arrow character (↓) in gray (#9ca3af)
- Arrow label text: font-size 12.5px, font-weight 500, color #111827 (BLACK)
- Never use gray or muted color for arrow labels — always black

PARALLEL PATH COLUMNS (for branching flows):
- CSS grid: grid-template-columns: 1fr 1fr 1fr (or 1fr 1fr for two paths)
- Each column has a color-coded header (rounded top) + body (rounded bottom, matching border)
- Step nodes inside columns: smaller font (10.5px), same border color as path
- Step numbers: small colored circles (18×18px, border-radius:50%) matching path color
- Outcome box at column bottom: bold, tinted background, 2px border, centered text
- Path colors:
    Path A (happy path)  → green  #16a34a / #86efac / #f0fdf4
    Path B (fallback)    → amber  #d97706 / #fde047 / #fffbeb
    Path C (alternative) → blue   #2563eb / #93c5fd / #eff6ff

SIDE NOTES (left and right columns):
- White background, 1px #e5e7eb border, 3px solid left border (color matches topic)
- Title: 10px, bold, uppercase, letter-spacing
- Body: 10.5px, color #374151, line-height 1.65
- Inline code: background #f1f5f9, color #0369a1, font-family SF Mono
- Use these to explain: design decisions, tradeoffs, gotchas, "why this matters"
- Align note vertical position to match the block it annotates in the center column

HEADER:
- Centered h1 (22px, font-weight 700, color #111827)
- Subtitle in SF Mono, 11px, uppercase, color #9ca3af
- Ownership strip below: colored swatches (26×10px, border-radius:3px) + labels
- Path legend below that: colored dots (13×13px circles) + path names

BOTTOM CALLOUT:
- Full-width, background:#eff6ff, border:1.5px solid #93c5fd, border-radius:8px
- Font-size 12.5px, line-height 1.7
- States the single most important architectural principle of the diagram

FONTS:
- Body: 'Inter', 'Helvetica Neue', Arial, sans-serif
- Code: 'SF Mono', 'Fira Code', monospace
- Never use system dark mode — explicit light colors only
```

---

## Key Rules (Non-Negotiable)

1. **White / light background always.** Never dark mode, never `#0f1117` or similar.
2. **Arrow labels are always black** (`color: #111827`) and at least `12.5px`. Never gray.
3. **Every block shows who owns it** — ownership badge at the bottom of each actor box.
4. **Zones show system boundaries** — the colored zone wrapper is what communicates "this is Gorse Core" vs "this is fashion-recommend." Don't skip zones.
5. **Decision diamonds for any branch.** Any if/else in the flow gets a diamond, not a text note.
6. **Parallel columns for parallel paths.** Never stack alternative paths vertically — use grid columns so the viewer can compare at a glance.
7. **Side notes explain, center flow shows.** The center is the what; side notes are the why.
8. **Service calls are typed pills.** Any API call, cache lookup, or gRPC call shown as a colored monospace pill — not plain text.

---

## Anatomy of a Good Diagram Prompt

When asking for a new diagram, provide:

```
Draw an HTML architecture flow diagram in the style of serving_path_diagram.html for:

SUBJECT: [what system or flow you are diagramming]

ACTORS: [list the components/services involved and which system owns them]
  e.g. - Frontend (fashion-recommend)
        - API Server (Gorse Core)
        - Redis Cache (Shared Storage)

FLOW: [describe the steps in plain English, including any decision points or parallel paths]
  e.g. 1. User sends request
        2. Server checks cache
        3. IF cache hit → serve directly
           IF cache miss → fall back to CF lookup
        4. Filter and return

PATHS: [if there are multiple paths, name them]
  e.g. Path A (green) — happy path
       Path B (amber) — fallback

SIDE NOTES: [topics to annotate in the left/right panels]
  e.g. - Why the server is stateless
        - What self-healing means here

BOTTOM CALLOUT: [the single most important principle to highlight]
```

---

## File Naming Convention

```
docs/
  learning_cycle.html          → Diagram A
  serving_path_diagram.html    → Diagram B
  worker_pipeline.html         → Worker Pipeline
  diagram_style_guide.md       → this file
```

---

## What Makes This Style Work

The diagrams are effective because they separate three concerns visually:

| Visual element | Answers |
|---|---|
| Zone color / tag | **Who** owns this component |
| Block content + service pills | **What** it does and how it communicates |
| Arrow + label | **How** data or control flows between components |
| Side notes | **Why** the design is the way it is |
| Decision diamonds | **When** the flow branches |
| Parallel columns | **Which** path is taken under which condition |

Every element has one job. The viewer's eye moves top-to-bottom through the center, reads labels on arrows to understand the transitions, and glances left/right to understand context and tradeoffs.
