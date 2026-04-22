"use client"

import { useRef, useState } from "react"
import type { BodyRegion, BodyView, BodySide } from "@/lib/types/backend"
import { Button } from "@/components/ui/button"

export type BodyPin = {
  id?: string
  body_view: BodyView
  body_region: BodyRegion
  side: BodySide | null
  loc_x: number
  loc_y: number
}

/**
 * Rough region boxes in normalized 0..1 space, one set per view.
 * Picked from the SVG silhouette below — good enough for bucketing.
 */
const REGIONS: Record<BodyView, Array<{
  region: BodyRegion
  side: BodySide
  x0: number; y0: number; x1: number; y1: number
}>> = {
  front: [
    { region: "head",       side: "midline", x0: 0.40, y0: 0.00, x1: 0.60, y1: 0.08 },
    { region: "face",       side: "midline", x0: 0.40, y0: 0.04, x1: 0.60, y1: 0.10 },
    { region: "neck",       side: "midline", x0: 0.44, y0: 0.10, x1: 0.56, y1: 0.14 },
    { region: "shoulder",   side: "left",    x0: 0.22, y0: 0.14, x1: 0.40, y1: 0.18 },
    { region: "shoulder",   side: "right",   x0: 0.60, y0: 0.14, x1: 0.78, y1: 0.18 },
    { region: "chest",      side: "midline", x0: 0.36, y0: 0.14, x1: 0.64, y1: 0.28 },
    { region: "arm_upper",  side: "left",    x0: 0.14, y0: 0.18, x1: 0.30, y1: 0.34 },
    { region: "arm_upper",  side: "right",   x0: 0.70, y0: 0.18, x1: 0.86, y1: 0.34 },
    { region: "abdomen",    side: "midline", x0: 0.36, y0: 0.28, x1: 0.64, y1: 0.44 },
    { region: "arm_lower",  side: "left",    x0: 0.10, y0: 0.34, x1: 0.26, y1: 0.50 },
    { region: "arm_lower",  side: "right",   x0: 0.74, y0: 0.34, x1: 0.90, y1: 0.50 },
    { region: "hand",       side: "left",    x0: 0.06, y0: 0.50, x1: 0.22, y1: 0.56 },
    { region: "hand",       side: "right",   x0: 0.78, y0: 0.50, x1: 0.94, y1: 0.56 },
    { region: "hip",        side: "left",    x0: 0.30, y0: 0.44, x1: 0.46, y1: 0.52 },
    { region: "hip",        side: "right",   x0: 0.54, y0: 0.44, x1: 0.70, y1: 0.52 },
    { region: "genital",    side: "midline", x0: 0.44, y0: 0.50, x1: 0.56, y1: 0.54 },
    { region: "thigh",      side: "left",    x0: 0.30, y0: 0.52, x1: 0.48, y1: 0.70 },
    { region: "thigh",      side: "right",   x0: 0.52, y0: 0.52, x1: 0.70, y1: 0.70 },
    { region: "knee",       side: "left",    x0: 0.30, y0: 0.68, x1: 0.48, y1: 0.74 },
    { region: "knee",       side: "right",   x0: 0.52, y0: 0.68, x1: 0.70, y1: 0.74 },
    { region: "leg_lower",  side: "left",    x0: 0.30, y0: 0.74, x1: 0.48, y1: 0.92 },
    { region: "leg_lower",  side: "right",   x0: 0.52, y0: 0.74, x1: 0.70, y1: 0.92 },
    { region: "foot",       side: "left",    x0: 0.28, y0: 0.92, x1: 0.48, y1: 1.00 },
    { region: "foot",       side: "right",   x0: 0.52, y0: 0.92, x1: 0.72, y1: 1.00 },
  ],
  back: [
    { region: "head",       side: "midline", x0: 0.40, y0: 0.00, x1: 0.60, y1: 0.08 },
    { region: "neck",       side: "midline", x0: 0.44, y0: 0.10, x1: 0.56, y1: 0.14 },
    { region: "shoulder",   side: "right",   x0: 0.22, y0: 0.14, x1: 0.40, y1: 0.18 },
    { region: "shoulder",   side: "left",    x0: 0.60, y0: 0.14, x1: 0.78, y1: 0.18 },
    { region: "back_upper", side: "midline", x0: 0.36, y0: 0.14, x1: 0.64, y1: 0.30 },
    { region: "back_lower", side: "midline", x0: 0.36, y0: 0.30, x1: 0.64, y1: 0.44 },
    { region: "arm_upper",  side: "right",   x0: 0.14, y0: 0.18, x1: 0.30, y1: 0.34 },
    { region: "arm_upper",  side: "left",    x0: 0.70, y0: 0.18, x1: 0.86, y1: 0.34 },
    { region: "arm_lower",  side: "right",   x0: 0.10, y0: 0.34, x1: 0.26, y1: 0.50 },
    { region: "arm_lower",  side: "left",    x0: 0.74, y0: 0.34, x1: 0.90, y1: 0.50 },
    { region: "glute",      side: "right",   x0: 0.36, y0: 0.44, x1: 0.50, y1: 0.54 },
    { region: "glute",      side: "left",    x0: 0.50, y0: 0.44, x1: 0.64, y1: 0.54 },
    { region: "thigh",      side: "right",   x0: 0.30, y0: 0.52, x1: 0.48, y1: 0.70 },
    { region: "thigh",      side: "left",    x0: 0.52, y0: 0.52, x1: 0.70, y1: 0.70 },
    { region: "knee",       side: "right",   x0: 0.30, y0: 0.68, x1: 0.48, y1: 0.74 },
    { region: "knee",       side: "left",    x0: 0.52, y0: 0.68, x1: 0.70, y1: 0.74 },
    { region: "leg_lower",  side: "right",   x0: 0.30, y0: 0.74, x1: 0.48, y1: 0.92 },
    { region: "leg_lower",  side: "left",    x0: 0.52, y0: 0.74, x1: 0.70, y1: 0.92 },
    { region: "foot",       side: "right",   x0: 0.28, y0: 0.92, x1: 0.48, y1: 1.00 },
    { region: "foot",       side: "left",    x0: 0.52, y0: 0.92, x1: 0.72, y1: 1.00 },
  ],
}

function classify(view: BodyView, x: number, y: number):
  { region: BodyRegion; side: BodySide } {
  for (const r of REGIONS[view]) {
    if (x >= r.x0 && x <= r.x1 && y >= r.y0 && y <= r.y1) {
      return { region: r.region, side: r.side }
    }
  }
  return { region: "other", side: x < 0.5 ? "left" : "right" }
}

type Props = {
  pins: BodyPin[]
  onAdd?: (pin: BodyPin) => void
  onRemove?: (pin: BodyPin) => void
  readOnly?: boolean
}

export function BodyMap({ pins, onAdd, onRemove, readOnly = false }: Props) {
  const [view, setView] = useState<BodyView>("front")
  const svgRef = useRef<SVGSVGElement | null>(null)

  function handleClick(e: React.MouseEvent<SVGSVGElement>) {
    if (readOnly || !onAdd) return
    const svg = svgRef.current
    if (!svg) return
    const pt = svg.createSVGPoint()
    pt.x = e.clientX
    pt.y = e.clientY
    const ctm = svg.getScreenCTM()
    if (!ctm) return
    const local = pt.matrixTransform(ctm.inverse())
    const x = local.x / 100
    const y = local.y / 200
    if (x < 0 || x > 1 || y < 0 || y > 1) return
    const { region, side } = classify(view, x, y)
    onAdd({ body_view: view, body_region: region, side, loc_x: x, loc_y: y })
  }

  const viewPins = pins.filter((p) => p.body_view === view)

  return (
    <div className="space-y-2">
      <div className="flex gap-2">
        <Button
          size="sm"
          variant={view === "front" ? "default" : "outline"}
          onClick={() => setView("front")}
        >
          Front
        </Button>
        <Button
          size="sm"
          variant={view === "back" ? "default" : "outline"}
          onClick={() => setView("back")}
        >
          Back
        </Button>
      </div>

      <div className="mx-auto w-full max-w-[260px] select-none">
        <svg
          ref={svgRef}
          viewBox="0 0 100 200"
          className="w-full h-auto border border-slate-200 rounded bg-white cursor-crosshair"
          onClick={handleClick}
          role="img"
          aria-label={`Body map – ${view}`}
        >
          <Silhouette view={view} />
          {viewPins.map((p, i) => (
            <g
              key={p.id ?? `new-${i}`}
              onClick={(e) => {
                e.stopPropagation()
                if (onRemove) onRemove(p)
              }}
              className="cursor-pointer"
            >
              <circle
                cx={p.loc_x * 100}
                cy={p.loc_y * 200}
                r={2.4}
                fill="#dc2626"
                stroke="white"
                strokeWidth={0.6}
              />
            </g>
          ))}
        </svg>
      </div>

      <p className="text-xs text-slate-500 text-center">
        Tap to add a pin · Tap a pin to remove
      </p>
    </div>
  )
}

function Silhouette({ view }: { view: BodyView }) {
  const outline =
    "M50 4 C55 4 58 8 58 13 C58 18 55 22 50 22 C45 22 42 18 42 13 C42 8 45 4 50 4 Z" +
    "M38 24 L62 24 L66 30 L84 40 L82 80 L74 82 L70 60 L66 90 L66 110 L58 110 L56 150 L54 190 L46 190 L44 150 L42 110 L34 110 L34 90 L30 60 L26 82 L18 80 L16 40 L34 30 Z"

  const backOutline =
    "M50 4 C55 4 58 8 58 13 C58 18 55 22 50 22 C45 22 42 18 42 13 C42 8 45 4 50 4 Z" +
    "M38 24 L62 24 L66 30 L84 40 L82 80 L74 82 L70 60 L66 90 L66 110 L58 110 L56 150 L54 190 L46 190 L44 150 L42 110 L34 110 L34 90 L30 60 L26 82 L18 80 L16 40 L34 30 Z"

  return (
    <g fill="#f1f5f9" stroke="#94a3b8" strokeWidth={0.8}>
      <path d={view === "front" ? outline : backOutline} />
    </g>
  )
}
