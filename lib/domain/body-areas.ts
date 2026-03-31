import type { SeverityLevel } from "@/lib/types"

export type BodyAreaId = (typeof BODY_AREA_DEFINITIONS)[number]["id"]

export const BODY_AREA_DEFINITIONS = [
  { id: "head", name: "Head & Face", x: 50, y: 15 },
  { id: "neck", name: "Neck", x: 50, y: 25 },
  { id: "chest", name: "Chest", x: 50, y: 35 },
  { id: "left-arm", name: "Left Arm", x: 25, y: 40 },
  { id: "right-arm", name: "Right Arm", x: 75, y: 40 },
  { id: "abdomen", name: "Abdomen", x: 50, y: 50 },
  { id: "left-hand", name: "Left Hand", x: 15, y: 55 },
  { id: "right-hand", name: "Right Hand", x: 85, y: 55 },
  { id: "pelvis", name: "Pelvis", x: 50, y: 60 },
  { id: "left-thigh", name: "Left Thigh", x: 40, y: 70 },
  { id: "right-thigh", name: "Right Thigh", x: 60, y: 70 },
  { id: "left-knee", name: "Left Knee", x: 40, y: 80 },
  { id: "right-knee", name: "Right Knee", x: 60, y: 80 },
  { id: "left-foot", name: "Left Foot", x: 40, y: 95 },
  { id: "right-foot", name: "Right Foot", x: 60, y: 95 },
] as const

export function bodyAreaLabel(id: string): string {
  const found = BODY_AREA_DEFINITIONS.find((a) => a.id === id)
  return found?.name ?? id
}

export function severityForDisplay(severity: SeverityLevel | undefined): string {
  if (!severity) return "—"
  return severity
}
