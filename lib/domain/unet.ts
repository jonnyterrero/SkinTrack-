/**
 * Optional lesion-segmentation / U-Net pipeline. Disabled unless explicitly enabled in env.
 */
export function isUnetAnalysisEnabled(): boolean {
  if (typeof process === "undefined") return false
  return process.env.NEXT_PUBLIC_ENABLE_UNET === "true" || process.env.NEXT_PUBLIC_ENABLE_UNET === "1"
}
