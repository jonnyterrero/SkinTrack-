/** Pearson correlation coefficient; null if undefined (constant series or too few points). */
export function pearsonCorrelation(xs: number[], ys: number[]): number | null {
  if (xs.length !== ys.length || xs.length < 3) return null
  const n = xs.length
  const mx = xs.reduce((a, b) => a + b, 0) / n
  const my = ys.reduce((a, b) => a + b, 0) / n
  let num = 0
  let dx = 0
  let dy = 0
  for (let i = 0; i < n; i++) {
    const vx = xs[i]! - mx
    const vy = ys[i]! - my
    num += vx * vy
    dx += vx * vx
    dy += vy * vy
  }
  const den = Math.sqrt(dx * dy)
  if (den === 0) return null
  return num / den
}

export function formatCorrelation(r: number | null): string {
  if (r === null) return "—"
  return r.toFixed(2)
}
