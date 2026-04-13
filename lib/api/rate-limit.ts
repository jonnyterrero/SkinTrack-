const WINDOW_MS = 60_000
const MAX_REQUESTS = 100

type TokenBucket = { count: number; resetAt: number }
const buckets = new Map<string, TokenBucket>()

export function checkRateLimit(userId: string): { allowed: boolean; remaining: number } {
  const now = Date.now()
  let bucket = buckets.get(userId)

  if (!bucket || now > bucket.resetAt) {
    bucket = { count: 0, resetAt: now + WINDOW_MS }
    buckets.set(userId, bucket)
  }

  bucket.count++

  if (bucket.count > MAX_REQUESTS) {
    return { allowed: false, remaining: 0 }
  }

  return { allowed: true, remaining: MAX_REQUESTS - bucket.count }
}

// Periodic cleanup to avoid unbounded memory growth
if (typeof setInterval !== "undefined") {
  setInterval(() => {
    const now = Date.now()
    for (const [key, bucket] of buckets) {
      if (now > bucket.resetAt) buckets.delete(key)
    }
  }, WINDOW_MS * 2)
}
