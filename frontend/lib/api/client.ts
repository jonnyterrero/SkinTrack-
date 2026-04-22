// Tiny fetch wrapper for first-party API routes. Same-origin, cookie auth.
// All helpers throw on non-2xx so callers can rely on resolved data.

type Json = Record<string, unknown> | Array<unknown>

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(path, { method: "GET", credentials: "same-origin" })
  if (!res.ok) throw new Error(await errorMessage(res))
  return (await res.json()) as T
}

export async function apiSend<T>(
  path: string,
  method: "POST" | "PUT" | "PATCH" | "DELETE",
  body?: Json,
): Promise<T | null> {
  const res = await fetch(path, {
    method,
    credentials: "same-origin",
    headers: body ? { "content-type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) throw new Error(await errorMessage(res))
  if (res.status === 204) return null
  return (await res.json()) as T
}

async function errorMessage(res: Response): Promise<string> {
  try {
    const data = await res.json()
    return typeof data?.error === "string" ? data.error : `HTTP ${res.status}`
  } catch {
    return `HTTP ${res.status}`
  }
}
