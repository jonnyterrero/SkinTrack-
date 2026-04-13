export function dataUrlToBlob(dataUrl: string): Blob {
  const [meta, b64] = dataUrl.split(",")
  if (!b64 || !meta) {
    throw new Error("Invalid data URL")
  }
  const mime = /data:([^;]+);/.exec(meta)?.[1] ?? "application/octet-stream"
  const binary = atob(b64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i)
  }
  return new Blob([bytes], { type: mime })
}

export function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onloadend = () => {
      if (typeof reader.result === "string") resolve(reader.result)
      else reject(new Error("Failed to read blob"))
    }
    reader.onerror = () => reject(reader.error ?? new Error("read error"))
    reader.readAsDataURL(blob)
  })
}

export function generateImageRef(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID()
  }
  return `img_${Date.now()}_${Math.random().toString(36).slice(2, 12)}`
}
