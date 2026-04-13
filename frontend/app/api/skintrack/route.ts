import { type NextRequest, NextResponse } from "next/server"

export async function GET() {
  try {
    return NextResponse.json({
      message:
        "SkinTrack+ stores health data in the browser (localStorage + IndexedDB). This route does not read your private records.",
      endpoint: "/api/skintrack",
      methods: ["GET", "POST"],
      persistence: "local-first",
      future:
        "Cloud backup uses the Supabase client from the browser (Data tab) after sign-in, not this route. See claude-supabase/supabase/schema.sql.",
      documentation: "Open the Data tab in the app for export/import and API notes.",
    })
  } catch {
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}

export async function POST(request: NextRequest) {
  try {
    const authHeader = request.headers.get("authorization")
    const apiKey = authHeader?.replace("Bearer ", "")

    if (!apiKey) {
      return NextResponse.json({ error: "API key required" }, { status: 401 })
    }

    const body = (await request.json()) as { type?: string; data?: unknown }

    if (!body.type || body.data === undefined) {
      return NextResponse.json({ error: "Invalid record format. Required: type, data" }, { status: 400 })
    }

    const record = {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      type: body.type,
      data: body.data,
    }

    return NextResponse.json({
      success: true,
      message:
        "Echo only: no server database write. Add the same entry in the app UI, or wait for Supabase-backed sync to be enabled.",
      record,
    })
  } catch {
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
