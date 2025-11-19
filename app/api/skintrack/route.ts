import { type NextRequest, NextResponse } from "next/server"

// GET - Retrieve all records
export async function GET(request: NextRequest) {
  try {
    // Check API key
    const authHeader = request.headers.get("authorization")
    const apiKey = authHeader?.replace("Bearer ", "")

    // In a real app, validate against stored API key
    // For now, we'll return a message about client-side storage

    return NextResponse.json({
      message: "SkinTrack+ uses client-side storage. Access data through the web interface or export feature.",
      endpoint: "/api/skintrack",
      methods: ["GET", "POST"],
      documentation: "See Integrations tab for full API documentation",
    })
  } catch (error) {
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}

// POST - Add a new record
export async function POST(request: NextRequest) {
  try {
    // Check API key
    const authHeader = request.headers.get("authorization")
    const apiKey = authHeader?.replace("Bearer ", "")

    if (!apiKey) {
      return NextResponse.json({ error: "API key required" }, { status: 401 })
    }

    const body = await request.json()

    // Validate record structure
    if (!body.type || !body.data) {
      return NextResponse.json({ error: "Invalid record format. Required: type, data" }, { status: 400 })
    }

    // Create record with timestamp
    const record = {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      type: body.type,
      data: body.data,
    }

    return NextResponse.json({
      success: true,
      message: "Record created. Note: SkinTrack+ uses client-side storage. Use the web interface to view.",
      record,
    })
  } catch (error) {
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
