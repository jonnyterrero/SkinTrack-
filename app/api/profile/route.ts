import { NextResponse } from "next/server"
import { profileUpdateSchema } from "@/lib/validators/profile"
import {
  getAuthenticatedClient,
  unauthorized,
  dbError,
  validationError,
} from "@/lib/api/helpers"

export async function GET() {
  const { supabase, user } = await getAuthenticatedClient()
  if (!supabase || !user) return unauthorized()

  const { data, error } = await supabase
    .from("profiles")
    .select("*")
    .eq("id", user.id)
    .single()

  if (error) return dbError(error.message)
  return NextResponse.json(data)
}

export async function PUT(request: Request) {
  const { supabase, user } = await getAuthenticatedClient()
  if (!supabase || !user) return unauthorized()

  const body = await request.json()
  const parsed = profileUpdateSchema.safeParse(body)
  if (!parsed.success) return validationError(parsed.error)

  const { error, data } = await supabase
    .from("profiles")
    .update({ ...parsed.data, updated_at: new Date().toISOString() })
    .eq("id", user.id)
    .select()
    .single()

  if (error) return dbError(error.message)
  return NextResponse.json(data)
}
