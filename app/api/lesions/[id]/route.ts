import { NextResponse } from "next/server"
import { updateLesionSchema } from "@/lib/validators/lesions"
import {
  getAuthenticatedClient,
  unauthorized,
  notFound,
  dbError,
  validationError,
} from "@/lib/api/helpers"

type Params = { params: Promise<{ id: string }> }

export async function GET(_request: Request, { params }: Params) {
  const { supabase, user } = await getAuthenticatedClient()
  if (!supabase || !user) return unauthorized()

  const { id } = await params
  const { data, error } = await supabase
    .from("lesions")
    .select("*")
    .eq("id", id)
    .eq("user_id", user.id)
    .single()

  if (error) {
    if (error.code === "PGRST116") return notFound("Lesion")
    return dbError(error.message)
  }
  return NextResponse.json(data)
}

export async function PATCH(request: Request, { params }: Params) {
  const { supabase, user } = await getAuthenticatedClient()
  if (!supabase || !user) return unauthorized()

  const { id } = await params
  const body = await request.json()
  const parsed = updateLesionSchema.safeParse(body)
  if (!parsed.success) return validationError(parsed.error)

  const { data, error } = await supabase
    .from("lesions")
    .update(parsed.data)
    .eq("id", id)
    .eq("user_id", user.id)
    .select()
    .single()

  if (error) {
    if (error.code === "PGRST116") return notFound("Lesion")
    return dbError(error.message)
  }
  return NextResponse.json(data)
}

export async function DELETE(_request: Request, { params }: Params) {
  const { supabase, user } = await getAuthenticatedClient()
  if (!supabase || !user) return unauthorized()

  const { id } = await params
  const { error } = await supabase
    .from("lesions")
    .delete()
    .eq("id", id)
    .eq("user_id", user.id)

  if (error) return dbError(error.message)
  return new NextResponse(null, { status: 204 })
}
