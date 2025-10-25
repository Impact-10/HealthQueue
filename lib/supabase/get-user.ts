import { createClient } from "./server"

/**
 * Fetch the currently authenticated user (server side) and return null on error.
 * Keeps a single place to adjust how we read the user to avoid scattered logic.
 */
export async function getCurrentUser() {
  const supabase = await createClient()
  const { data, error } = await supabase.auth.getUser()
  if (error) return { user: null, error }
  return { user: data.user, error: null }
}
