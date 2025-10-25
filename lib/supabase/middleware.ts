import { createServerClient } from "@supabase/ssr"
import { NextResponse, type NextRequest } from "next/server"

// Paths configuration
const AUTH_ROOT = "/auth"
const LOGIN_PATH = `${AUTH_ROOT}/login`
const SIGNUP_PATH = `${AUTH_ROOT}/signup`
const VERIFY_EMAIL_PATH = `${AUTH_ROOT}/verify-email`
const ERROR_PATH = `${AUTH_ROOT}/error`
const DASHBOARD_PATH = "/dashboard"

// Helper to create a Supabase server client bound to middleware request/response
function createMiddlewareClient(request: NextRequest, responseRef: { current: NextResponse }) {
  return createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return request.cookies.getAll()
        },
        setAll(cookiesToSet) {
          cookiesToSet.forEach(({ name, value }) => request.cookies.set(name, value))
          responseRef.current = NextResponse.next({ request })
          cookiesToSet.forEach(({ name, value, options }) => responseRef.current.cookies.set(name, value, options))
        },
      },
    },
  )
}

export async function updateSession(request: NextRequest) {
  const responseRef = { current: NextResponse.next({ request }) }
  const supabase = createMiddlewareClient(request, responseRef)

  const { data, error: userError } = await supabase.auth.getUser()
  const user = data?.user

  const path = request.nextUrl.pathname
  const isAuthRoute = path.startsWith(AUTH_ROOT)
  const isPublic = path === "/" || path === "/home" || path === "/myths" || path === "/dos-donts" || path === "/first-aid" || path.startsWith("/api") || isAuthRoute

  // If user not logged in and accessing a protected route -> redirect to login with next param
  if (!user && !isPublic) {
    const url = request.nextUrl.clone()
    url.pathname = LOGIN_PATH
    url.searchParams.set("next", path)
    return NextResponse.redirect(url)
  }

  // If user is logged in but email not confirmed yet & not currently on verify page -> send them there
  if (user && !user.email_confirmed_at && path !== VERIFY_EMAIL_PATH) {
    // allow signup page redirect to verify, but block other auth pages
    const url = request.nextUrl.clone()
    url.pathname = VERIFY_EMAIL_PATH
    return NextResponse.redirect(url)
  }

  // Authenticated users hitting login/signup should go to dashboard
  if (user && (path === LOGIN_PATH || path === SIGNUP_PATH)) {
    const url = request.nextUrl.clone()
    url.pathname = DASHBOARD_PATH
    return NextResponse.redirect(url)
  }

  // Optional: If on verify page but email already confirmed -> dashboard
  if (user && user.email_confirmed_at && path === VERIFY_EMAIL_PATH) {
    const url = request.nextUrl.clone()
    url.pathname = DASHBOARD_PATH
    return NextResponse.redirect(url)
  }

  // Pass through (must return the supabase-bound response)
  return responseRef.current
}
