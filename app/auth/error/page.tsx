import { AuthErrorClient } from './error-client'

export const dynamic = 'force-dynamic'
export const metadata = { title: 'Auth Error' }

export default function AuthErrorPage() {
  return <AuthErrorClient />
}
