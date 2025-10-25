import { redirect } from 'next/navigation'

export const dynamic = 'force-dynamic'
export const metadata = { title: 'Home' }

export default function RootPage() {
  redirect('/home')
}
