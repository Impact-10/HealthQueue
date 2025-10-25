import type { Metadata } from 'next'
import { GeistSans } from 'geist/font/sans'
import { GeistMono } from 'geist/font/mono'
import { Analytics } from '@vercel/analytics/next'
import './globals.css'

export const metadata: Metadata = {
  title: {
    default: 'DiagonsAI : Health Intelligence',
    template: 'DiagonsAI : %s',
  },
  description: 'DiagonsAI - AI assisted health insights, consultations, and wellness intelligence.',
  generator: 'DiagonsAI',
  applicationName: 'DiagonsAI',
  icons: [{ rel: 'icon', url: '/favicon.ico' }],
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`font-sans ${GeistSans.variable} ${GeistMono.variable} min-h-screen bg-gradient-to-b from-[#05070D] via-[#060b14] to-[#0b1320] antialiased selection:bg-primary/30` }>
        {children}
        <Analytics />
      </body>
    </html>
  )
}
