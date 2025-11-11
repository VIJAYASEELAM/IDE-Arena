import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'AfterQuery Coding Harness Trajectories',
  description: 'AfterQuery Coding Harness Trajectories',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
