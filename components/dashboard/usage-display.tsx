"use client"

import { useState, useEffect } from "react"
import { createClient } from "@/lib/supabase/client"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { MessageSquare, Upload, AlertTriangle } from "lucide-react"

interface UsageDisplayProps {
  userId: string
}

interface UsageData {
  message_count: number
  file_upload_count: number
  can_send_message: boolean
  can_upload_file: boolean
  message_limit: number
  file_limit: number
}

export function UsageDisplay({ userId }: UsageDisplayProps) {
  const [usage, setUsage] = useState<UsageData | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const supabase = createClient()

  useEffect(() => {
    loadUsage()
    // Refresh usage every 30 seconds
    const interval = setInterval(loadUsage, 30000)
    return () => clearInterval(interval)
  }, [userId])

  const loadUsage = async () => {
    try {
      const { data, error } = await supabase.rpc("check_usage_limit", {
        p_user_id: userId,
        p_daily_message_limit: 50,
        p_daily_file_limit: 10,
      })

      if (error) throw error
      setUsage(data)
    } catch (error) {
      console.error("Error loading usage:", error)
    } finally {
      setIsLoading(false)
    }
  }

  if (isLoading || !usage) {
    return (
      <Card className="bg-slate-900 border-slate-800">
        <CardContent className="p-4">
          <div className="text-slate-400 text-sm">Loading usage...</div>
        </CardContent>
      </Card>
    )
  }

  const messagePercentage = (usage.message_count / usage.message_limit) * 100
  const filePercentage = (usage.file_upload_count / usage.file_limit) * 100

  return (
    <Card className="bg-slate-900 border-slate-800">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm text-slate-200 flex items-center gap-2">
          Daily Usage
          {(!usage.can_send_message || !usage.can_upload_file) && <AlertTriangle className="h-4 w-4 text-amber-500" />}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center gap-1 text-slate-300">
              <MessageSquare className="h-3 w-3" />
              Messages
            </div>
            <span className="text-slate-400">
              {usage.message_count}/{usage.message_limit}
            </span>
          </div>
          <Progress
            value={messagePercentage}
            className="h-2"
            style={{
              backgroundColor: "rgb(51 65 85)", // slate-700
            }}
          />
          {!usage.can_send_message && <p className="text-xs text-amber-400">Daily message limit reached</p>}
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center gap-1 text-slate-300">
              <Upload className="h-3 w-3" />
              File Uploads
            </div>
            <span className="text-slate-400">
              {usage.file_upload_count}/{usage.file_limit}
            </span>
          </div>
          <Progress
            value={filePercentage}
            className="h-2"
            style={{
              backgroundColor: "rgb(51 65 85)", // slate-700
            }}
          />
          {!usage.can_upload_file && <p className="text-xs text-amber-400">Daily file upload limit reached</p>}
        </div>

        <div className="text-xs text-slate-500 pt-2 border-t border-slate-800">Limits reset daily at midnight UTC</div>
      </CardContent>
    </Card>
  )
}
