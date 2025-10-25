"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Upload, X, File, Loader2 } from "lucide-react"
import { cn } from "@/lib/utils"

interface FileUploadProps {
  onUpload: (urls: string[]) => void
  onClose: () => void
}

export function FileUpload({ onUpload, onClose }: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const files = Array.from(e.dataTransfer.files)
    setSelectedFiles(files)
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    setSelectedFiles(files)
  }

  const handleUpload = async () => {
    if (selectedFiles.length === 0) return

    setIsUploading(true)
    const uploadedUrls: string[] = []

    try {
      for (const file of selectedFiles) {
        const formData = new FormData()
        formData.append("file", file)

        const response = await fetch("/api/upload", {
          method: "POST",
          body: formData,
        })

        if (!response.ok) {
          throw new Error(`Failed to upload ${file.name}`)
        }

        const data = await response.json()
        uploadedUrls.push(data.url)
      }

      onUpload(uploadedUrls)
    } catch (error) {
      console.error("Upload error:", error)
      alert("Failed to upload files. Please try again.")
    } finally {
      setIsUploading(false)
    }
  }

  const removeFile = (index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index))
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes"
    const k = 1024
    const sizes = ["Bytes", "KB", "MB", "GB"]
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Number.parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i]
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
      <Card className="w-full max-w-md bg-slate-900 border-slate-700">
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-white">Upload Files</CardTitle>
          <Button variant="ghost" size="sm" onClick={onClose} className="h-6 w-6 p-0">
            <X className="h-4 w-4" />
          </Button>
        </CardHeader>
        <CardContent className="space-y-4">
          <div
            className={cn(
              "border-2 border-dashed rounded-lg p-6 text-center transition-colors",
              isDragging ? "border-blue-500 bg-blue-500/10" : "border-slate-600",
            )}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <Upload className="mx-auto h-8 w-8 text-slate-400 mb-2" />
            <p className="text-slate-300 mb-2">Drag and drop files here</p>
            <p className="text-slate-400 text-sm mb-4">or</p>
            <Button
              variant="outline"
              onClick={() => fileInputRef.current?.click()}
              className="border-slate-600 text-slate-300 hover:bg-slate-800"
            >
              Choose Files
            </Button>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              onChange={handleFileSelect}
              className="hidden"
              accept=".pdf,.doc,.docx,.txt,.jpg,.jpeg,.png,.gif"
            />
          </div>

          {selectedFiles.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-slate-300">Selected Files:</h4>
              {selectedFiles.map((file, index) => (
                <div key={index} className="flex items-center justify-between bg-slate-800 p-2 rounded">
                  <div className="flex items-center gap-2 min-w-0">
                    <File className="h-4 w-4 text-slate-400 flex-shrink-0" />
                    <div className="min-w-0">
                      <p className="text-sm text-slate-300 truncate">{file.name}</p>
                      <p className="text-xs text-slate-400">{formatFileSize(file.size)}</p>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => removeFile(index)}
                    className="h-6 w-6 p-0 text-slate-400 hover:text-red-400"
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </div>
              ))}
            </div>
          )}

          <div className="flex gap-2">
            <Button
              variant="outline"
              onClick={onClose}
              className="flex-1 border-slate-600 text-slate-300 hover:bg-slate-800 bg-transparent"
            >
              Cancel
            </Button>
            <Button
              onClick={handleUpload}
              disabled={selectedFiles.length === 0 || isUploading}
              className="flex-1 bg-blue-600 hover:bg-blue-700"
            >
              {isUploading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Uploading...
                </>
              ) : (
                "Upload"
              )}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
