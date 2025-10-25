"use client"

import { useCallback, useEffect, useRef, useState } from 'react'

// Minimal ambient type shims (for browsers that support Web Speech). If lib.dom adds them natively, these merge.
// We keep them very loose to avoid maintenance overhead.
// eslint-disable-next-line @typescript-eslint/no-unused-vars
type _Any = any
// Only declare if not already present
// @ts-ignore
declare global {
  // @ts-ignore
  interface SpeechRecognition extends EventTarget {
    lang: string
    continuous: boolean
    interimResults: boolean
    start: () => void
    stop: () => void
    onaudioend?: () => void
    onend?: () => void
    onerror?: (ev: _Any) => void
    onresult?: (ev: _Any) => void
  }
  // @ts-ignore
  interface Window { SpeechRecognition?: any; webkitSpeechRecognition?: any }
}

interface SpeechOptions {
  interim?: boolean
  lang?: string
  continuous?: boolean
}

interface UseSpeechInput {
  supported: boolean
  listening: boolean
  interim: string
  final: string
  error: string | null
  start: () => void
  stop: () => void
  reset: () => void
}

// Lightweight Web Speech API hook (browser only). Gracefully no-ops if unsupported.
export function useSpeechInput(opts: SpeechOptions = {}): UseSpeechInput {
  const { interim = true, lang = 'en-US', continuous = false } = opts
  const [supported, setSupported] = useState(false)
  const [listening, setListening] = useState(false)
  const [interimTxt, setInterimTxt] = useState('')
  const [finalTxt, setFinalTxt] = useState('')
  const [error, setError] = useState<string | null>(null)

  const recognitionRef = useRef<SpeechRecognition | null>(null)

  useEffect(() => {
    if (typeof window === 'undefined') return
    const SpeechRecognitionImpl: any = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    if (!SpeechRecognitionImpl) return
    setSupported(true)
    const rec: SpeechRecognition = new SpeechRecognitionImpl()
    rec.lang = lang
    rec.interimResults = interim
    rec.continuous = continuous

    rec.onresult = (e: any) => {
      let interimAggregate = ''
      let finalAggregate = ''
      for (let i = 0; i < e.results.length; i++) {
        const res = e.results[i]
        const text = res[0].transcript
        if (res.isFinal) {
          finalAggregate += text
        } else if (i >= e.resultIndex) {
          // only include current interim block(s)
          interimAggregate += text
        }
      }
      if (finalAggregate) {
        setFinalTxt(prev => (prev + (prev && !prev.endsWith(' ') ? ' ' : '') + finalAggregate.trim()).trim())
      }
      setInterimTxt(interimAggregate)
    }
    rec.onerror = (e: any) => {
      setError(e.error || 'speech_error')
      setListening(false)
    }
    rec.onend = () => {
      setListening(false)
    }
    recognitionRef.current = rec
  }, [continuous, interim, lang])

  const start = useCallback(() => {
    if (!supported || !recognitionRef.current) return
    setError(null)
  setInterimTxt('')
  setFinalTxt('')
    try {
      recognitionRef.current.start()
      setListening(true)
    } catch (e) {
      // ignored (start called while already started)
    }
  }, [supported])

  const stop = useCallback(() => {
    recognitionRef.current?.stop()
  }, [])

  const reset = useCallback(() => {
  setInterimTxt('')
  setFinalTxt('')
  }, [])

  return { supported, listening, interim: interimTxt, final: finalTxt, error, start, stop, reset }
}
