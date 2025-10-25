"use client";

import { useCallback, useEffect, useRef, useState } from "react";

interface SpeakOptions {
  rate?: number; // 0.1 - 10
  pitch?: number; // 0 - 2
  volume?: number; // 0 - 1
  voiceURI?: string; // match a voice
  lang?: string; // BCP-47
  onend?: () => void;
  onerror?: (e: SpeechSynthesisErrorEvent) => void;
}

interface UseTextToSpeech {
  supported: boolean;
  speaking: boolean;
  pending: boolean;
  paused: boolean;
  voices: SpeechSynthesisVoice[];
  speak: (text: string, opts?: SpeakOptions) => void;
  cancel: () => void;
  pause: () => void;
  resume: () => void;
}

export function useTextToSpeech(): UseTextToSpeech {
  const [supported] = useState(typeof window !== "undefined" && "speechSynthesis" in window);
  const [speaking, setSpeaking] = useState(false);
  const [pending, setPending] = useState(false);
  const [paused, setPaused] = useState(false);
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

  // Load voices (may populate asynchronously in some browsers)
  useEffect(() => {
    if (!supported) return;
    function loadVoices() {
      const list = window.speechSynthesis.getVoices();
      if (list.length) setVoices(list);
    }
    loadVoices();
    window.speechSynthesis.addEventListener("voiceschanged", loadVoices);
    return () => window.speechSynthesis.removeEventListener("voiceschanged", loadVoices);
  }, [supported]);

  const resetFlags = () => {
    setSpeaking(false);
    setPending(false);
    setPaused(false);
  };

  const cancel = useCallback(() => {
    if (!supported) return;
    window.speechSynthesis.cancel();
    resetFlags();
  }, [supported]);

  const speak = useCallback((text: string, opts?: SpeakOptions) => {
    if (!supported) return;
    if (!text || !text.trim()) return;
    // If already speaking, cancel previous
    if (window.speechSynthesis.speaking || window.speechSynthesis.pending) {
      window.speechSynthesis.cancel();
    }

    const utter = new SpeechSynthesisUtterance(text);
    utter.rate = opts?.rate ?? 1;
    utter.pitch = opts?.pitch ?? 1;
    utter.volume = opts?.volume ?? 1;
    if (opts?.lang) utter.lang = opts.lang;

    if (opts?.voiceURI) {
      const v = voices.find(vo => vo.voiceURI === opts.voiceURI);
      if (v) utter.voice = v;
    }

    utter.onstart = () => {
      setSpeaking(true);
      setPending(false);
      setPaused(false);
    };
    utter.onend = () => {
      resetFlags();
      opts?.onend?.();
    };
    utter.onerror = (e) => {
      resetFlags();
      opts?.onerror?.(e as SpeechSynthesisErrorEvent);
    };

    utteranceRef.current = utter;
    setPending(true);
    window.speechSynthesis.speak(utter);
  }, [supported, voices]);

  const pause = useCallback(() => {
    if (!supported) return;
    if (window.speechSynthesis.speaking && !window.speechSynthesis.paused) {
      window.speechSynthesis.pause();
      setPaused(true);
    }
  }, [supported]);

  const resume = useCallback(() => {
    if (!supported) return;
    if (window.speechSynthesis.paused) {
      window.speechSynthesis.resume();
      setPaused(false);
    }
  }, [supported]);

  // Keep flags in sync (in case of external interruptions)
  useEffect(() => {
    if (!supported) return;
    const interval = setInterval(() => {
      const synth = window.speechSynthesis;
      setSpeaking(synth.speaking);
      setPending(synth.pending);
      setPaused(synth.paused);
    }, 1000);
    return () => clearInterval(interval);
  }, [supported]);

  return { supported, speaking, pending, paused, voices, speak, cancel, pause, resume };
}
