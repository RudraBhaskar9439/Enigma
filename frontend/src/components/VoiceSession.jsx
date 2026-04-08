import { useEffect, useRef, useState } from 'react'

const HUME_API_KEY = import.meta.env.VITE_HUME_API_KEY
const HUME_CONFIG_ID = import.meta.env.VITE_HUME_CONFIG_ID

export default function VoiceSession({ onBack }) {
  const configured = HUME_API_KEY && HUME_API_KEY !== 'your_hume_api_key_here'

  const [status, setStatus] = useState('idle') // idle | connecting | connected | speaking | error
  const [transcript, setTranscript] = useState([]) // [{role, text}]
  const [errorMsg, setErrorMsg] = useState('')

  const wsRef = useRef(null)
  const mediaStreamRef = useRef(null)
  const recorderRef = useRef(null)
  const audioQueueRef = useRef([])
  const playingRef = useRef(false)

  function log(...a) { console.log('[voice]', ...a) }

  async function start() {
    setErrorMsg('')
    setTranscript([])
    setStatus('connecting')

    // 1. Mic
    let stream
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, sampleRate: 48000 },
      })
      mediaStreamRef.current = stream
      log('mic granted')
    } catch (e) {
      setErrorMsg(`Mic denied: ${e.message}`)
      setStatus('error')
      return
    }

    // 2. WebSocket to Hume EVI
    const url = `wss://api.hume.ai/v0/evi/chat?api_key=${encodeURIComponent(HUME_API_KEY)}&config_id=${encodeURIComponent(HUME_CONFIG_ID)}`
    const ws = new WebSocket(url)
    ws.binaryType = 'arraybuffer'
    wsRef.current = ws

    ws.onopen = () => {
      log('ws open')
      setStatus('connected')

      // 3. Start MediaRecorder, send audio chunks as base64
      const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : ''
      const recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined)
      recorderRef.current = recorder

      recorder.ondataavailable = async (ev) => {
        if (ev.data.size === 0 || ws.readyState !== WebSocket.OPEN) return
        const buf = await ev.data.arrayBuffer()
        const b64 = arrayBufferToBase64(buf)
        ws.send(JSON.stringify({ type: 'audio_input', data: b64 }))
      }
      recorder.start(100) // chunk every 100ms
    }

    ws.onmessage = (ev) => {
      let msg
      try { msg = JSON.parse(ev.data) } catch { return }
      if (msg.type === 'audio_output') {
        const audioBuf = base64ToArrayBuffer(msg.data)
        audioQueueRef.current.push(audioBuf)
        playNext()
      } else if (msg.type === 'user_message') {
        setTranscript(t => [...t, { role: 'user', text: msg.message?.content || '' }])
      } else if (msg.type === 'assistant_message') {
        setTranscript(t => [...t, { role: 'bot', text: msg.message?.content || '' }])
      } else if (msg.type === 'error') {
        log('hume error', msg)
        setErrorMsg(msg.message || JSON.stringify(msg))
        setStatus('error')
      } else {
        log('msg', msg.type, msg)
      }
    }

    ws.onerror = (e) => {
      log('ws error', e)
      setErrorMsg('WebSocket error — check API key')
      setStatus('error')
    }

    ws.onclose = (e) => {
      log('ws close', e.code, e.reason)
      stop(false)
      if (status !== 'error') setStatus('idle')
    }
  }

  function playNext() {
    if (playingRef.current) return
    const buf = audioQueueRef.current.shift()
    if (!buf) return
    playingRef.current = true
    setStatus('speaking')
    const blob = new Blob([buf], { type: 'audio/wav' })
    const url = URL.createObjectURL(blob)
    const audio = new Audio(url)
    audio.onended = () => {
      URL.revokeObjectURL(url)
      playingRef.current = false
      if (audioQueueRef.current.length > 0) playNext()
      else setStatus('connected')
    }
    audio.onerror = () => {
      URL.revokeObjectURL(url)
      playingRef.current = false
      if (audioQueueRef.current.length > 0) playNext()
      else setStatus('connected')
    }
    audio.play().catch(err => {
      log('audio play err', err)
      playingRef.current = false
    })
  }

  function stop(closeWs = true) {
    try { recorderRef.current?.state !== 'inactive' && recorderRef.current?.stop() } catch {}
    recorderRef.current = null
    try { mediaStreamRef.current?.getTracks().forEach(t => t.stop()) } catch {}
    mediaStreamRef.current = null
    if (closeWs && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.close()
    }
    audioQueueRef.current = []
    playingRef.current = false
    if (closeWs) setStatus('idle')
  }

  useEffect(() => () => stop(true), [])

  const orbClass =
    status === 'idle' || status === 'error' ? 'idle'
    : status === 'speaking' ? 'speaking'
    : ''

  const statusLabel = {
    idle: 'Tap start to begin a voice session',
    connecting: 'Connecting…',
    connected: 'Listening… speak naturally',
    speaking: 'Vishwamitra is speaking…',
    error: errorMsg || 'Connection error',
  }[status]

  return (
    <section className="session">
      <div className="session-header">
        <button className="back" onClick={onBack}>← Back</button>
        <div className="session-title">Voice Conversation</div>
        <div style={{ width: 60 }} />
      </div>

      {!configured ? (
        <div className="voice">
          <div className="voice-orb idle" />
          <div className="voice-status">
            Add <code>VITE_HUME_API_KEY</code> and <code>VITE_HUME_CONFIG_ID</code> to <code>frontend/.env</code> and restart dev server.
          </div>
        </div>
      ) : (
        <div className="voice">
          <div className={`voice-orb ${orbClass}`} />
          <div className="voice-status">{statusLabel}</div>
          <div className="voice-controls">
            {status === 'idle' || status === 'error' ? (
              <button className="btn btn-primary" onClick={start}>Start voice session</button>
            ) : (
              <button className="btn btn-secondary" onClick={() => stop(true)}>End session</button>
            )}
          </div>

          {transcript.length > 0 && (
            <div style={{ marginTop: 32, textAlign: 'left', maxHeight: 280, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 8 }}>
              {transcript.map((m, i) => (
                <div key={i} className={`bubble ${m.role}`}>{m.text}</div>
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  )
}

function arrayBufferToBase64(buf) {
  let binary = ''
  const bytes = new Uint8Array(buf)
  const chunk = 0x8000
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk))
  }
  return btoa(binary)
}

function base64ToArrayBuffer(b64) {
  const binary = atob(b64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  return bytes.buffer
}
