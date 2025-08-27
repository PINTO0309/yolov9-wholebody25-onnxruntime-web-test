import { useRef, useState, useCallback } from 'react'

export const useScreenRecorder = () => {
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const recordedChunksRef = useRef<Blob[]>([])
  const [isRecording, setIsRecording] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showSourceSelector, setShowSourceSelector] = useState(false)

  const startRecordingWithSource = useCallback(async (sourceId?: string) => {
    try {
      setError(null)
      recordedChunksRef.current = []

      let stream: MediaStream

      // デバッグ情報
      console.log('=== Screen Recording Debug ===')
      console.log('window.electronAPI:', window.electronAPI)
      console.log('typeof window.electronAPI:', typeof window.electronAPI)
      console.log('Checking recording method...')

      // Electron環境の場合はdesktopCapturerを使用
      if (typeof window !== 'undefined' && window.electronAPI && typeof window.electronAPI.getSources === 'function') {
        console.log('Using Electron desktop capturer')
        const sources = await window.electronAPI.getSources()
        console.log('Available sources:', sources)
        
        if (sources.length === 0) {
          throw new Error('No screen sources available')
        }

        // sourceIdが指定されていない場合は、メイン画面を選択
        const targetSourceId = sourceId || (sources.find(s => s.name === 'Entire screen' || s.name.includes('Screen')) || sources[0]).id
        console.log('Selected source ID:', targetSourceId)
        
        stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            // @ts-ignore - Electron特有のconstraints
            mandatory: {
              chromeMediaSource: 'desktop',
              chromeMediaSourceId: targetSourceId,
              minWidth: 1280,
              maxWidth: 1920,
              minHeight: 720,
              maxHeight: 1080,
              frameRate: 30
            }
          }
        })
      } else {
        // Electron環境でもgetSourcesが使えない場合の簡易的な録画
        console.log('Using simplified screen capture')
        try {
          // Electron環境用の簡易的な画面キャプチャ
          stream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: {
              // @ts-ignore
              mandatory: {
                chromeMediaSource: 'screen',
                minWidth: 1280,
                maxWidth: 1920,
                minHeight: 720,
                maxHeight: 1080,
                frameRate: 30
              }
            }
          })
        } catch (fallbackError) {
          console.warn('Simplified capture failed, trying getDisplayMedia:', fallbackError)
          // 最終的なフォールバック
          stream = await navigator.mediaDevices.getDisplayMedia({
            video: {
              frameRate: 30,
              width: { ideal: 1920 },
              height: { ideal: 1080 }
            },
            audio: false
          })
        }
      }

      // MediaRecorderの設定
      const options: MediaRecorderOptions = {
        mimeType: 'video/webm;codecs=vp9',
        videoBitsPerSecond: 5000000 // 5 Mbps
      }

      // フォールバック: VP9がサポートされていない場合はVP8を使用
      if (!MediaRecorder.isTypeSupported(options.mimeType!)) {
        options.mimeType = 'video/webm;codecs=vp8'
        if (!MediaRecorder.isTypeSupported(options.mimeType!)) {
          options.mimeType = 'video/webm'
        }
      }

      const mediaRecorder = new MediaRecorder(stream, options)
      mediaRecorderRef.current = mediaRecorder

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordedChunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = () => {
        // 録画停止時にストリームのトラックを停止
        stream.getTracks().forEach(track => track.stop())
        
        // 録画データをダウンロード
        if (recordedChunksRef.current.length > 0) {
          const blob = new Blob(recordedChunksRef.current, { type: 'video/webm' })
          const url = URL.createObjectURL(blob)
          const a = document.createElement('a')
          a.href = url
          a.download = `recording-${new Date().toISOString().replace(/:/g, '-')}.webm`
          a.click()
          URL.revokeObjectURL(url)
        }
        
        setIsRecording(false)
      }

      mediaRecorder.onerror = (event: any) => {
        console.error('MediaRecorder error:', event)
        setError('Recording error occurred')
        setIsRecording(false)
      }

      // ユーザーが画面共有をキャンセルした場合の処理
      stream.getVideoTracks()[0].onended = () => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
          stopRecording()
        }
      }

      mediaRecorder.start(100) // 100msごとにデータを収集
      setIsRecording(true)
      console.log('Recording started with codec:', options.mimeType)
    } catch (err: any) {
      console.error('Failed to start recording:', err)
      if (err.name === 'NotAllowedError') {
        setError('Screen recording permission denied')
      } else {
        setError(`Failed to start recording: ${err.message}`)
      }
      setIsRecording(false)
    }
  }, [])

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop()
      console.log('Recording stopped')
    }
  }, [])

  const startRecording = useCallback(async () => {
    // Electron環境では直接録画を開始
    await startRecordingWithSource()
  }, [startRecordingWithSource])

  const toggleRecording = useCallback(() => {
    if (isRecording) {
      stopRecording()
    } else {
      startRecording()
    }
  }, [isRecording, startRecording, stopRecording])

  return {
    isRecording,
    error,
    startRecording,
    stopRecording,
    toggleRecording,
    showSourceSelector,
    setShowSourceSelector,
    startRecordingWithSource
  }
}