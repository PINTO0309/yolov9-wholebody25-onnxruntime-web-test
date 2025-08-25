import React, { useState, useEffect, useRef, useCallback } from 'react'
import { YOLOv9Detector } from './utils/yolov9'
import { BoundingBox, ExecutionProvider, ModelConfig } from './utils/types'
import { useWebcam } from './hooks/useWebcam'
import { useScreenRecorder } from './hooks/useScreenRecorder'
import DetectionCanvas from './components/DetectionCanvas'
import { checkWebGPUSupport } from './utils/webgpu-check'
import './App.css'

function App() {
  const [executionProvider, setExecutionProvider] = useState<ExecutionProvider>('webgl')
  const [isModelLoading, setIsModelLoading] = useState(false)
  const [isModelLoaded, setIsModelLoaded] = useState(false)
  const [detections, setDetections] = useState<BoundingBox[]>([])
  const [inferenceTime, setInferenceTime] = useState(0)
  const [isDetecting, setIsDetecting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [initializationStatus, setInitializationStatus] = useState<string>('')
  const [webGPUSupported, setWebGPUSupported] = useState(false)
  const [debugMode, setDebugMode] = useState(false)
  
  const detectorRef = useRef<YOLOv9Detector | null>(null)
  const animationIdRef = useRef<number | null>(null)
  const offscreenCanvasRef = useRef<OffscreenCanvas | null>(null)
  
  const { videoRef, isReady, error: webcamError } = useWebcam(640, 480)
  const { isRecording, error: recordError, toggleRecording } = useScreenRecorder()

  const handleStatusUpdate = useCallback((status: string) => {
    setInitializationStatus(status)
  }, [])

  const initializeModel = useCallback(async () => {
    try {
      setIsModelLoading(true)
      setError(null)
      setInitializationStatus('')
      
      // 既存のdetectorの破棄
      if (detectorRef.current) {
        try {
          await detectorRef.current.dispose()
          detectorRef.current = null
        } catch (disposeError) {
          console.warn('Error disposing previous detector:', disposeError)
        }
      }

      const config: ModelConfig = {
        modelPath: '/models/yolov9_s_wholebody25_0100_1x3x640x640.onnx',
        inputShape: [1, 3, 640, 640],
        confidenceThreshold: 0.5,
        iouThreshold: 0.45,
        executionProvider
      }

      const detector = new YOLOv9Detector(config, handleStatusUpdate)
      await detector.initialize()
      
      detectorRef.current = detector
      setIsModelLoaded(true)
      
      if (!offscreenCanvasRef.current) {
        offscreenCanvasRef.current = new OffscreenCanvas(640, 480)
      }
    } catch (err: any) {
      console.error('Failed to load model:', err)
      const errorMessage = err?.message || String(err)
      setError(`Failed to load model: ${errorMessage}`)
      setIsModelLoaded(false)
      detectorRef.current = null
    } finally {
      setIsModelLoading(false)
    }
  }, [executionProvider, handleStatusUpdate])

  useEffect(() => {
    // WebGPUサポートチェック
    checkWebGPUSupport().then(supported => {
      setWebGPUSupported(supported)
      if (!supported && executionProvider === 'webgpu') {
        console.warn('WebGPU not supported, falling back to WebGL')
        setExecutionProvider('webgl')
      }
    })
    
    // Electron APIの確認（デバッグ用）
    console.log('=== Checking Electron API availability ===')
    console.log('window.electronAPI:', window.electronAPI)
    console.log('typeof window.electronAPI:', typeof window.electronAPI)
    if (window.electronAPI) {
      console.log('electronAPI.getSources:', typeof window.electronAPI.getSources)
      console.log('electronAPI.platform:', window.electronAPI.platform)
    } else {
      console.error('window.electronAPI is undefined!')
      // Try to access it directly to see if it's a timing issue
      setTimeout(() => {
        console.log('After timeout - window.electronAPI:', window.electronAPI)
      }, 1000)
    }
  }, [])

  useEffect(() => {
    if (isReady) {
      initializeModel()
    }

    return () => {
      // クリーンアップ処理
      if (detectorRef.current) {
        const detector = detectorRef.current
        detectorRef.current = null
        // 非同期でdisposeを実行（エラーは無視）
        detector.dispose().catch(error => {
          console.warn('Error during cleanup dispose:', error)
        })
      }
    }
  }, [isReady, initializeModel])

  const startDetection = useCallback(() => {
    if (!isModelLoaded || !videoRef.current || !detectorRef.current) return
    
    setIsDetecting(true)
    
    const detect = async () => {
      if (!videoRef.current || !detectorRef.current || !offscreenCanvasRef.current) return
      
      const video = videoRef.current
      const offscreenCanvas = offscreenCanvasRef.current
      const ctx = offscreenCanvas.getContext('2d')
      
      if (!ctx || video.readyState !== video.HAVE_ENOUGH_DATA) {
        animationIdRef.current = requestAnimationFrame(detect)
        return
      }
      
      ctx.drawImage(video, 0, 0, 640, 480)
      const imageData = ctx.getImageData(0, 0, 640, 480)
      
      try {
        const startTime = performance.now()
        const boxes = await detectorRef.current.detect(imageData)
        const endTime = performance.now()
        
        setInferenceTime(endTime - startTime)
        setDetections(boxes)
      } catch (err) {
        console.error('Detection error:', err)
      }
      
      animationIdRef.current = requestAnimationFrame(detect)
    }
    
    detect()
  }, [isModelLoaded, videoRef])

  const stopDetection = useCallback(() => {
    setIsDetecting(false)
    if (animationIdRef.current) {
      cancelAnimationFrame(animationIdRef.current)
      animationIdRef.current = null
    }
    setDetections([])
  }, [])

  const toggleProvider = useCallback(async () => {
    const newProvider = executionProvider === 'webgl' ? 'webgpu' : 'webgl'
    
    // 検出を停止
    stopDetection()
    
    // モデルをアンロード
    setIsModelLoaded(false)
    
    // プロバイダーを切り替え（これによりuseEffectがトリガーされ、新しいモデルが初期化される）
    setExecutionProvider(newProvider)
  }, [executionProvider, stopDetection])

  const toggleDebug = useCallback(async () => {
    const newDebugMode = !debugMode
    setDebugMode(newDebugMode)
    
    // Toggle DevTools via Electron API
    if (window.electronAPI && window.electronAPI.toggleDevTools) {
      await window.electronAPI.toggleDevTools(newDebugMode)
    }
  }, [debugMode])

  return (
    <div className="app">
      <div className="video-container">
        <video
          ref={videoRef}
          width={640}
          height={480}
          style={{ display: 'none' }}
        />
        {isReady && (
          <DetectionCanvas
            width={640}
            height={480}
            detections={detections}
            videoRef={videoRef}
          />
        )}
      </div>
      
      <div className="controls">
        <button
          onClick={toggleProvider}
          disabled={isModelLoading || isDetecting || (!webGPUSupported && executionProvider === 'webgl')}
          className="btn"
          title={!webGPUSupported && executionProvider === 'webgl' ? 'WebGPU is not supported on this system' : ''}
        >
          Switch to {executionProvider === 'webgl' ? 'WebGPU' : 'WebGL'}
          {!webGPUSupported && executionProvider === 'webgl' && ' (Not Supported)'}
        </button>
        
        <button
          onClick={isDetecting ? stopDetection : startDetection}
          disabled={!isModelLoaded || isModelLoading}
          className="btn btn-primary"
        >
          {isDetecting ? 'Stop Detection' : 'Start Detection'}
        </button>
        
        <button
          onClick={toggleRecording}
          className={`btn ${isRecording ? 'btn-recording' : ''}`}
          title={isRecording ? 'Stop recording' : 'Start recording screen'}
        >
          {isRecording ? '⏹ Stop Recording' : '⏺ Record Screen'}
        </button>
        
        <button
          onClick={toggleDebug}
          className={`btn ${debugMode ? 'btn-debug-active' : ''}`}
          title={debugMode ? 'Debug mode is ON' : 'Debug mode is OFF'}
        >
          {debugMode ? '🐛 Debug ON' : '🐛 Debug OFF'}
        </button>
      </div>
      
      <div className="info-panel">
        <div className="info-item">
          <span className="info-label">Runtime:</span>
          <span className="info-value">{executionProvider.toUpperCase()}</span>
        </div>
        <div className="info-item">
          <span className="info-label">Inference:</span>
          <span className="info-value">{inferenceTime.toFixed(2)} ms</span>
        </div>
        <div className="info-item">
          <span className="info-label">FPS:</span>
          <span className="info-value">
            {inferenceTime > 0 ? (1000 / inferenceTime).toFixed(1) : '0'}
          </span>
        </div>
        <div className="info-item">
          <span className="info-label">Detections:</span>
          <span className="info-value">{detections.length}</span>
        </div>
      </div>
      
      {isModelLoading && (
        <div className="loading-overlay">
          <div className="loading-spinner"></div>
          <p>{initializationStatus || 'Loading model...'}</p>
        </div>
      )}
      
      {initializationStatus && !isModelLoading && (
        <div className="status-message">
          {initializationStatus}
        </div>
      )}
      
      {(error || webcamError || recordError) && (
        <div className="error-message">
          {error || webcamError || recordError}
        </div>
      )}
      
      {isRecording && (
        <div className="recording-indicator">
          <span className="recording-dot"></span>
          Recording...
        </div>
      )}
    </div>
  )
}

export default App