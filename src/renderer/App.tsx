import { useState, useEffect, useRef, useCallback } from 'react'
import { YOLOv9Detector } from './utils/yolov9'
import { PersonSegmentation, SegmentationConfig } from './utils/segmentation'
import { BoundingBox, ExecutionProvider, ModelConfig } from './utils/types'
import { useWebcam } from './hooks/useWebcam'
import { useScreenRecorder } from './hooks/useScreenRecorder'
import DetectionCanvas from './components/DetectionCanvas'
import GPUSelector from './components/GPUSelector'
import { checkWebGPUSupport, getGPUCount } from './utils/webgpu-check'
import './App.css'

function App() {
  const [executionProvider, setExecutionProvider] = useState<ExecutionProvider>('webgl')
  const [isModelLoading, setIsModelLoading] = useState(false)
  const [isModelLoaded, setIsModelLoaded] = useState(false)
  const [detections, setDetections] = useState<BoundingBox[]>([])
  const [segmentationMask, setSegmentationMask] = useState<Uint8Array | null>(null)
  const [inferenceTime, setInferenceTime] = useState(0)
  const [segmentationTime, setSegmentationTime] = useState(0)
  const [isDetecting, setIsDetecting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [initializationStatus, setInitializationStatus] = useState<string>('')
  const [webGPUSupported, setWebGPUSupported] = useState(false)
  const [debugMode, setDebugMode] = useState(false)
  const [gpuCount, setGpuCount] = useState(0)
  const [yoloGpuIndex, setYoloGpuIndex] = useState<number>(0)
  const [segmentationGpuIndex, setSegmentationGpuIndex] = useState<number>(0)
  const [actualYoloProvider, setActualYoloProvider] = useState<string>('')
  const [actualSegProvider, setActualSegProvider] = useState<string>('')

  const detectorRef = useRef<YOLOv9Detector | null>(null)
  const segmentationRef = useRef<PersonSegmentation | null>(null)
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

      // 既存のdetectorとsegmentationの破棄
      if (detectorRef.current) {
        try {
          await detectorRef.current.dispose()
          detectorRef.current = null
        } catch (disposeError) {
          console.warn('Error disposing previous detector:', disposeError)
        }
      }

      if (segmentationRef.current) {
        try {
          await segmentationRef.current.dispose()
          segmentationRef.current = null
        } catch (disposeError) {
          console.warn('Error disposing previous segmentation:', disposeError)
        }
      }

      // Select model files based on execution provider
      // Use non-quantized models for WebGPU/WebGL for better performance
      const useQuantized = executionProvider !== 'webgpu' && executionProvider !== 'webgl'
      const modelSuffix = useQuantized ? '_quant' : ''

      const yoloConfig: ModelConfig = {
        modelPath: `/models/yolov9_s_wholebody25_0100_1x3x640x640${modelSuffix}.onnx`,
        inputShape: [1, 3, 640, 640],
        confidenceThreshold: 0.3,
        iouThreshold: 0.45,
        executionProvider
      }

      const segmentationConfig: SegmentationConfig = {
        modelPath: `/models/peopleseg_b0_0.8741_1x3x640x640${modelSuffix}.onnx`,
        inputShape: [1, 3, 640, 640],
        threshold: 0.5,
        executionProvider
      }

      console.log(`Using ${useQuantized ? 'quantized' : 'non-quantized'} models for ${executionProvider}`)

      const detector = new YOLOv9Detector(yoloConfig, handleStatusUpdate)
      const segmentation = new PersonSegmentation(segmentationConfig, handleStatusUpdate)

      await detector.initialize(yoloGpuIndex)
      await segmentation.initialize(segmentationGpuIndex)

      detectorRef.current = detector
      segmentationRef.current = segmentation

      // Get actual providers that were successfully initialized
      setActualYoloProvider(detector.getActiveProvider())
      setActualSegProvider(segmentation.getActiveProvider())

      setIsModelLoaded(true)

      if (!offscreenCanvasRef.current) {
        offscreenCanvasRef.current = new OffscreenCanvas(640, 480)
      }
    } catch (err: any) {
      console.error('Failed to load models:', err)
      const errorMessage = err?.message || String(err)
      setError(`Failed to load models: ${errorMessage}`)
      setIsModelLoaded(false)
      detectorRef.current = null
      segmentationRef.current = null
      setActualYoloProvider('')
      setActualSegProvider('')
    } finally {
      setIsModelLoading(false)
    }
  }, [executionProvider, handleStatusUpdate, yoloGpuIndex, segmentationGpuIndex])

  useEffect(() => {
    // WebGPUサポートチェック
    checkWebGPUSupport().then(async supported => {
      setWebGPUSupported(supported)
      if (!supported && executionProvider === 'webgpu') {
        console.warn('WebGPU not supported, falling back to WebGL')
        setExecutionProvider('webgl')
      } else if (supported) {
        // GPU数を取得
        const count = await getGPUCount()
        setGpuCount(count)
        console.log(`=== Multi-GPU Detection ===`)
        console.log(`GPU count detected: ${count}`)

        // WebGPU API制限の警告
        if (count === 1) {
          console.warn('Only 1 GPU detected. This may be due to WebGPU API limitations.')
          console.warn('For multi-GPU support in Chrome/Edge, try:')
          console.warn('1. Enable chrome://flags/#enable-webgpu-developer-features')
          console.warn('2. Launch with --enable-unsafe-webgpu flag')
        }

        // マルチGPU環境では異なるGPUを初期設定
        if (count > 1) {
          console.log('Multiple GPUs detected, enabling GPU selection')
          setSegmentationGpuIndex(1)
        } else {
          console.log('Single GPU mode, GPU selection will be disabled')
        }
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
      if (segmentationRef.current) {
        const segmentation = segmentationRef.current
        segmentationRef.current = null
        segmentation.dispose().catch(error => {
          console.warn('Error during cleanup segmentation dispose:', error)
        })
      }
    }
  }, [isReady, initializeModel])

  const startDetection = useCallback(() => {
    if (!isModelLoaded || !videoRef.current || !detectorRef.current || !segmentationRef.current) return

    setIsDetecting(true)

    const detect = async () => {
      // 停止フラグをチェック
      if (!animationIdRef.current) return

      if (!videoRef.current || !detectorRef.current || !segmentationRef.current || !offscreenCanvasRef.current) return

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
        // Run YOLO detection
        const yoloStartTime = performance.now()
        const boxes = await detectorRef.current.detect(imageData)
        const yoloEndTime = performance.now()

        // Run segmentation
        const segStartTime = performance.now()
        const mask = await segmentationRef.current.segment(imageData)
        const segEndTime = performance.now()

        setInferenceTime(yoloEndTime - yoloStartTime)
        setSegmentationTime(segEndTime - segStartTime)
        setDetections(boxes)
        setSegmentationMask(mask)
      } catch (err) {
        console.error('Detection/Segmentation error:', err)
      }

      // 停止フラグを再度チェック
      if (animationIdRef.current) {
        animationIdRef.current = requestAnimationFrame(detect)
      }
    }

    // アニメーションIDを初期化して開始
    animationIdRef.current = 1 // nullでない値を設定
    detect()
  }, [isModelLoaded, videoRef])

  const stopDetection = useCallback(() => {
    console.log('Stopping detection...')
    setIsDetecting(false)
    if (animationIdRef.current) {
      cancelAnimationFrame(animationIdRef.current)
      animationIdRef.current = null
    }
    setDetections([])
    setSegmentationMask(null)
    console.log('Detection stopped')
  }, [])

  const toggleProvider = useCallback(async () => {
    const newProvider = executionProvider === 'webgl' ? 'webgpu' : 'webgl'

    // 検出を停止
    stopDetection()

    // モデルをアンロード
    setIsModelLoaded(false)

    // Clear actual provider states
    setActualYoloProvider('')
    setActualSegProvider('')

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
            segmentationMask={segmentationMask}
          />
        )}
      </div>

      {webGPUSupported && executionProvider === 'webgpu' && (
        <div className="gpu-controls">
          <GPUSelector
            label="YOLO GPU:"
            onSelectGPU={setYoloGpuIndex}
            disabled={isModelLoading || isDetecting || gpuCount <= 1}
            singleGpuMode={gpuCount <= 1}
          />
          <GPUSelector
            label="Seg GPU:"
            onSelectGPU={setSegmentationGpuIndex}
            disabled={isModelLoading || isDetecting || gpuCount <= 1}
            singleGpuMode={gpuCount <= 1}
          />
        </div>
      )}

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
          <span className="info-label">YOLO Runtime:</span>
          <span className="info-value">
            {actualYoloProvider ? actualYoloProvider.toUpperCase() : executionProvider.toUpperCase()}
          </span>
        </div>
        <div className="info-item">
          <span className="info-label">Seg Runtime:</span>
          <span className="info-value">
            {actualSegProvider ? actualSegProvider.toUpperCase() : executionProvider.toUpperCase()}
          </span>
        </div>
        <div className="info-item">
          <span className="info-label">YOLO:</span>
          <span className="info-value">{inferenceTime.toFixed(2)} ms</span>
        </div>
        <div className="info-item">
          <span className="info-label">Segmentation:</span>
          <span className="info-value">{segmentationTime.toFixed(2)} ms</span>
        </div>
        <div className="info-item">
          <span className="info-label">FPS:</span>
          <span className="info-value">
            {(inferenceTime + segmentationTime) > 0 ? (1000 / (inferenceTime + segmentationTime)).toFixed(1) : '0'}
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