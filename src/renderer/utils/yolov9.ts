import * as ort from 'onnxruntime-web'
import { BoundingBox, ModelConfig } from './types'
import { nonMaxSuppression } from './nms'
import { selectGPUAdapterByIndex, getAllAvailableAdapters } from './webgpu-check'

// ONNX Runtime Webの環境設定
ort.env.wasm.wasmPaths = 'http://localhost:5173/'
ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4
ort.env.wasm.simd = true

export class YOLOv9Detector {
  private session: ort.InferenceSession | null = null
  private config: ModelConfig
  private labels: string[] = [
    'Body', 'Adult', 'Child', 'Male', 'Female',
    'Body_with_Wheelchair', 'Body_with_Crutches', 'Head', 'Front', 'Right_Front',
    'Right_Side', 'Right_Back', 'Back', 'Left_Back', 'Left_Side',
    'Left_Front', 'Face', 'Eye', 'Nose', 'Mouth',
    'Ear', 'Hand', 'Hand_Left', 'Hand_Right', 'Foot'
  ]
  // 描画しないクラスIDのリスト
  private excludedClassIds: Set<number> = new Set([
    1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 22, 23
  ])
  private onStatusUpdate?: (status: string) => void

  constructor(config: ModelConfig, onStatusUpdate?: (status: string) => void) {
    this.config = config
    this.onStatusUpdate = onStatusUpdate
  }

  private updateStatus(status: string) {
    if (this.onStatusUpdate) {
      this.onStatusUpdate(status)
    }
    console.log('Status:', status)
  }

  private toTyped(data: number[] | Float32Array, dtype: "float32" | "int32" = "float32") {
    if (ArrayBuffer.isView(data)) return data as ArrayBufferView;
    return dtype === "float32" ? new Float32Array(data as number[]) : new Int32Array(data as number[]);
  }

  async initialize(gpuIndex?: number): Promise<void> {
    this.updateStatus('Initializing ONNX Runtime...')

    // Try providers in order: WebNN → WebGPU → WebGL
    const providers = [
      { name: 'webnn', providers: ['webnn', 'wasm'] },
      // { name: 'webgpu', providers: ['webgpu', 'wasm'] },
      // { name: 'webgl', providers: ['webgl', 'wasm'] }
    ]

    let sessionCreated = false
    let lastError: any = null

    for (const provider of providers) {
      try {
        this.updateStatus(`Trying ${provider.name.toUpperCase()} provider...`)
        console.log(`Attempting to initialize with ${provider.name}...`)

        // WebGPU使用時はGPU選択を行う
        if (provider.name === 'webgpu') {
          this.updateStatus('Selecting GPU adapter...')
          const availableAdapters = await getAllAvailableAdapters()
          console.log('Available GPU adapters:', availableAdapters.map(a => a.description))

          const selectedAdapter = await selectGPUAdapterByIndex(gpuIndex ?? 0)
          if (selectedAdapter) {
            const info = await selectedAdapter.requestAdapterInfo()
            console.log('Selected GPU for YOLO:', info.description || 'Unknown GPU')
            this.updateStatus(`Using GPU for YOLO: ${info.description || 'Unknown GPU'}`)
          } else {
            console.warn('No suitable GPU adapter found for YOLO')
            throw new Error('No suitable GPU adapter found')
          }
        }

        const options: ort.InferenceSession.SessionOptions = {
          executionProviders: provider.providers,
          graphOptimizationLevel: 'all'
        }

        this.updateStatus(`Loading model with ${provider.name}...`)

        if (provider.name === 'webgpu' || provider.name === 'webgl') {
          this.updateStatus('Compiling shaders...')
        }

        this.session = await ort.InferenceSession.create(this.config.modelPath, options)

        // Update config to reflect the successfully used provider
        this.config.executionProvider = provider.name as 'webgl' | 'webgpu' | 'wasm'

        this.updateStatus(`Model loaded successfully with ${provider.name.toUpperCase()}`)
        console.log(`Successfully initialized with ${provider.name}`)
        console.log('Input names:', this.session.inputNames)
        console.log('Output names:', this.session.outputNames)
        console.log('Execution Provider:', provider.name)

        sessionCreated = true

        setTimeout(() => {
          this.updateStatus('')
        }, 2000)

        break // Success, exit the loop

      } catch (error) {
        console.warn(`Failed to initialize with ${provider.name}:`, error)
        lastError = error

        // Clean up failed session if any
        if (this.session) {
          try {
            await this.session.release()
          } catch (e) {
            console.warn('Error releasing failed session:', e)
          }
          this.session = null
        }
      }
    }

    if (!sessionCreated) {
      this.updateStatus('Failed to initialize model with any provider')
      console.error('Failed to initialize model with any provider. Last error:', lastError)
      throw lastError || new Error('Failed to initialize model with any provider')
    }
  }

  async detect(imageData: ImageData): Promise<BoundingBox[]> {
    if (!this.session) {
      throw new Error('Model not initialized')
    }

    const preprocessed = this.preprocessImage(imageData)

    let output: ort.Tensor
    const startTime = performance.now()

    if (this.config.executionProvider === 'webgpu' || this.config.executionProvider === 'webnn') {
      // WebGPU/WebNN最適化: preferredLocationsを使用してGPU上でテンソルを作成
      try {
        // const input = new ort.Tensor('float32', preprocessed, this.config.inputShape)
        const feeds: Record<string, ort.Tensor> = {
          images: new ort.Tensor("float32", preprocessed, this.config.inputShape),
        };

        // WebGPU/WebNN最適化オプション
        const runOptions: ort.InferenceSession.RunOptions = {
          logSeverityLevel: 3,
          logVerbosityLevel: 0
        }

        const outputs = await this.session.run(feeds, runOptions)
        output = outputs['output0'] || outputs[this.session.outputNames[0]]
      } catch (gpuError) {
        console.warn(`${this.config.executionProvider.toUpperCase()} execution failed, falling back to normal mode:`, gpuError)
        const input = new ort.Tensor('float32', preprocessed, this.config.inputShape)
        const outputs = await this.session.run({ images: input })
        output = outputs['output0'] || outputs[this.session.outputNames[0]]
      }
    } else {
      // WebGL/WASM実行
      const input = new ort.Tensor('float32', preprocessed, this.config.inputShape)
      const outputs = await this.session.run({ images: input })
      output = outputs['output0'] || outputs[this.session.outputNames[0]]
    }

    const inferenceTime = performance.now() - startTime
    console.log(`Inference time: ${inferenceTime.toFixed(2)}ms (Provider: ${this.config.executionProvider})`)

    const boxes = this.postprocess(output as ort.Tensor, imageData.width, imageData.height)

    return boxes
  }

  private preprocessImage(imageData: ImageData): Float32Array {
    const { width, height } = imageData
    const [, channels, modelHeight, modelWidth] = this.config.inputShape
    const preprocessed = new Float32Array(channels * modelHeight * modelWidth)

    const canvas = new OffscreenCanvas(modelWidth, modelHeight)
    const ctx = canvas.getContext('2d')!

    const tempCanvas = new OffscreenCanvas(width, height)
    const tempCtx = tempCanvas.getContext('2d')!
    tempCtx.putImageData(imageData, 0, 0)

    // 640x480 -> 640x640への変換: 上下にパディングを追加
    // padY = (640 - 480) / 2 = 80
    const padX = 0
    const padY = (modelHeight - height) / 2

    ctx.fillStyle = '#000000'
    ctx.fillRect(0, 0, modelWidth, modelHeight)
    ctx.drawImage(tempCanvas, padX, padY, width, height)

    const resizedData = ctx.getImageData(0, 0, modelWidth, modelHeight).data

    for (let i = 0; i < modelHeight * modelWidth; i++) {
      preprocessed[i] = resizedData[i * 4] / 255.0
      preprocessed[i + modelHeight * modelWidth] = resizedData[i * 4 + 1] / 255.0
      preprocessed[i + 2 * modelHeight * modelWidth] = resizedData[i * 4 + 2] / 255.0
    }

    return preprocessed
  }

  private postprocess(
    output: ort.Tensor,
    originalWidth: number,
    originalHeight: number
  ): BoundingBox[] {
    const outputData = output.data as Float32Array
    const dims = output.dims
    console.log('Output tensor dims:', dims)

    const boxes: BoundingBox[] = []
    const modelHeight = this.config.inputShape[2]

    // 640x480 -> 640x640への変換で使用したパディング値
    const padX = 0
    const padY = (modelHeight - originalHeight) / 2

    // YOLOv9の出力フォーマットを確認
    // 通常は [1, num_classes + 4, num_boxes] または [1, num_boxes, num_classes + 4]
    let numBoxes: number
    let numClasses: number
    let isTransposed = false

    if (dims.length === 3) {
      if (dims[1] > dims[2]) {
        // [1, num_boxes, num_classes + 4]
        numBoxes = dims[1]
        numClasses = dims[2] - 4
        isTransposed = false
      } else {
        // [1, num_classes + 4, num_boxes] - 転置が必要
        numBoxes = dims[2]
        numClasses = dims[1] - 4
        isTransposed = true
      }
    } else {
      console.error('Unexpected output shape:', dims)
      return []
    }

    console.log(`Processing ${numBoxes} boxes with ${numClasses} classes (transposed: ${isTransposed})`)

    for (let i = 0; i < numBoxes; i++) {
      let cx, cy, w, h
      let classScores: number[] = []

      if (isTransposed) {
        // [1, num_classes + 4, num_boxes] format
        cx = outputData[0 * numBoxes + i]
        cy = outputData[1 * numBoxes + i]
        w = outputData[2 * numBoxes + i]
        h = outputData[3 * numBoxes + i]

        for (let c = 0; c < numClasses; c++) {
          classScores.push(outputData[(4 + c) * numBoxes + i])
        }
      } else {
        // [1, num_boxes, num_classes + 4] format
        const baseIdx = i * (numClasses + 4)
        cx = outputData[baseIdx]
        cy = outputData[baseIdx + 1]
        w = outputData[baseIdx + 2]
        h = outputData[baseIdx + 3]

        for (let c = 0; c < numClasses; c++) {
          classScores.push(outputData[baseIdx + 4 + c])
        }
      }

      // 最大信頼度とクラスを見つける
      let maxConfidence = 0
      let bestClass = -1
      for (let c = 0; c < classScores.length; c++) {
        if (classScores[c] > maxConfidence) {
          maxConfidence = classScores[c]
          bestClass = c
        }
      }

      if (maxConfidence > this.config.confidenceThreshold && !this.excludedClassIds.has(bestClass)) {
        // 除外リストに含まれていないクラスのみ処理
        // モデル座標系(640x640ピクセル)から元の画像座標系(640x480)への変換
        // モデルは640x640のピクセル座標を直接返す
        const x1 = (cx - w / 2) - padX
        const y1 = (cy - h / 2) - padY
        const x2 = (cx + w / 2) - padX
        const y2 = (cy + h / 2) - padY

        boxes.push({
          x1: Math.max(0, x1),
          y1: Math.max(0, y1),
          x2: Math.min(originalWidth, x2),
          y2: Math.min(originalHeight, y2),
          confidence: maxConfidence,
          class: bestClass,
          label: this.labels[bestClass] || `class_${bestClass}`
        })
      }
    }

    return nonMaxSuppression(boxes, this.config.iouThreshold)
  }

  getInferenceTime(): number {
    return 0
  }

  getActiveProvider(): string {
    return this.config.executionProvider
  }

  async dispose(): Promise<void> {
    // セッションのクリーンアップ
    if (this.session) {
      try {
        const tempSession = this.session
        this.session = null  // 先にnullに設定して重複解放を防ぐ
        await tempSession.release()
        console.log('Session released successfully')
      } catch (error) {
        console.warn('Error releasing session (may already be released):', error)
        // セッションが既に解放されている場合はエラーを無視
      }
    }
  }
}