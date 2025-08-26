import * as ort from 'onnxruntime-web'

export interface SegmentationConfig {
  modelPath: string
  inputShape: number[]
  threshold: number
  executionProvider: 'webgl' | 'webgpu' | 'wasm'
}

export class PersonSegmentation {
  private session: ort.InferenceSession | null = null
  private config: SegmentationConfig
  private initializationStatus: string = ''
  private onStatusUpdate?: (status: string) => void

  constructor(config: SegmentationConfig, onStatusUpdate?: (status: string) => void) {
    this.config = config
    this.onStatusUpdate = onStatusUpdate
  }

  private updateStatus(status: string) {
    this.initializationStatus = status
    if (this.onStatusUpdate) {
      this.onStatusUpdate(status)
    }
    console.log('Segmentation Status:', status)
  }

  async initialize(): Promise<void> {
    try {
      this.updateStatus('Initializing segmentation model...')

      const options: ort.InferenceSession.SessionOptions = {
        executionProviders: this.config.executionProvider === 'webgpu'
          ? ['webgpu', 'wasm']
          : ['webgl', 'wasm'],
        graphOptimizationLevel: 'all'
      }

      this.updateStatus('Loading segmentation model...')
      this.session = await ort.InferenceSession.create(this.config.modelPath, options)

      this.updateStatus('Segmentation model loaded successfully')
      console.log('Segmentation Input names:', this.session.inputNames)
      console.log('Segmentation Output names:', this.session.outputNames)

      setTimeout(() => {
        this.updateStatus('')
      }, 2000)
    } catch (error) {
      this.updateStatus('Failed to initialize segmentation model')
      console.error('Failed to initialize segmentation model:', error)
      throw error
    }
  }

  private preprocessImage(imageData: ImageData): Float32Array {
    const { width, height, data } = imageData
    const [, channels, modelHeight, modelWidth] = this.config.inputShape
    const preprocessed = new Float32Array(channels * modelHeight * modelWidth)

    const canvas = new OffscreenCanvas(modelWidth, modelHeight)
    const ctx = canvas.getContext('2d')!

    const tempCanvas = new OffscreenCanvas(width, height)
    const tempCtx = tempCanvas.getContext('2d')!
    tempCtx.putImageData(imageData, 0, 0)

    const padX = 0
    const padY = (modelHeight - height) / 2

    ctx.fillStyle = '#000000'
    ctx.fillRect(0, 0, modelWidth, modelHeight)
    ctx.drawImage(tempCanvas, padX, padY, width, height)

    const resizedData = ctx.getImageData(0, 0, modelWidth, modelHeight).data

    const mean = [0.485, 0.456, 0.406]
    const std = [0.229, 0.224, 0.225]

    for (let i = 0; i < modelHeight * modelWidth; i++) {
      const r = resizedData[i * 4] / 255.0
      const g = resizedData[i * 4 + 1] / 255.0
      const b = resizedData[i * 4 + 2] / 255.0

      preprocessed[i] = (r - mean[0]) / std[0]
      preprocessed[i + modelHeight * modelWidth] = (g - mean[1]) / std[1]
      preprocessed[i + 2 * modelHeight * modelWidth] = (b - mean[2]) / std[2]
    }

    return preprocessed
  }

  async segment(imageData: ImageData): Promise<Uint8Array> {
    if (!this.session) {
      throw new Error('Segmentation model not initialized')
    }

    const preprocessed = this.preprocessImage(imageData)

    const input = new ort.Tensor('float32', preprocessed, this.config.inputShape)

    const startTime = performance.now()
    const outputs = await this.session.run({ input: input })
    const inferenceTime = performance.now() - startTime
    console.log(`Segmentation inference time: ${inferenceTime.toFixed(2)}ms`)

    const output = outputs[this.session.outputNames[0]] as ort.Tensor
    const outputData = output.data as Float32Array

    console.log('Segmentation output dims:', output.dims)

    const [, , modelHeight, modelWidth] = output.dims

    const mask = new Uint8Array(modelWidth * modelHeight)

    for (let i = 0; i < modelWidth * modelHeight; i++) {
      mask[i] = outputData[i] >= this.config.threshold ? 255 : 0
    }

    const padY = (modelHeight - imageData.height) / 2
    const croppedMask = new Uint8Array(imageData.width * imageData.height)

    for (let y = 0; y < imageData.height; y++) {
      for (let x = 0; x < imageData.width; x++) {
        const srcIdx = (y + padY) * modelWidth + x
        const dstIdx = y * imageData.width + x
        croppedMask[dstIdx] = mask[srcIdx]
      }
    }

    return croppedMask
  }

  async dispose(): Promise<void> {
    if (this.session) {
      try {
        const tempSession = this.session
        this.session = null
        await tempSession.release()
        console.log('Segmentation session released successfully')
      } catch (error) {
        console.warn('Error releasing segmentation session:', error)
      }
    }
  }
}

export function renderSegmentationMask(
  ctx: CanvasRenderingContext2D,
  mask: Uint8Array,
  width: number,
  height: number,
  color: string = 'rgba(0, 255, 0, 0.3)'
): void {
  const imageData = ctx.createImageData(width, height)
  const data = imageData.data

  const rgba = color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/)
  if (!rgba) return

  const r = parseInt(rgba[1])
  const g = parseInt(rgba[2])
  const b = parseInt(rgba[3])
  const a = rgba[4] ? parseFloat(rgba[4]) * 255 : 255

  for (let i = 0; i < mask.length; i++) {
    if (mask[i] > 0) {
      const pixelIndex = i * 4
      data[pixelIndex] = r
      data[pixelIndex + 1] = g
      data[pixelIndex + 2] = b
      data[pixelIndex + 3] = a
    }
  }

  ctx.save()
  ctx.globalCompositeOperation = 'multiply'
  ctx.putImageData(imageData, 0, 0)
  ctx.globalCompositeOperation = 'lighter'
  ctx.putImageData(imageData, 0, 0)
  ctx.restore()
}