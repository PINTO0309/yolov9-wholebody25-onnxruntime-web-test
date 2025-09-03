export interface Detection {
  x: number
  y: number
  width: number
  height: number
  confidence: number
  class: number
  label: string
}

export interface BoundingBox {
  x1: number
  y1: number
  x2: number
  y2: number
  confidence: number
  class: number
  label: string
}

export type ExecutionProvider = 'webgl' | 'webgpu' | 'webnn' | 'wasm'

export interface ModelConfig {
  modelPath: string
  inputShape: [number, number, number, number]
  confidenceThreshold: number
  iouThreshold: number
  executionProvider: ExecutionProvider
}