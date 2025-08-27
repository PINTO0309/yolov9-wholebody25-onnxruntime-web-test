// WebGPU type definitions for TypeScript
interface GPUAdapter {
  readonly name: string;
  readonly features: ReadonlySet<string>;
  readonly limits: Record<string, number>;
  requestDevice(descriptor?: GPUDeviceDescriptor): Promise<GPUDevice>;
  requestAdapterInfo(): Promise<GPUAdapterInfo>;
}

interface GPUAdapterInfo {
  readonly vendor: string;
  readonly architecture: string;
  readonly device: string;
  readonly description: string;
}

interface GPUDevice {
  readonly features: ReadonlySet<string>;
  readonly limits: Record<string, number>;
  destroy(): void;
}

interface GPUDeviceDescriptor {
  requiredFeatures?: string[];
  requiredLimits?: Record<string, number>;
}

interface GPU {
  requestAdapter(options?: GPURequestAdapterOptions): Promise<GPUAdapter | null>;
}

interface GPURequestAdapterOptions {
  powerPreference?: 'low-power' | 'high-performance';
  forceFallbackAdapter?: boolean;
}

interface Navigator {
  readonly gpu?: GPU;
}

declare global {
  interface Window {
    readonly navigator: Navigator;
  }
}