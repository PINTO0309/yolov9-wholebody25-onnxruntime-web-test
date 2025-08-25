export async function checkWebGPUSupport(): Promise<boolean> {
  console.log('Checking WebGPU support...');
  console.log('navigator.gpu available:', !!navigator.gpu);
  
  if (!navigator.gpu) {
    console.warn('WebGPU API is not available in this environment');
    console.log('User Agent:', navigator.userAgent);
    return false;
  }

  try {
    console.log('Requesting WebGPU adapter...');
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
      forceFallbackAdapter: false
    });
    
    if (!adapter) {
      console.warn('No WebGPU adapter found - GPU may not support WebGPU');
      return false;
    }

    console.log('WebGPU adapter found:', adapter);
    const info = await adapter.requestAdapterInfo();
    console.log('Adapter info:', info);

    const device = await adapter.requestDevice();
    
    if (!device) {
      console.warn('Failed to get WebGPU device');
      return false;
    }

    console.log('✅ WebGPU is fully supported and available');
    console.log('Device features:', Array.from(device.features));
    console.log('Device limits:', device.limits);
    
    // デバイスを破棄
    device.destroy();
    
    return true;
  } catch (error) {
    console.error('WebGPU check failed:', error);
    return false;
  }
}

export function getWebGPUInfo(): { supported: boolean; message: string } {
  if (!navigator.gpu) {
    return {
      supported: false,
      message: 'WebGPU API not available'
    };
  }

  return {
    supported: true,
    message: 'WebGPU API available (adapter check required)'
  };
}