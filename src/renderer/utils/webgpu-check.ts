export interface GPUAdapterDetails {
  adapter: GPUAdapter;
  info: GPUAdapterInfo;
  vendor: string;
  architecture: string;
  device: string;
  description: string;
}

export async function getAllAvailableAdapters(): Promise<GPUAdapterDetails[]> {
  if (!navigator.gpu) {
    console.warn('WebGPU API is not available');
    return [];
  }

  const adapters: GPUAdapterDetails[] = [];
  const seenDeviceIds = new Set<string>();

  // 注意: 現在のWebGPU APIでは複数のGPUを直接列挙する方法がないため、
  // powerPreferenceを使って異なるGPUにアクセスを試みます
  // 将来的にはnavigator.gpu.getAdapters()のようなAPIが追加される可能性があります
  
  // デフォルトのアダプターを取得
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter) {
      const info = await adapter.requestAdapterInfo();
      const deviceId = `${info.vendor}-${info.device}-${info.architecture}`;
      
      if (!seenDeviceIds.has(deviceId)) {
        seenDeviceIds.add(deviceId);
        adapters.push({
          adapter,
          info,
          vendor: info.vendor || 'Unknown',
          architecture: info.architecture || 'Unknown',
          device: info.device || 'Unknown',
          description: info.description || 'Unknown GPU'
        });
      }
    }
  } catch (error) {
    console.error('Failed to get default adapter:', error);
  }

  // 高性能GPUを明示的にリクエスト（異なるGPUが返される可能性）
  try {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance'
    });
    if (adapter) {
      const info = await adapter.requestAdapterInfo();
      const deviceId = `${info.vendor}-${info.device}-${info.architecture}`;
      
      if (!seenDeviceIds.has(deviceId)) {
        seenDeviceIds.add(deviceId);
        adapters.push({
          adapter,
          info,
          vendor: info.vendor || 'Unknown',
          architecture: info.architecture || 'Unknown',
          device: info.device || 'Unknown',
          description: info.description || 'Unknown GPU'
        });
      }
    }
  } catch (error) {
    console.error('Failed to get high-performance adapter:', error);
  }

  // 省電力GPUを明示的にリクエスト（統合GPUが返される可能性）
  try {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'low-power'
    });
    if (adapter) {
      const info = await adapter.requestAdapterInfo();
      const deviceId = `${info.vendor}-${info.device}-${info.architecture}`;
      
      if (!seenDeviceIds.has(deviceId)) {
        seenDeviceIds.add(deviceId);
        adapters.push({
          adapter,
          info,
          vendor: info.vendor || 'Unknown',
          architecture: info.architecture || 'Unknown',
          device: info.device || 'Unknown',
          description: info.description || 'Unknown GPU'
        });
      }
    }
  } catch (error) {
    console.error('Failed to get low-power adapter:', error);
  }

  console.log(`Found ${adapters.length} unique GPU adapter(s)`);
  adapters.forEach((adapter, index) => {
    console.log(`  ${index}: ${adapter.description} (${adapter.vendor})`);
  });

  return adapters;
}

export async function selectGPUAdapterByIndex(index: number): Promise<GPUAdapter | null> {
  if (!navigator.gpu) {
    console.warn('WebGPU API is not available');
    return null;
  }

  const adapters = await getAllAvailableAdapters();

  if (adapters.length === 0) {
    console.warn('No WebGPU adapters found');
    return null;
  }

  if (index < 0 || index >= adapters.length) {
    console.warn(`Invalid adapter index: ${index}, falling back to first adapter`);
    return adapters[0].adapter;
  }

  const selectedAdapter = adapters[index];
  console.log(`Selected GPU ${index}:`, selectedAdapter.description);
  return selectedAdapter.adapter;
}


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
    const adapter = await navigator.gpu.requestAdapter();

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

    // 利用可能な全アダプター情報を表示
    const allAdapters = await getAllAvailableAdapters();
    console.log(`Found ${allAdapters.length} GPU adapter(s):`);
    allAdapters.forEach((adapterInfo, index) => {
      console.log(`  ${index + 1}. ${adapterInfo.description} (${adapterInfo.vendor})`);
    });

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

export async function getGPUCount(): Promise<number> {
  if (!navigator.gpu) {
    return 0;
  }

  const adapters = await getAllAvailableAdapters();
  // 実際のハードウェアGPUの数を返す
  return adapters.length;
}