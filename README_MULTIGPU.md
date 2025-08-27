# Multi-GPU Support for WebGPU

## Current Limitations

WebGPU API currently has limitations in detecting and accessing multiple GPUs. Most browsers (Chrome, Edge) only expose a single GPU by default, even on systems with multiple GPUs.

## Known Issues

### NVIDIA Multi-GPU Systems
On systems with multiple NVIDIA GPUs (like yours with 4x GPUs), WebGPU may only detect one GPU due to browser security restrictions.

## Workarounds

### For Chrome/Edge

1. **Enable Developer Features**
   - Navigate to `chrome://flags/#enable-webgpu-developer-features`
   - Set to "Enabled"
   - Restart browser

2. **Launch with Unsafe WebGPU Flag**
   ```bash
   # Linux/Mac
   google-chrome --enable-unsafe-webgpu --enable-features=WebGPUDeveloperFeatures

   # Windows
   chrome.exe --enable-unsafe-webgpu --enable-features=WebGPUDeveloperFeatures
   ```

3. **For Electron App**
   - Add the following to your Electron main process:
   ```javascript
   app.commandLine.appendSwitch('enable-unsafe-webgpu')
   app.commandLine.appendSwitch('enable-features', 'WebGPUDeveloperFeatures')
   ```

### For Firefox
Firefox Nightly may have better multi-GPU support. Enable WebGPU in `about:config`:
- Set `dom.webgpu.enabled` to `true`

## Debugging

Check browser console for GPU detection logs:
```
=== GPU Detection Debug ===
Requesting default adapter...
Requesting high-performance adapter...
Requesting low-power adapter...
=== GPU Detection Summary ===
Total unique GPUs found: X
```

## Alternative Solutions

If WebGPU cannot detect all GPUs:

1. **Use CUDA/DirectML backend** (if available in ONNX Runtime)
2. **Use WebGL** (single GPU only)
3. **Native application** instead of web-based

## Future Improvements

The WebGPU specification is still evolving. Future versions may include:
- `navigator.gpu.getAdapters()` for listing all GPUs
- Better multi-GPU device selection APIs
- Improved cross-GPU communication

## Current Implementation

Our implementation attempts to detect multiple GPUs by requesting adapters with different power preferences:
1. Default adapter
2. High-performance adapter (discrete GPU)
3. Low-power adapter (integrated GPU)

This may detect up to 2 GPUs on systems with both integrated and discrete graphics, but won't detect multiple discrete GPUs of the same type.