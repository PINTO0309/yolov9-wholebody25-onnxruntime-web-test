# YOLOv9 Wholebody Detection with ONNX Runtime Web

A high-performance real-time object detection application built with Electron, React, TypeScript, and ONNX Runtime Web. This application performs wholebody detection using YOLOv9 model with support for both WebGPU and WebGL execution providers.

## 🌟 Features

- **Real-time Object Detection**: YOLOv9 wholebody detection with 25 classes
- **Dual Runtime Support**: Seamlessly switch between WebGPU and WebGL execution providers
- **Live Webcam Feed**: 640x480 webcam input processing
- **Performance Monitoring**: Real-time inference speed and FPS display
- **Screen Recording**: Record detection sessions to WebM format
- **Debug Mode**: Toggle developer tools for debugging
- **Optimized Rendering**: Uses OffscreenCanvas for conflict-free rendering
- **Non-Maximum Suppression**: Built-in NMS for accurate detection results

## 📋 Requirements

- Node.js 18.x or higher
- pnpm package manager
- Modern browser with WebGPU support (Chrome 113+ recommended)
- Webcam for live detection
- NVIDIA GPU (recommended for optimal WebGPU performance)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/yolov9-wholebody25-onnxruntime-web-test.git
cd yolov9-wholebody25-onnxruntime-web-test
```

2. Install dependencies using pnpm:
```bash
pnpm install
```

3. Ensure the YOLOv9 model is in place:
```
public/models/yolov9_s_wholebody25_0100_1x3x640x640.onnx
```

## 🎮 Usage

### Development Mode

Start the application in development mode:
```bash
pnpm run dev
```

### Production Build

Build the application for production:
```bash
pnpm run build
```

### Preview Production Build

Preview the built application:
```bash
pnpm run preview
```

## 🎯 Controls

- **Switch to WebGPU/WebGL**: Toggle between WebGPU and WebGL execution providers
- **Start/Stop Detection**: Begin or end real-time object detection
- **Record Screen**: Start/stop screen recording (saves as WebM)
- **Debug ON/OFF**: Toggle developer tools visibility

## 🏗️ Architecture

### Technology Stack

- **Frontend Framework**: React 18 with TypeScript
- **Desktop Framework**: Electron with Electron Vite
- **ML Runtime**: ONNX Runtime Web 1.20.1
- **Build Tool**: Vite 5.4
- **Package Manager**: pnpm

### Project Structure

```
yolov9-wholebody25-onnxruntime-web-test/
├── src/
│   ├── main/              # Electron main process
│   │   └── index.ts       # Main process entry point
│   ├── preload/           # Electron preload scripts
│   │   └── index.ts       # IPC bridge for renderer
│   ├── renderer/          # React application
│   │   ├── App.tsx        # Main application component
│   │   ├── components/    # React components
│   │   │   └── DetectionCanvas.tsx
│   │   ├── hooks/         # Custom React hooks
│   │   │   ├── useWebcam.ts
│   │   │   └── useScreenRecorder.ts
│   │   └── utils/         # Utility functions
│   │       ├── yolov9.ts  # YOLOv9 detection logic
│   │       ├── types.ts   # TypeScript definitions
│   │       └── webgpu-check.ts
├── public/
│   └── models/            # ONNX model files
├── dist-electron/         # Build output
└── electron.vite.config.ts # Electron Vite configuration
```

### Key Components

#### YOLOv9 Detector (`src/renderer/utils/yolov9.ts`)
- Handles model loading and initialization
- Performs image preprocessing (640x640 input)
- Runs inference using ONNX Runtime Web
- Applies Non-Maximum Suppression (NMS)
- Filters detections by confidence threshold

#### Detection Canvas (`src/renderer/components/DetectionCanvas.tsx`)
- Renders bounding boxes on video feed
- Displays class labels and confidence scores
- Handles coordinate transformation from model space to display space

#### Screen Recorder Hook (`src/renderer/hooks/useScreenRecorder.ts`)
- Manages screen capture using Electron's desktopCapturer
- Records to WebM format at 30 FPS
- Handles multiple fallback mechanisms

## ⚙️ Configuration

### Model Configuration

The model configuration can be adjusted in `src/renderer/App.tsx`:

```typescript
const config: ModelConfig = {
  modelPath: '/models/yolov9_s_wholebody25_0100_1x3x640x640.onnx',
  inputShape: [1, 3, 640, 640],
  confidenceThreshold: 0.5,  // Adjust detection confidence
  iouThreshold: 0.45,        // Adjust NMS threshold
  executionProvider: 'webgl' // or 'webgpu'
}
```

### Excluded Classes

The following class IDs are excluded from detection display:
- 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 22, 23

These can be modified in `src/renderer/utils/yolov9.ts`.

### WebGPU Configuration

WebGPU flags are configured in `src/main/index.ts`:
- Vulkan support enabled
- ANGLE backend configuration
- GPU rasterization optimizations
- Zero-copy enabled for performance

## 🔧 Troubleshooting

### WebGPU Not Supported

If WebGPU shows as "Not Supported":
1. Ensure you're using Chrome 113+ or Edge 113+
2. Check `chrome://gpu/` for WebGPU status
3. Enable experimental web platform features in browser flags
4. Verify GPU drivers are up to date

### Model Loading Errors

If the model fails to load:
1. Verify the model file exists in `public/models/`
2. Check console for detailed error messages
3. Ensure WASM files are properly copied to public directory
4. Try switching to WebGL execution provider

### Screen Recording Issues

If screen recording fails:
1. Grant screen recording permissions when prompted
2. Check if Electron has necessary system permissions
3. Verify the Debug mode is OFF (DevTools can interfere)

### Performance Issues

For optimal performance:
1. Use WebGPU provider with compatible GPU
2. Close unnecessary applications
3. Ensure adequate lighting for webcam
4. Reduce confidence threshold if needed

## 🔐 Security Considerations

- Context isolation is enabled for secure IPC communication
- Preload scripts use contextBridge for safe API exposure
- Web security is disabled for local file access (development only)
- No node integration in renderer process

## 📊 Performance Benchmarks

Typical inference times on NVIDIA RTX 3060:
- **WebGPU**: ~15-25ms per frame (40-65 FPS)
- **WebGL**: ~30-50ms per frame (20-33 FPS)

*Performance varies based on hardware and system load*

## 🛠️ Development

### Adding New Features

1. Create feature branch from main
2. Implement changes following existing patterns
3. Test with both WebGPU and WebGL providers
4. Ensure TypeScript types are properly defined
5. Update this README if needed

### Debugging

1. Click "Debug ON" button to open DevTools
2. Check console for detailed logs
3. Monitor network tab for model loading
4. Use performance tab for profiling

### Building for Distribution

```bash
# Build for current platform
pnpm run build
pnpm run dist

# Build for specific platform
pnpm run dist:win   # Windows
pnpm run dist:mac   # macOS
pnpm run dist:linux # Linux
```

## 📝 License

Copyright 2025 Katsuya Hyodo

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

See the [LICENSE](LICENSE) file for the full license text.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- YOLOv9 model architecture and weights
- ONNX Runtime Web team for the inference engine
- Electron team for the desktop framework
- React and TypeScript communities

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Provide detailed error messages and system information

---

**Note**: This application is for educational and research purposes. Ensure compliance with applicable privacy laws when using webcam and screen recording features.