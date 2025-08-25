import { app, BrowserWindow, shell, ipcMain, desktopCapturer } from 'electron'
import { join } from 'path'

// WebGPU/WebGLのための最適化設定
// 開発環境、本番環境両方でWebGPUを有効化
app.commandLine.appendSwitch('enable-unsafe-webgpu');
app.commandLine.appendSwitch("enable-webgpu-developer-features");

// Linux環境でWebGPUを有効化するために必要なVulkanサポート
app.commandLine.appendSwitch("enable-features", "Vulkan,WebGPU,WebGPUService");

// GPU設定の最適化
app.commandLine.appendSwitch("ignore-gpu-blocklist");
app.commandLine.appendSwitch("enable-gpu-rasterization");
app.commandLine.appendSwitch("enable-zero-copy");
app.commandLine.appendSwitch("disable-gpu-sandbox");

// ANGLEバックエンドを使用（chrome://gpu/と同じ設定）
app.commandLine.appendSwitch("use-gl", "angle");
app.commandLine.appendSwitch("use-angle", "gl");

// WebGPUアダプターの設定（NVIDIA GPU向け）
app.commandLine.appendSwitch("use-webgpu-adapter", "default");

// SharedImageエラーを回避するための設定
app.commandLine.appendSwitch("disable-features", "UseSkiaRenderer,UseChromeOSDirectVideoDecoder");

// ログレベルの設定（0=INFO, 1=WARNING, 2=ERROR, 3=FATAL）
app.commandLine.appendSwitch("log-level", "3");

let mainWindow: BrowserWindow | null = null

function createWindow(): void {
  const preloadPath = join(__dirname, '../preload/index.js')
  console.log('Preload script path:', preloadPath)
  console.log('Preload script exists:', require('fs').existsSync(preloadPath))
  console.log('__dirname:', __dirname)
  
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    show: false,
    autoHideMenuBar: true,
    webPreferences: {
      preload: preloadPath,
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: false,
      // Enable webviewTag for debugging
      webviewTag: false,
      // Ensure sandbox is disabled for desktopCapturer
      sandbox: false
    }
  })

  mainWindow.on('ready-to-show', () => {
    mainWindow?.show()
    // DevTools will be controlled by the Debug toggle button
  })

  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url)
    return { action: 'deny' }
  })
  
  // Add logging for when the page is loaded
  mainWindow.webContents.on('did-finish-load', () => {
    console.log('Page finished loading')
    // Check if preload script worked
    mainWindow?.webContents.executeJavaScript(`
      console.log('Checking from main process:');
      console.log('window.electronAPI:', typeof window.electronAPI);
      typeof window.electronAPI !== 'undefined' ? 'API loaded' : 'API not loaded';
    `).then(result => {
      console.log('Preload API status:', result)
    })
  })

  if (!app.isPackaged && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

app.whenReady().then(() => {
  // Handle IPC for desktop capturer
  ipcMain.handle('get-sources', async () => {
    try {
      const sources = await desktopCapturer.getSources({
        types: ['window', 'screen'],
        thumbnailSize: { width: 320, height: 180 }
      })
      return sources.map(source => ({
        id: source.id,
        name: source.name,
        thumbnail: source.thumbnail.toDataURL()
      }))
    } catch (error) {
      console.error('Error getting sources:', error)
      throw error
    }
  })

  // Handle IPC for DevTools toggle
  ipcMain.handle('toggle-devtools', async (event, enable: boolean) => {
    if (mainWindow) {
      if (enable) {
        mainWindow.webContents.openDevTools()
      } else {
        mainWindow.webContents.closeDevTools()
      }
    }
  })

  createWindow()

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})