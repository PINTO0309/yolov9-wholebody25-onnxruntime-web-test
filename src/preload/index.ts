import { contextBridge, ipcRenderer } from 'electron'

// Debug: Log that preload script is executing
console.log('Preload script is executing...')
console.log('Process platform:', process.platform)

try {
  contextBridge.exposeInMainWorld('electronAPI', {
    platform: process.platform,
    getSources: async () => {
      console.log('getSources called from preload')
      try {
        const sources = await ipcRenderer.invoke('get-sources')
        console.log('Sources obtained:', sources.length)
        return sources
      } catch (error) {
        console.error('Error in getSources:', error)
        throw error
      }
    },
    toggleDevTools: async (enable: boolean) => {
      await ipcRenderer.invoke('toggle-devtools', enable)
    }
  })
  console.log('electronAPI exposed successfully')
} catch (error) {
  console.error('Failed to expose electronAPI:', error)
}