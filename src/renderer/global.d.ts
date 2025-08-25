interface Window {
  electronAPI: {
    platform: string
    getSources: () => Promise<Array<{
      id: string
      name: string
      thumbnail: string
    }>>
    toggleDevTools: (enable: boolean) => Promise<void>
  }
}