import { useState, useEffect } from 'react'
import { getAllAvailableAdapters, GPUAdapterDetails } from '../utils/webgpu-check'

interface GPUSelectorProps {
  label?: string;
  onSelectGPU?: (adapterIndex: number) => void;
  disabled?: boolean;
  singleGpuMode?: boolean;
}

export default function GPUSelector({ label = "GPU:", onSelectGPU, disabled, singleGpuMode }: GPUSelectorProps) {
  const [adapters, setAdapters] = useState<GPUAdapterDetails[]>([])
  const [selectedAdapterIndex, setSelectedAdapterIndex] = useState<number>(0)
  const [loading, setLoading] = useState(true)
  const selectId = `gpu-select-${label.replace(/[^a-zA-Z0-9]/g, '')}`

  useEffect(() => {
    const loadAdapters = async () => {
      setLoading(true)
      try {
        const availableAdapters = await getAllAvailableAdapters()
        setAdapters(availableAdapters)
      } catch (error) {
        console.error('Failed to load GPU adapters:', error)
      } finally {
        setLoading(false)
      }
    }

    loadAdapters()
  }, [])

  const handleSelectionChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const index = parseInt(e.target.value)
    setSelectedAdapterIndex(index)

    if (onSelectGPU && !isNaN(index)) {
      onSelectGPU(index)
    }
  }

  if (loading) {
    return <div className="gpu-selector">Loading GPUs...</div>
  }

  if (adapters.length === 0) {
    return <div className="gpu-selector">No GPU adapters found</div>
  }

  return (
    <div className="gpu-selector">
      <label htmlFor={selectId}>
        {label}
        {singleGpuMode && <span className="single-gpu-indicator" title="Single GPU mode - selection disabled"> (Single GPU)</span>}
      </label>
      <select 
        id={selectId} 
        value={selectedAdapterIndex} 
        onChange={handleSelectionChange}
        disabled={disabled}
        title={singleGpuMode ? "GPU selection is disabled in single GPU environment" : ""}
      >
        {adapters.map((adapter, index) => (
          <option key={index} value={index}>
            {adapter.description || `GPU ${index + 1}`} ({adapter.vendor || 'Unknown'})
          </option>
        ))}
      </select>
    </div>
  )
}