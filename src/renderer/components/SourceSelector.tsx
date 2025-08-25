import React, { useState, useEffect } from 'react'

interface Source {
  id: string
  name: string
  thumbnail: string
}

interface SourceSelectorProps {
  onSelect: (sourceId: string) => void
  onCancel: () => void
}

const SourceSelector: React.FC<SourceSelectorProps> = ({ onSelect, onCancel }) => {
  const [sources, setSources] = useState<Source[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const getSources = async () => {
      try {
        if (window.electronAPI) {
          const availableSources = await window.electronAPI.getSources()
          setSources(availableSources)
        }
      } catch (error) {
        console.error('Failed to get sources:', error)
      } finally {
        setLoading(false)
      }
    }

    getSources()
  }, [])

  return (
    <div className="source-selector-overlay">
      <div className="source-selector-modal">
        <h2>Select Screen to Record</h2>
        {loading ? (
          <div className="loading">Loading sources...</div>
        ) : (
          <div className="source-grid">
            {sources.map((source) => (
              <div
                key={source.id}
                className="source-item"
                onClick={() => onSelect(source.id)}
              >
                <img src={source.thumbnail} alt={source.name} />
                <p>{source.name}</p>
              </div>
            ))}
          </div>
        )}
        <button onClick={onCancel} className="btn">
          Cancel
        </button>
      </div>
    </div>
  )
}

export default SourceSelector