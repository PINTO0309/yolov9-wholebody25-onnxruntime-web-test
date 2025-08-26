import React, { useEffect, useRef } from 'react'
import { BoundingBox } from '../utils/types'

interface DetectionCanvasProps {
  width: number
  height: number
  detections: BoundingBox[]
  videoRef: React.RefObject<HTMLVideoElement>
  segmentationMask?: Uint8Array | null
}

const DetectionCanvas: React.FC<DetectionCanvasProps> = ({
  width,
  height,
  detections,
  videoRef,
  segmentationMask
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const video = videoRef.current
    if (!canvas || !video) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const draw = () => {
      ctx.clearRect(0, 0, width, height)
      
      if (video.readyState === video.HAVE_ENOUGH_DATA) {
        ctx.drawImage(video, 0, 0, width, height)
        
        if (segmentationMask && segmentationMask.length > 0) {
          ctx.save()
          ctx.globalAlpha = 0.3
          ctx.fillStyle = 'rgb(0, 255, 0)'
          
          for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
              const idx = y * width + x
              if (segmentationMask[idx] > 0) {
                ctx.fillRect(x, y, 1, 1)
              }
            }
          }
          
          ctx.restore()
        }
        
        detections.forEach((detection) => {
          const { x1, y1, x2, y2, confidence, label } = detection
          const boxWidth = x2 - x1
          const boxHeight = y2 - y1
          
          const hue = (detection.class * 137) % 360
          const color = `hsl(${hue}, 70%, 50%)`
          
          ctx.strokeStyle = color
          ctx.lineWidth = 2
          ctx.strokeRect(x1, y1, boxWidth, boxHeight)
          
          ctx.fillStyle = color
          ctx.fillRect(x1, y1 - 20, boxWidth, 20)
          
          ctx.fillStyle = 'white'
          ctx.font = '14px Arial'
          const text = `${label} ${(confidence * 100).toFixed(1)}%`
          ctx.fillText(text, x1 + 4, y1 - 5)
        })
      }
    }

    draw()
  }, [detections, width, height, videoRef, segmentationMask])

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        zIndex: 2
      }}
    />
  )
}

export default DetectionCanvas