import { useEffect, useRef, useState } from 'react'

export const useWebcam = (width: number = 640, height: number = 480) => {
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const [isReady, setIsReady] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const initWebcam = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: width },
            height: { ideal: height },
            facingMode: 'user'
          }
        })

        if (videoRef.current) {
          videoRef.current.srcObject = stream
          streamRef.current = stream
          
          videoRef.current.onloadedmetadata = () => {
            videoRef.current?.play()
            setIsReady(true)
          }
        }
      } catch (err) {
        console.error('Failed to access webcam:', err)
        setError('Failed to access webcam. Please check permissions.')
      }
    }

    initWebcam()

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
      }
    }
  }, [width, height])

  return { videoRef, isReady, error }
}