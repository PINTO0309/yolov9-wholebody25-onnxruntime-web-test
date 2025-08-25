import { BoundingBox } from './types'

function calculateIoU(box1: BoundingBox, box2: BoundingBox): number {
  const x1 = Math.max(box1.x1, box2.x1)
  const y1 = Math.max(box1.y1, box2.y1)
  const x2 = Math.min(box1.x2, box2.x2)
  const y2 = Math.min(box1.y2, box2.y2)

  const intersectionWidth = Math.max(0, x2 - x1)
  const intersectionHeight = Math.max(0, y2 - y1)
  const intersectionArea = intersectionWidth * intersectionHeight

  const box1Area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
  const box2Area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)

  const unionArea = box1Area + box2Area - intersectionArea

  return intersectionArea / unionArea
}

export function nonMaxSuppression(
  boxes: BoundingBox[],
  iouThreshold: number = 0.45
): BoundingBox[] {
  if (boxes.length === 0) return []

  const sortedBoxes = [...boxes].sort((a, b) => b.confidence - a.confidence)
  const selected: BoundingBox[] = []

  while (sortedBoxes.length > 0) {
    const current = sortedBoxes.shift()!
    selected.push(current)

    const remaining: BoundingBox[] = []
    for (const box of sortedBoxes) {
      if (box.class !== current.class || calculateIoU(current, box) < iouThreshold) {
        remaining.push(box)
      }
    }
    sortedBoxes.length = 0
    sortedBoxes.push(...remaining)
  }

  return selected
}