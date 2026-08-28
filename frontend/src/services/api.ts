import type { RecognitionResult } from '../types'

const API_URL = import.meta.env.VITE_API_URL || '/api'

export async function recognizePlate(file: File): Promise<RecognitionResult> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_URL}/recognize`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to recognize plate')
  }

  return response.json()
}
