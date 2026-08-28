import { useState } from 'react'
import { ImageUploader } from './components/ImageUploader'
import { DetectionResults } from './components/DetectionResults'
import { recognizePlate } from './services/api'
import type { RecognitionResult } from './types'

function App() {
  const [result, setResult] = useState<RecognitionResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleImageSelected = async (file: File) => {
    setIsLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await recognizePlate(file)
      setResult(response)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process image')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <h1 className="text-xl font-bold text-gray-800">
            🚗 Cambodian Plate Recognition
          </h1>
          <span className="text-sm text-gray-500">Vehicle • Category • Plate Number</span>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid md:grid-cols-2 gap-8">
          <div>
            <h2 className="text-lg font-semibold text-gray-700 mb-4">Upload Image</h2>
            <ImageUploader onImageSelected={handleImageSelected} isLoading={isLoading} />
          </div>

          <div>
            <h2 className="text-lg font-semibold text-gray-700 mb-4">Results</h2>
            <DetectionResults result={result} isLoading={isLoading} />
          </div>
        </div>

        {error && (
          <div className="mt-6 bg-red-50 border border-red-200 text-red-700 rounded-lg p-4">
            {error}
          </div>
        )}
      </main>
    </div>
  )
}

export default App
