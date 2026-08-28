import type { RecognitionResult } from '../types'

interface DetectionResultsProps {
  result: RecognitionResult | null
  isLoading: boolean
}

export function DetectionResults({ result, isLoading }: DetectionResultsProps) {
  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6 flex items-center justify-center min-h-40">
        <div className="flex items-center gap-3">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
          <span className="text-gray-500">Analyzing image...</span>
        </div>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-lg font-semibold text-gray-700 mb-4">Detection Results</h2>
        <p className="text-gray-400 text-center py-8">Upload an image to see results</p>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-lg font-bold text-gray-800 mb-4">Recognition Results</h2>

      <div className="space-y-4">
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="text-xs text-gray-500 uppercase mb-1">Vehicle</div>
          <div className="text-xl font-bold text-gray-800">{result.vehicle || 'Not detected'}</div>
        </div>

        <div className="bg-blue-50 rounded-lg p-4">
          <div className="text-xs text-blue-600 uppercase mb-1">Plate Category</div>
          <div className="text-xl font-bold text-blue-700">{result.category || 'Not detected'}</div>
        </div>

        <div className="bg-green-50 rounded-lg p-4">
          <div className="text-xs text-green-600 uppercase mb-1">Plate Number</div>
          <div className="text-2xl font-bold text-green-700 tracking-wider">
            {result.plate_number || 'Not detected'}
          </div>
        </div>

        {result.processing_time && (
          <div className="text-xs text-gray-400 text-right">
            Processing time: {result.processing_time}s
          </div>
        )}
      </div>
    </div>
  )
}
