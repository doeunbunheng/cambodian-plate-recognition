import { useState, useRef } from 'react'

interface ImageUploaderProps {
  onImageSelected: (file: File, preview: string) => void
  isLoading: boolean
}

export function ImageUploader({ onImageSelected, isLoading }: ImageUploaderProps) {
  const [preview, setPreview] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFile = (file: File) => {
    if (!file.type.startsWith('image/')) return
    const previewUrl = URL.createObjectURL(file)
    setPreview(previewUrl)
    onImageSelected(file, previewUrl)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  return (
    <div className="flex flex-col items-center gap-4">
      <div
        className={`w-full max-w-md h-64 rounded-lg border-2 border-dashed flex flex-col items-center justify-center cursor-pointer transition-colors ${
          dragOver
            ? 'border-blue-500 bg-blue-50'
            : 'border-gray-300 bg-gray-50 hover:border-blue-400 hover:bg-blue-50/50'
        }`}
        onClick={() => fileInputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        {preview ? (
          <img
            src={preview}
            alt="Selected Plate"
            className="w-full h-full object-contain rounded-lg"
          />
        ) : (
          <>
            <svg
              className="w-16 h-16 text-gray-400 mb-2"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14M4 8V6a2 2 0 012-2h12a4 4 0 014 4v6m-6 0a2 2 0 10-4 0"
              />
            </svg>
            <p className="text-gray-500 font-medium">
              {isLoading ? 'Processing...' : 'Click to upload or drag & drop image'}
            </p>
            <p className="text-sm text-gray-400 mt-1">JPG, PNG files supported</p>
          </>
        )}
      </div>
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) handleFile(file)
        }}
      />
    </div>
  )
}
