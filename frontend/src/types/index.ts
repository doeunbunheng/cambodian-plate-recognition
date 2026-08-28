export interface RecognitionResult {
  vehicle: string;
  category: string;
  plate_number: string;
  processing_time?: number;
}
