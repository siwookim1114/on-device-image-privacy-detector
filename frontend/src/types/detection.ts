export interface BoundingBox {
  x: number; y: number; width: number; height: number;
}

export interface Detection {
  id: string;
  category: 'face' | 'text' | 'object';
  bbox: BoundingBox;
  confidence: number;
  attributes: Record<string, unknown>;
}

export interface FaceDetection extends Detection {
  category: 'face';
  landmarks: Record<string, [number, number]> | null;
  size: string | null;
  clarity: string | null;
  angle: string | null;
}

export interface TextDetection extends Detection {
  category: 'text';
  text_content: string;
  text_type: string | null;
  polygon: number[][] | null;
  language: string;
}

export interface ObjectDetection extends Detection {
  category: 'object';
  object_class: string;
  contains_text: boolean;
  contains_screen: boolean;
}

export interface DetectionResults {
  faces: FaceDetection[];
  text_regions: TextDetection[];
  objects: ObjectDetection[];
  scene_context: Record<string, unknown>;
  processing_time_ms: number;
  image_path?: string;
  annotated_image_path?: string;
}
