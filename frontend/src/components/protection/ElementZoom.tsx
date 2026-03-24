import { useRef, useEffect } from 'react';

import type { BoundingBox } from '../../types/detection';

interface ElementZoomProps {
  imageUrl: string;
  bbox: BoundingBox;
  label: string;
}

const PADDING_RATIO = 0.2;
const MIN_ZOOM = 2;
const MAX_ZOOM = 4;

export function ElementZoom({ imageUrl, bbox, label }: ElementZoomProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let cancelled = false;
    const img = new Image();

    img.onload = () => {
      if (cancelled || !canvas) return;

      const imgW = img.naturalWidth;
      const imgH = img.naturalHeight;

      // Compute padded crop region in image-space pixels
      const padX = bbox.width * PADDING_RATIO;
      const padY = bbox.height * PADDING_RATIO;

      const cropX = Math.max(0, bbox.x - padX);
      const cropY = Math.max(0, bbox.y - padY);
      const cropW = Math.min(imgW - cropX, bbox.width + padX * 2);
      const cropH = Math.min(imgH - cropY, bbox.height + padY * 2);

      // Determine zoom factor — larger crops use lower zoom to stay readable
      const rawZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, 300 / Math.max(cropW, cropH)));
      const zoom = Math.round(rawZoom * 2) / 2; // round to nearest 0.5

      canvas.width = Math.round(cropW * zoom);
      canvas.height = Math.round(cropH * zoom);

      ctx.drawImage(
        img,
        cropX, cropY, cropW, cropH,
        0, 0, canvas.width, canvas.height,
      );
    };

    img.src = imageUrl;

    return () => {
      cancelled = true;
      img.onload = null;
    };
  }, [imageUrl, bbox]);

  return (
    <div className="bg-gray-800 rounded-lg p-2 max-h-48 overflow-hidden flex flex-col gap-1.5">
      <div className="overflow-hidden rounded flex-1 min-h-0 flex items-center justify-center bg-gray-950">
        <canvas
          ref={canvasRef}
          className="max-w-full max-h-full object-contain"
          aria-label={`Zoomed view of ${label}`}
        />
      </div>
      <p className="text-xs text-gray-400 text-center truncate leading-tight px-1">{label}</p>
    </div>
  );
}
