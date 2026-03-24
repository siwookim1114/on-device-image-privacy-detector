import { type ReactNode, useEffect, useRef, useState } from 'react';
import { Stage, Layer, Image as KonvaImage } from 'react-konva';

import { useCanvasInteraction, type CanvasInteractionState } from '../../hooks/useCanvasInteraction';

interface ImageCanvasProps {
  imageUrl: string;
  /** Overlay components (e.g. BboxOverlay) rendered on top of the image. */
  children?: ReactNode;
  /**
   * Called once the image has loaded and its natural dimensions are known.
   * Consumers can use this to correctly map bounding-box coordinates.
   */
  onImageLoaded?: (width: number, height: number) => void;
  /**
   * Called once the container size is known or changes via ResizeObserver.
   * Consumers can use this (together with onImageLoaded) to call fitToScreen
   * from an external CanvasInteractionState.
   */
  onContainerResized?: (width: number, height: number) => void;
  /**
   * Optional external canvas interaction state. When provided the component
   * uses this state (zoom, offsets, handlers) instead of creating its own.
   * Pass the return value of useCanvasInteraction() from a parent component
   * to allow external zoom controls (e.g. ZoomControls) to drive the canvas.
   */
  interaction?: CanvasInteractionState;
}

export function ImageCanvas({
  imageUrl,
  children,
  onImageLoaded,
  onContainerResized,
  interaction,
}: ImageCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const [htmlImage, setHtmlImage] = useState<HTMLImageElement | null>(null);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });

  const internalInteraction = useCanvasInteraction();

  // Use external interaction state when provided; fall back to internal.
  const { zoom, offsetX, offsetY, handleWheel, handleDragEnd, fitToScreen } =
    interaction ?? internalInteraction;

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const { width, height } = entry.contentRect;
      setContainerSize({ width, height });
      onContainerResized?.(width, height);
    });

    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (!imageUrl) return;

    const img = new Image();

    img.onload = () => {
      setHtmlImage(img);
      setImageSize({ width: img.naturalWidth, height: img.naturalHeight });
      onImageLoaded?.(img.naturalWidth, img.naturalHeight);
    };

    img.onerror = () => {
      console.error('[ImageCanvas] Failed to load image:', imageUrl);
    };

    img.src = imageUrl;

    return () => {
      img.onload = null;
      img.onerror = null;
    };
  }, [imageUrl, onImageLoaded]);

  useEffect(() => {
    if (
      imageSize.width > 0 &&
      imageSize.height > 0 &&
      containerSize.width > 0 &&
      containerSize.height > 0
    ) {
      fitToScreen(
        imageSize.width,
        imageSize.height,
        containerSize.width,
        containerSize.height,
      );
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageSize.width, imageSize.height, containerSize.width, containerSize.height]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      // Block the browser's default Space scroll when focus is on the canvas.
      if (e.code === 'Space') e.preventDefault();
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  return (
    <div
      ref={containerRef}
      className="flex-1 bg-gray-950 overflow-hidden relative"
    >
      {containerSize.width > 0 && containerSize.height > 0 && (
        <Stage
          width={containerSize.width}
          height={containerSize.height}
          onWheel={handleWheel}
        >
          <Layer
            x={offsetX}
            y={offsetY}
            scaleX={zoom}
            scaleY={zoom}
            draggable
            onDragEnd={handleDragEnd}
          >
            {htmlImage && (
              <KonvaImage
                image={htmlImage}
                x={0}
                y={0}
                width={imageSize.width}
                height={imageSize.height}
              />
            )}
          </Layer>

          <Layer x={offsetX} y={offsetY} scaleX={zoom} scaleY={zoom}>
            {children}
          </Layer>
        </Stage>
      )}
    </div>
  );
}
