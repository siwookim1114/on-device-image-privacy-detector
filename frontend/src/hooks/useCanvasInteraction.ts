import { useState, useCallback } from 'react';
import type Konva from 'konva';
import { fitScale } from '../lib/canvas-utils';

const MIN_ZOOM = 0.1;
const MAX_ZOOM = 5.0;
const ZOOM_FACTOR = 1.1;

export interface CanvasInteractionState {
  zoom: number;
  offsetX: number;
  offsetY: number;
  handleWheel: (e: Konva.KonvaEventObject<WheelEvent>) => void;
  handleDragEnd: (e: Konva.KonvaEventObject<DragEvent>) => void;
  fitToScreen: (
    imageWidth: number,
    imageHeight: number,
    containerWidth: number,
    containerHeight: number,
  ) => void;
}

export function useCanvasInteraction(): CanvasInteractionState {
  const [zoom, setZoom] = useState(1);
  const [offsetX, setOffsetX] = useState(0);
  const [offsetY, setOffsetY] = useState(0);

  /**
   * Zoom in/out centred on the mouse cursor position within the stage.
   * The zoom is applied around the pointer so the point under the cursor
   * stays stationary — standard map-style zoom behaviour.
   */
  const handleWheel = useCallback((e: Konva.KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();

    const stage = e.target.getStage();
    if (!stage) return;

    const pointer = stage.getPointerPosition();
    if (!pointer) return;

    const direction = e.evt.deltaY < 0 ? 1 : -1;

    setZoom((prevZoom) => {
      const newZoom = direction > 0
        ? Math.min(prevZoom * ZOOM_FACTOR, MAX_ZOOM)
        : Math.max(prevZoom / ZOOM_FACTOR, MIN_ZOOM);

      // Adjust offsets so the point under the cursor stays fixed.
      //   pointer.x = imagePointX * newZoom + newOffsetX
      //   pointer.x = imagePointX * prevZoom + prevOffsetX
      // => imagePointX = (pointer.x - prevOffsetX) / prevZoom
      // => newOffsetX  = pointer.x - imagePointX * newZoom
      const imagePointX = (pointer.x - offsetX) / prevZoom;
      const imagePointY = (pointer.y - offsetY) / prevZoom;

      setOffsetX(pointer.x - imagePointX * newZoom);
      setOffsetY(pointer.y - imagePointY * newZoom);

      return newZoom;
    });
  }, [offsetX, offsetY]);

  /**
   * After a drag gesture completes, record the new position from the layer/stage.
   * The Konva draggable layer reports its absolute position via e.target.
   */
  const handleDragEnd = useCallback((e: Konva.KonvaEventObject<DragEvent>) => {
    setOffsetX(e.target.x());
    setOffsetY(e.target.y());
  }, []);

  /**
   * Compute and apply the scale + offset that centres the image inside the
   * container with "object-fit: contain" behaviour.
   */
  const fitToScreen = useCallback(
    (
      imageWidth: number,
      imageHeight: number,
      containerWidth: number,
      containerHeight: number,
    ) => {
      const { scale, offsetX: ox, offsetY: oy } = fitScale(
        imageWidth,
        imageHeight,
        containerWidth,
        containerHeight,
      );
      setZoom(scale);
      setOffsetX(ox);
      setOffsetY(oy);
    },
    [],
  );

  return { zoom, offsetX, offsetY, handleWheel, handleDragEnd, fitToScreen };
}
