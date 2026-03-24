import { useState, useEffect } from 'react';

export interface ImageLoadState {
  image: HTMLImageElement | null;
  width: number;
  height: number;
  loading: boolean;
  error: string | null;
}

export function useImageLoad(url: string | null): ImageLoadState {
  const [state, setState] = useState<ImageLoadState>({
    image: null,
    width: 0,
    height: 0,
    loading: false,
    error: null,
  });

  useEffect(() => {
    if (!url) {
      setState({ image: null, width: 0, height: 0, loading: false, error: null });
      return;
    }

    setState({ image: null, width: 0, height: 0, loading: true, error: null });

    let cancelled = false;
    const img = new Image();

    img.onload = () => {
      if (cancelled) return;
      setState({
        image: img,
        width: img.naturalWidth,
        height: img.naturalHeight,
        loading: false,
        error: null,
      });
    };

    img.onerror = () => {
      if (cancelled) return;
      setState({
        image: null,
        width: 0,
        height: 0,
        loading: false,
        error: `Failed to load image: ${url}`,
      });
    };

    img.src = url;

    return () => {
      cancelled = true;
      img.onload = null;
      img.onerror = null;
    };
  }, [url]);

  return state;
}
