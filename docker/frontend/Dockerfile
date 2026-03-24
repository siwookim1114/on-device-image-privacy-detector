# =============================================================================
# Frontend Dockerfile — On-Device Image Privacy Detector
# =============================================================================
#
# Two-stage build:
#   Stage 1 (builder): Node 20 — installs deps, runs `vite build`
#   Stage 2 (serve):   nginx:alpine — serves the static /dist at port 80
#
# The backend API base URL is injected at build time via ARG/ENV so the JS
# bundle contains the correct address.  If you need to point at a different
# backend, pass --build-arg VITE_API_BASE_URL=... to `docker compose build`.
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: build
# ---------------------------------------------------------------------------
FROM node:20-alpine AS builder

WORKDIR /app

# Build args — baked into the Vite bundle
ARG VITE_API_BASE_URL=http://localhost:8000/api/v1
ARG VITE_WS_BASE_URL=ws://localhost:8000

ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
ENV VITE_WS_BASE_URL=$VITE_WS_BASE_URL

# Install dependencies first (cached layer)
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts

# Copy source and build
COPY . .
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2: serve
# ---------------------------------------------------------------------------
FROM nginx:1.27-alpine AS serve

# Remove the default nginx static files
RUN rm -rf /usr/share/nginx/html/*

# Copy the Vite build output
COPY --from=builder /app/dist /usr/share/nginx/html

# Custom nginx config:
#   - Serve SPA — all unknown paths return index.html (React Router)
#   - Proxy /api/v1 and /ws to the backend container
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
