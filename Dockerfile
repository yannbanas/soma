# ── Stage 1: Build ──────────────────────────────────────────────────
FROM rust:1.85-alpine AS builder

RUN apk add --no-cache musl-dev

WORKDIR /build
COPY . .

RUN cargo build --release -p soma-cli

# ── Stage 2: Runtime ────────────────────────────────────────────────
FROM alpine:3.21

RUN apk add --no-cache ca-certificates

COPY --from=builder /build/target/release/soma /usr/local/bin/soma

# Default data directory
ENV SOMA_DATA_DIR=/data
VOLUME ["/data"]

# MCP (stdio) + REST API
EXPOSE 8080 4242

# Default: daemon with REST API
CMD ["soma", "daemon", "--http", "8080"]
