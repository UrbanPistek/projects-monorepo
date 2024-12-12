# Dockerfile.distroless
FROM golang:1.18-bullseye AS base

COPY app.go .

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o /app app.go

FROM alpine:latest

COPY --from=base /app .

CMD ["./app"]
