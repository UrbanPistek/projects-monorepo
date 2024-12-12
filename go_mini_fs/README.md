# file-server

Build & Run Application
```
go build app.go
./app
```

Build & Run with Docker & attach volume to access files via volume:
```
docker build -t go-fs-mini .
docker run --rm --name go-fs-mini -v ./files:/data -p 6191:6193 -d go-fs-mini

Build & Push to dockerhub
```
docker build -t urbanpistek/go-mini-fs .
docker push urbanpistek/go-mini-fs:latest
```
