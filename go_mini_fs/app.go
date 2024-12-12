package main

import (
    "fmt"
    "net/http"
    "os"
)

// enableCORS wraps an http.Handler and adds CORS headers to the response
func enableCORS(handler http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        
        // Add CORS headers
        w.Header().Set("Access-Control-Allow-Origin", "*")  // Allow all origins
        w.Header().Set("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

        // Handle preflight requests
        if r.Method == "OPTIONS" {
            w.WriteHeader(http.StatusOK)
            return
        }

        // Serve the file
        handler.ServeHTTP(w, r)
    })
}

func main() {
    // Replace "." with the actual path of the directory you want to expose.
    directoryPath := "/data"

    // Check if the directory exists
    _, err := os.Stat(directoryPath)
    if os.IsNotExist(err) {
        fmt.Printf("Directory '%s' not found.\n", directoryPath)
        return
    }

    // Create a file server handler to serve the directory's contents
    fileServer := http.FileServer(http.Dir(directoryPath))

    // Wrap the file server with CORS support
    corsFileServer := enableCORS(fileServer)

    // Create a new HTTP server and handle requests
    http.Handle("/", corsFileServer)

    // Start the server on port 8080
    port := 6193
    fmt.Printf("Server started at http://localhost:%d\n", port)
    err = http.ListenAndServe(fmt.Sprintf(":%d", port), nil)
    if err != nil {
        fmt.Printf("Error starting server: %s\n", err)
    }
}
