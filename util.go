package main

import (
	"runtime"
	"sync"
)

// parallelRows splits [0, n) into chunks processed concurrently.
func parallelRows(n int, fn func(y0, y1 int)) {
	workers := runtime.NumCPU()
	if n < 2*workers || n < 16 {
		fn(0, n)
		return
	}
	chunk := (n + workers*4 - 1) / (workers * 4)
	var wg sync.WaitGroup
	var mu sync.Mutex
	var firstPanic interface{}
	sem := make(chan struct{}, workers)
	for y := 0; y < n; y += chunk {
		y0, y1 := y, mini(n, y+chunk)
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			defer func() {
				if p := recover(); p != nil {
					mu.Lock()
					if firstPanic == nil {
						firstPanic = p
					}
					mu.Unlock()
				}
			}()
			fn(y0, y1)
		}()
	}
	wg.Wait()
	if firstPanic != nil {
		panic(firstPanic) // re-raised in the caller's goroutine so it can be reported per file
	}
}
