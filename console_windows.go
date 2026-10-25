//go:build windows

package main

import (
	"syscall"
	"unsafe"
)

// launchedFromExplorer reports whether the program owns its console window alone, which is
// the case when it was started by double-click or drag & drop rather than from a terminal.
func launchedFromExplorer() bool {
	proc := syscall.NewLazyDLL("kernel32.dll").NewProc("GetConsoleProcessList")
	if proc.Find() != nil {
		return false
	}
	var pids [4]uint32
	n, _, _ := proc.Call(uintptr(unsafe.Pointer(&pids[0])), 4)
	return n == 1
}
