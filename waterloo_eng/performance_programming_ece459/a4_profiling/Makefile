PERF_FLAGS := -F 1000 -g --call-graph dwarf

# EXEC_FLAGS := 80 2 4000 6 6
EXEC_FLAGS := 80 8 10000 12 12
# EXEC_FLAGS := 5 1 10 1 1

flamegraph:
	cargo flamegraph -c "record $(PERF_FLAGS)" -- $(EXEC_FLAGS)
