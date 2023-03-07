compdb:
	ninja -C build -t compdb > compile_commands.json

clean:
	rm -rf build

generate:
	cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release

build:
	ninja -C build -v

run_cpu:
	time ./build/apps/cli/cli -l data/lhs_20.npy -r data/rhs_20.npy -o data/out.npy -c false

run_gpu:
	time ./build/apps/cli/cli -l data/lhs_20.npy -r data/rhs_20.npy -o data/out.npy -c true

bench_cpu:
	time ./build/apps/cli/cli -l data/lhs_20200.txt -r data/rhs_20200.txt -o data/out.npy -c false

bench_gpu:
	time ./build/apps/cli/cli -l data/lhs_20200.txt -r data/rhs_20200.txt -o data/out.npy -c true

bench_cpu2:
	time ./build/apps/cli/cli -l data/lhs_small.npy -r data/rhs_small.npy -o data/out.npy -c false

bench_gpu2:
	time ./build/apps/cli/cli -l data/lhs_small.npy -r data/rhs_small.npy -o data/out.npy -c true

gdb:
	cuda-gdb --args ./build/apps/cli/cli -l data/lhs_20.npy -r data/rhs_20.npy -o data/out.npy -c true

.PHONY: compdb clean generate build run_cpu run_gpu bench_cpu bench_gpu gdb
