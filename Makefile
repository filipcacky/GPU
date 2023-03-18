compdb:
	ninja -C build -t compdb > compile_commands.json

clean:
	rm -rf build

generate:
	cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release

build:
	ninja -C build -v

run_cpu:
	time ./build/apps/cli/cli -l data/0000027_ex5_A.mtx -r data/0000027_ex5_b.npy -c false

run_gpu:
	time ./build/apps/cli/cli -l data/0000027_ex5_A.mtx -r data/0000027_ex5_b.npy -c true

.PHONY: compdb clean generate build run_cpu run_gpu bench_cpu bench_gpu gdb
