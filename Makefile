compdb:
	ninja -C build -t compdb > compile_commands.json

clean:
	rm -rf build

generate:
	cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release

build:
	ninja -C build -v

run_cpu:
	time ./build/apps/cli/cli -l data/0007102_Kuu_A.mtx -c false -s true

run_gpu:
	time ./build/apps/cli/cli -l data/0007102_Kuu_A.mtx -c true -s true

.PHONY: compdb clean generate build run_cpu run_gpu bench_cpu bench_gpu gdb
