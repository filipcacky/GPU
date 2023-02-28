compdb:
	ninja -C build -t compdb > compile_commands.json

clean:
	rm -rf build

generate: clean
	mkdir build
	cmake -B build -GNinja -DCMAKE_BUILD_TYPE=ReleaseWithDebugSymbols

build:
	ninja -C build

run:
	./build/apps/cli/cli -l data/lhs_small.npy -r data/rhs_small.npy -o data/small_out.npy

bench:
	./build/apps/cli/cli -l data/lhs_10000.npy -r data/rhs_10000.npy -o data/10000_out.npy

.PHONY: compdb clean generate build run bench
