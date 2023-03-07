add_subdirectory(libnpy/)

include(FetchContent)
include(CheckLanguage)

FetchContent_Declare(
  Thrust
  GIT_REPOSITORY https://github.com/NVIDIA/thrust.git
  GIT_TAG cuda-11.6
  )

FetchContent_MakeAvailable(Thrust)

thrust_create_target(ThrustCUDA HOST CPP DEVICE CUDA)

