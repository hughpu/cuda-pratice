
file(GLOB srcs CONFIGURE_DEPENDS src/*.cu include/cudapractice/*.cuh)

add_executable(main ${srcs})
target_include_directories(main PUBLIC include)
target_compile_options(main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)