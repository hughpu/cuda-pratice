
file(GLOB srcs CONFIGURE_DEPENDS src/*.cu include/cudapractice/*.cuh)

add_executable(main ${srcs})
target_link_libraries(main
    PUBLIC
        cublas
)
target_include_directories(main PUBLIC include)
target_compile_options(main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()