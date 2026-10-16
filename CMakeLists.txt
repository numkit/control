cmake_minimum_required(VERSION 3.20)
project(numkit_control CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(NUMKIT_WITH_SIMD "Enable Highway SIMD acceleration" ON)

include(FetchContent)

# Dependency: numkit/value
if(NOT TARGET numkit_value)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../value/CMakeLists.txt")
        message(STATUS "Using local numkit/value from ${CMAKE_CURRENT_SOURCE_DIR}/../value")
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../value" _deps/numkit_value)
    else()
        FetchContent_Declare(
            numkit_value
            GIT_REPOSITORY https://github.com/numkit/value.git
            GIT_TAG main
        )
        FetchContent_MakeAvailable(numkit_value)
    endif()
endif()

# Dependency: numkit/sys
if(NOT TARGET numkit_sys)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../sys/CMakeLists.txt")
        message(STATUS "Using local numkit/sys from ${CMAKE_CURRENT_SOURCE_DIR}/../sys")
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../sys" _deps/numkit_sys)
    else()
        FetchContent_Declare(
            numkit_sys
            GIT_REPOSITORY https://github.com/numkit/sys.git
            GIT_TAG main
        )
        FetchContent_MakeAvailable(numkit_sys)
    endif()
endif()

# Dependency: numkit/ops
if(NOT TARGET numkit_ops)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../ops/CMakeLists.txt")
        message(STATUS "Using local numkit/ops from ${CMAKE_CURRENT_SOURCE_DIR}/../ops")
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../ops" _deps/numkit_ops)
    else()
        FetchContent_Declare(
            numkit_ops
            GIT_REPOSITORY https://github.com/numkit/ops.git
            GIT_TAG main
        )
        FetchContent_MakeAvailable(numkit_ops)
    endif()
endif()

# Dependency: numkit/signal
if(NOT TARGET numkit_signal_core)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../signal/CMakeLists.txt")
        message(STATUS "Using local numkit/signal from ${CMAKE_CURRENT_SOURCE_DIR}/../signal")
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../signal" _deps/numkit_signal)
    else()
        FetchContent_Declare(
            numkit_signal
            GIT_REPOSITORY https://github.com/numkit/signal.git
            GIT_TAG main
        )
        FetchContent_MakeAvailable(numkit_signal)
    endif()
endif()

file(GLOB_RECURSE CONTROL_COMPUTE_SOURCES
    "src/analyze/*.cpp"
    "src/connect/*.cpp"
    "src/conversion/*.cpp"
    "src/discretize/*.cpp"
    "src/freq/*.cpp"
    "src/internal/*.cpp"
    "src/lti/*.cpp"
    "src/lyapunov/*.cpp"
    "src/props/*.cpp"
    "src/response/*.cpp"
    "src/riccati/*.cpp"
    "src/state/*.cpp"
)

file(GLOB_RECURSE CONTROL_PLUGIN_SOURCES
    "src/register/*.cpp"
    "src/module.cpp"
)

add_library(numkit_control_core STATIC ${CONTROL_COMPUTE_SOURCES})
add_library(numkit::control_core ALIAS numkit_control_core)
set_target_properties(numkit_control_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(numkit_control_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(numkit_control_core PUBLIC numkit_value numkit_sys numkit_ops numkit_signal_core)

if(MSVC)
    target_compile_options(numkit_control_core PRIVATE /W4 /MP /bigobj)
else()
    target_compile_options(numkit_control_core PRIVATE -Wall -Wextra -fPIC)
endif()

# Shared plugin module
add_library(numkit_control SHARED ${CONTROL_PLUGIN_SOURCES})
add_library(numkit::control ALIAS numkit_control)

target_compile_definitions(numkit_control PRIVATE NUMKIT_PLUGIN_BUILD=1)

target_include_directories(numkit_control
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(numkit_control PUBLIC numkit_control_core numkit_value numkit_sys numkit_ops numkit_signal_core)

if(MSVC)
    target_compile_options(numkit_control PRIVATE /W4 /MP /bigobj)
else()
    target_compile_options(numkit_control PRIVATE -Wall -Wextra -fPIC)
endif()
