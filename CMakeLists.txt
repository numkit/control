cmake_minimum_required(VERSION 3.20)
project(numkit_control CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(NUMKIT_WITH_SIMD "Enable Highway SIMD acceleration" ON)
option(NUMKIT_BUILD_TESTS "Build GoogleTest suite" OFF)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(_NUMKIT_TOP_BUILD_TESTS ${NUMKIT_BUILD_TESTS})
endif()

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
    if(MSVC AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../signal/build/Release/numkit_signal_core.lib")
        add_library(numkit_signal_core STATIC IMPORTED)
        set_target_properties(numkit_signal_core PROPERTIES
            IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../signal/build/Release/numkit_signal_core.lib"
            INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../signal/include"
            INTERFACE_LINK_LIBRARIES "numkit_value;numkit_sys;numkit_ops"
        )
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../signal/CMakeLists.txt")
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

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(NUMKIT_BUILD_TESTS ${_NUMKIT_TOP_BUILD_TESTS} CACHE BOOL "" FORCE)
endif()

if(NUMKIT_BUILD_TESTS AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    enable_testing()
    if(NOT TARGET numkit)
        if(MSVC AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build/Release/numkit.lib")
            add_library(numkit STATIC IMPORTED)
            set_target_properties(numkit PROPERTIES
                IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build/Release/numkit.lib"
                INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/include"
                INTERFACE_COMPILE_DEFINITIONS "NUMKIT_CORE_BUILD=1"
                INTERFACE_LINK_LIBRARIES "numkit_value;numkit_sys;numkit_ops"
            )
        else()
            set(NUMKIT_BUILD_CLI OFF CACHE BOOL "" FORCE)
            set(NUMKIT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
            if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/CMakeLists.txt")
                add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../numkit" _deps/numkit)
            else()
                FetchContent_Declare(
                    numkit_core
                    GIT_REPOSITORY https://github.com/numkit/numkit.git
                    GIT_TAG main
                )
                FetchContent_MakeAvailable(numkit_core)
            endif()
            set(NUMKIT_BUILD_TESTS ON CACHE BOOL "" FORCE)
        endif()
    endif()

    if(NOT TARGET GTest::gtest_main)
        if(MSVC AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build/lib/Release/gtest_main.lib")
            add_library(GTest::gtest STATIC IMPORTED)
            set_target_properties(GTest::gtest PROPERTIES
                IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build/lib/Release/gtest.lib"
                INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build/_deps/googletest-src/googletest/include"
            )
            add_library(GTest::gtest_main STATIC IMPORTED)
            set_target_properties(GTest::gtest_main PROPERTIES
                IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build/lib/Release/gtest_main.lib"
                INTERFACE_LINK_LIBRARIES "GTest::gtest"
            )
        else()
            set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
            set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
            set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
            if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build/_deps/googletest-src/CMakeLists.txt")
                set(FETCHCONTENT_SOURCE_DIR_GOOGLETEST "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build/_deps/googletest-src" CACHE PATH "" FORCE)
            endif()
            FetchContent_Declare(
                googletest
                GIT_REPOSITORY https://github.com/google/googletest.git
                GIT_TAG v1.14.0
            )
            FetchContent_MakeAvailable(googletest)
        endif()
    endif()

    file(GLOB CONTROL_TEST_SOURCES "tests/*_test.cpp")
    add_executable(numkit_control_test ${CONTROL_TEST_SOURCES})
    target_include_directories(numkit_control_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_link_libraries(numkit_control_test PRIVATE
        numkit_control_core
        numkit
        GTest::gtest_main
    )
    if(NUMKIT_WITH_SIMD)
        target_link_libraries(numkit_control_test PRIVATE hwy hwy_contrib)
    endif()
    add_dependencies(numkit_control_test numkit_control)
    if(MSVC)
        target_compile_options(numkit_control_test PRIVATE /W3 /MP /bigobj)
    endif()
    add_test(NAME numkit_control_test COMMAND numkit_control_test)
endif()
