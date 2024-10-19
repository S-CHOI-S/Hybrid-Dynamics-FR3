# CMake generated Testfile for 
# Source directory: /home/kist-robot2/mujoco-2.3.1_source/mujoco-2.3.1/test
# Build directory: /home/kist-robot2/mujoco-2.3.1_source/mujoco-2.3.1/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(MujocoTestTest.MjUserWarningFailsTest "/home/kist-robot2/mujoco-2.3.1_source/mujoco-2.3.1/bin/fixture_test" "--gtest_filter=MujocoTestTest.MjUserWarningFailsTest")
set_tests_properties(MujocoTestTest.MjUserWarningFailsTest PROPERTIES  WORKING_DIRECTORY "/home/kist-robot2/mujoco-2.3.1_source/mujoco-2.3.1/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.16/Modules/GoogleTest.cmake;353;add_test;/home/kist-robot2/mujoco-2.3.1_source/mujoco-2.3.1/test/CMakeLists.txt;31;gtest_add_tests;/home/kist-robot2/mujoco-2.3.1_source/mujoco-2.3.1/test/CMakeLists.txt;62;mujoco_test;/home/kist-robot2/mujoco-2.3.1_source/mujoco-2.3.1/test/CMakeLists.txt;0;")
add_test(MujocoTestTest.MjUserErrorFailsTest "/home/kist-robot2/mujoco-2.3.1_source/mujoco-2.3.1/bin/fixture_test" "--gtest_filter=MujocoTestTest.MjUserErrorFailsTest")
set_tests_properties(MujocoTestTest.MjUserErrorFailsTest PROPERTIES  WORKING_DIRECTORY "/home/kist-robot2/mujoco-2.3.1_source/mujoco-2.3.1/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.16/Modules/GoogleTest.cmake;353;add_test;/home/kist-robot2/mujoco-2.3.1_source/mujoco-2.3.1/test/CMakeLists.txt;31;gtest_add_tests;/home/kist-robot2/mujoco-2.3.1_source/mujoco-2.3.1/test/CMakeLists.txt;62;mujoco_test;/home/kist-robot2/mujoco-2.3.1_source/mujoco-2.3.1/test/CMakeLists.txt;0;")
add_test(MujocoErrorTestGuardTest.NestedErrorGuards "/home/kist-robot2/mujoco-2.3.1_source/mujoco-2.3.1/bin/fixture_test" "--gtest_filter=MujocoErrorTestGuardTest.NestedErrorGuards")
set_tests_properties(MujocoErrorTestGuardTest.NestedErrorGuards PROPERTIES  WORKING_DIRECTORY "/home/kist-robot2/mujoco-2.3.1_source/mujoco-2.3.1/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.16/Modules/GoogleTest.cmake;353;add_test;/home/kist-robot2/mujoco-2.3.1_source/mujoco-2.3.1/test/CMakeLists.txt;31;gtest_add_tests;/home/kist-robot2/mujoco-2.3.1_source/mujoco-2.3.1/test/CMakeLists.txt;62;mujoco_test;/home/kist-robot2/mujoco-2.3.1_source/mujoco-2.3.1/test/CMakeLists.txt;0;")
add_test(HeaderTest.EnumsAreInts "/home/kist-robot2/mujoco-2.3.1_source/mujoco-2.3.1/bin/header_test" "--gtest_filter=HeaderTest.EnumsAreInts")
set_tests_properties(HeaderTest.EnumsAreInts PROPERTIES  WORKING_DIRECTORY "/home/kist-robot2/mujoco-2.3.1_source/mujoco-2.3.1/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.16/Modules/GoogleTest.cmake;353;add_test;/home/kist-robot2/mujoco-2.3.1_source/mujoco-2.3.1/test/CMakeLists.txt;31;gtest_add_tests;/home/kist-robot2/mujoco-2.3.1_source/mujoco-2.3.1/test/CMakeLists.txt;65;mujoco_test;/home/kist-robot2/mujoco-2.3.1_source/mujoco-2.3.1/test/CMakeLists.txt;0;")
add_test(PipelineTest.SparseDenseEquivalent "/home/kist-robot2/mujoco-2.3.1_source/mujoco-2.3.1/bin/pipeline_test" "--gtest_filter=PipelineTest.SparseDenseEquivalent")
set_tests_properties(PipelineTest.SparseDenseEquivalent PROPERTIES  WORKING_DIRECTORY "/home/kist-robot2/mujoco-2.3.1_source/mujoco-2.3.1/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.16/Modules/GoogleTest.cmake;353;add_test;/home/kist-robot2/mujoco-2.3.1_source/mujoco-2.3.1/test/CMakeLists.txt;31;gtest_add_tests;/home/kist-robot2/mujoco-2.3.1_source/mujoco-2.3.1/test/CMakeLists.txt;68;mujoco_test;/home/kist-robot2/mujoco-2.3.1_source/mujoco-2.3.1/test/CMakeLists.txt;0;")
subdirs("benchmark")
subdirs("engine")
subdirs("sample")
subdirs("user")
subdirs("xml")
subdirs("plugin/elasticity")
