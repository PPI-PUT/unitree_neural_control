// Copyright 2023 Maciej Krupka
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef UNITREE_A1_NEURAL_CONTROL__VISIBILITY_CONTROL_HPP_
#define UNITREE_A1_NEURAL_CONTROL__VISIBILITY_CONTROL_HPP_

////////////////////////////////////////////////////////////////////////////////
#if defined(__WIN32)
  #if defined(UNITREE_A1_NEURAL_CONTROL_BUILDING_DLL) || defined(UNITREE_A1_NEURAL_CONTROL_EXPORTS)
    #define UNITREE_A1_NEURAL_CONTROL_PUBLIC __declspec(dllexport)
    #define UNITREE_A1_NEURAL_CONTROL_LOCAL
  #else  // defined(UNITREE_A1_NEURAL_CONTROL_BUILDING_DLL) || defined(UNITREE_A1_NEURAL_CONTROL_EXPORTS)
    #define UNITREE_A1_NEURAL_CONTROL_PUBLIC __declspec(dllimport)
    #define UNITREE_A1_NEURAL_CONTROL_LOCAL
  #endif  // defined(UNITREE_A1_NEURAL_CONTROL_BUILDING_DLL) || defined(UNITREE_A1_NEURAL_CONTROL_EXPORTS)
#elif defined(__linux__)
  #define UNITREE_A1_NEURAL_CONTROL_PUBLIC __attribute__((visibility("default")))
  #define UNITREE_A1_NEURAL_CONTROL_LOCAL __attribute__((visibility("hidden")))
#elif defined(__APPLE__)
  #define UNITREE_A1_NEURAL_CONTROL_PUBLIC __attribute__((visibility("default")))
  #define UNITREE_A1_NEURAL_CONTROL_LOCAL __attribute__((visibility("hidden")))
#else
  #error "Unsupported Build Configuration"
#endif

#endif  // UNITREE_A1_NEURAL_CONTROL__VISIBILITY_CONTROL_HPP_
