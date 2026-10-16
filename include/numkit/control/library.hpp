/// @file library.hpp
/// @ingroup group_control
// toolboxes/control/include/nk/control/library.hpp
//
// Control-system builtins — function-form only. LTI systems are plain
// nk struct values with a `kind` tag ('tf' / 'zpk' / 'ss'),
// matching the user's "supercalculator, no OOP" rule.

#pragma once

namespace numkit {

/// @addtogroup group_control
/// @{
class PluginContext;

class ControlLibrary
{
public:
    static void install(PluginContext &engine);
};


/// @}
} // namespace numkit
