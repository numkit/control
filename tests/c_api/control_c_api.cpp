#include <numkit/control/c_api.h>
#include <gtest/gtest.h>

TEST(NumkitControlCApi, TransferFunctionAndDcgain)
{
    const int64_t base = numkit_debug_live_values();
    numkit_error err{};
    numkit_control_ctx *ctx = numkit_control_ctx_create(&err);
    ASSERT_NE(ctx, nullptr) << err.message;

    double numData[1] = {2.0};
    double denData[3] = {1.0, 3.0, 2.0};
    size_t d1[2] = {1, 1}, d3[2] = {1, 3};
    numkit_val *num = numkit_val_view(numData, NUMKIT_DTYPE_DOUBLE, d1, 2, &err);
    numkit_val *den = numkit_val_view(denData, NUMKIT_DTYPE_DOUBLE, d3, 2, &err);
    const numkit_val *tfArgs[2] = {num, den};
    numkit_val *sys = nullptr;
    ASSERT_EQ(numkit_control_call(ctx, "tf", tfArgs, 2, 1, &sys, &err), 0) << err.message;

    const numkit_fn *dcgainFn = numkit_control_resolve(ctx, "dcgain");
    ASSERT_NE(dcgainFn, nullptr);
    const numkit_val *dcArgs[1] = {sys};
    numkit_val *gain = nullptr;
    ASSERT_EQ(numkit_control_invoke(ctx, dcgainFn, dcArgs, 1, 1, &gain, &err), 0) << err.message;
    EXPECT_NEAR(numkit_val_as_double(gain, &err), 1.0, 1e-12);

    numkit_val_release(gain);
    numkit_val_release(sys);
    numkit_val_release(den);
    numkit_val_release(num);
    numkit_control_ctx_destroy(ctx);
    EXPECT_EQ(numkit_debug_live_values(), base);
}
