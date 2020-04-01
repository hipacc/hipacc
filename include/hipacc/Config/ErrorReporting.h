#pragma once

#include <llvm/Support/raw_ostream.h>
#include <clang/Basic/Diagnostic.h>

#include <string>

namespace clang {
namespace hipacc {
    inline void hipacc_check(bool condition, char const* error_msg)
    {
        if(condition)
            return;

        llvm::errs() << "HIPACC WARNING: " << error_msg;
    }

    inline void hipacc_check(bool condition, std::string const& error_msg)
    {
        if(condition)
            return;

        llvm::errs() << "HIPACC WARNING: " << error_msg;
    }

    template <unsigned N>
    void hipacc_require(bool condition, const char (&error_msg)[N], DiagnosticsEngine* diag_engine = nullptr, clang::SourceLocation src_loc = clang::SourceLocation{})
    {
        if(condition)
            return;

        if(diag_engine != nullptr)
        {
            auto DiagIDConstant = diag_engine->getCustomDiagID(DiagnosticsEngine::Error, error_msg);

            if(src_loc.isValid())
                diag_engine->Report(src_loc, DiagIDConstant);

            else diag_engine->Report(DiagIDConstant);
        }

        else
        {
            llvm::errs() << "HIPACC ERROR: " << error_msg;
            exit(EXIT_FAILURE);
        }
    }
}
}