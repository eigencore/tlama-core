from packaging.version import Version as TrueVersion


def _get_dtype(dtype):
    pass

def Version(version):
    try:
        return TrueVersion(version)
    except:
        from inspect import getframeinfo, stack
        caller = getframeinfo(stack()[1][0])
        raise RuntimeError(
            f"Tlama-Core: Could not get version for `{version}`\n"\
            f"File name = [{caller.filename}] Line number = [{caller.lineno}]"
        )