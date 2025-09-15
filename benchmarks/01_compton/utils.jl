build_backend_name(backend_arg::String) = startswith(backend_arg, "--") ? chop(backend_arg, head = 2, tail = 0) : trow(ArgumentError("Given string is not an command line argument!"))
