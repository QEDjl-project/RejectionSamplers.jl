function build_backend_name(backend_arg::String)
    return if startswith(backend_arg, "--")
        chop(backend_arg, head = 2, tail = 0)
    else
        throw(
            ArgumentError(
                "Given string is not a command line argument!"
            )
        )
    end
end
