if abspath(PROGRAM_FILE) == @__FILE__
    if !haskey(ENV, "CI_COMMIT_REF_NAME")
        @error "Environment variable CI_COMMIT_REF_NAME is not set."
        exit(1)
    end

    ci_commit_ref_name = ENV["CI_COMMIT_REF_NAME"]

    user = ""
    branch = ""
    pr_number = 0

    if startswith(ci_commit_ref_name, "pr-")
        split_commit_ref_name = split(ci_commit_ref_name, "/")

        if length(split_commit_ref_name) < 3
            throw(ErrorException("ci_commit_ref_name cannot be spited in 3 or more components."))
        end

        for component in split_commit_ref_name
            if component == ""
                throw(ErrorException("one of the components of ci_commit_ref_name is empty."))
            end
        end

        if (!startswith(split_commit_ref_name[1], "pr-"))
            throw(ErrorException("ci_commit_ref_name needs to start with `pr-`"))
        end

        # parse to Int only to check if it is a number
        pr_number = parse(Int, split_commit_ref_name[1][(length("pr-") + 1):end])
        if (pr_number <= 0)
            throw(
                ErrorException(
                    "a PR number always needs to be a positive integer number bigger than 0: $pr_number",
                )
            )
        end

        user = split_commit_ref_name[2]
        branch = split_commit_ref_name[4]
    else
        user = "QEDjl-project"
        branch = ci_commit_ref_name
    end

    pr_str = ""
    if pr_number > 0
        pr_str = "--pr $(pr_number)"
    end

    commit_sha=strip(read(`git rev-parse HEAD`, String))

    for action in ["create-commit", "create-report", "do-upload"]
        codecov_cmd = "./codecov -v $(action) --git-service github --branch $(user):$(branch) --sha $commit_sha $pr_str"
        println(codecov_cmd)
    end
end
