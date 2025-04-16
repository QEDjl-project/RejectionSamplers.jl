using SafeTestsets

begin

    @time @safetestset "Scan Tests" begin
        include("filter_scan.jl")
    end

end
