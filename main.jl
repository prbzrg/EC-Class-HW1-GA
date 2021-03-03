using Statistics
using Plots

include("ga.jl")

using .GA

str_toint(s) = parse(Int, join(Int.(s)), base=2)

fitness(x::Integer) = 20 * (cos(x/3 + 1)/(x/3 + 1))
fitness(x::Vector{Bool}) = fitness(str_toint(x))

x = 0:63
y = fitness.(x)

@show x[argmin(y)]
@show y[argmin(y)]

ff_plt = plot(
    x,
    y,
    label=nothing,
    title="Fitness Function",
    size=(1024, 512),
)
scatter!(x, y, label=nothing)
savefig(ff_plt, "figs/f_f_fig.png")

let quz="uniform-1", n_pop=8, n_genes=ceil(Integer, log2(63)), uniform_p=1.0, n_repeat=100, stop_bybest=true
    ga01_his = SimpleGA[]
    ga05_his = SimpleGA[]
    ga10_his = SimpleGA[]
    ga50_his = SimpleGA[]

    for i in 1:n_repeat
        ga01 = SimpleGA(
            n_pop=n_pop,
            n_genes=n_genes,
            fitness=fitness,
            mutation_p=0.01,
            uniform_p=uniform_p,
        )
        optimize!(ga01, stop_bybest=stop_bybest)
        push!(ga01_his, ga01)

        ga05 = SimpleGA(
            n_pop=n_pop,
            n_genes=n_genes,
            fitness=fitness,
            mutation_p=0.05,
        )
        optimize!(ga05, stop_bybest=stop_bybest)
        push!(ga05_his, ga05)

        ga10 = SimpleGA(
            n_pop=n_pop,
            n_genes=n_genes,
            fitness=fitness,
            mutation_p=0.1,
        )
        optimize!(ga10, stop_bybest=stop_bybest)
        push!(ga10_his, ga10)

        ga50 = SimpleGA(
            n_pop=n_pop,
            n_genes=n_genes,
            fitness=fitness,
            mutation_p=0.5,
        )
        optimize!(ga50, stop_bybest=stop_bybest)
        push!(ga50_his, ga50)
    end

    mean_f(his) = mean(fitness.(broadcast(x -> x.best_chro, his)))
    var_f(his) = var(fitness.(broadcast(x -> x.best_chro, his)))

    mean_i(his) = mean(broadcast(x -> x.iter_n, his))
    var_i(his) = var(broadcast(x -> x.iter_n, his))

    fgm_c(his) = sum(broadcast(x -> str_toint(x.best_chro) == 5, his))

    mut_plt_gm = plot(
        [1, 5, 10, 50],
        fgm_c.([ga01_his, ga05_his, ga10_his, ga50_his]),
        label=nothing,
        title="Mutation Change - Global Minimum Count",
        size=(1024, 512),
    )
    savefig(mut_plt_gm, "figs/$(quz)_mut_gm_fig.png")

    mut_plt_f_m = plot(
        [1, 5, 10, 50],
        mean_f.([ga01_his, ga05_his, ga10_his, ga50_his]),
        label=nothing,
        title="Mutation Change - Fitness Mean",
        size=(1024, 512),
    )
    savefig(mut_plt_f_m, "figs/$(quz)_mut_f_m_fig.png")
    mut_plt_f_v = plot(
        [1, 5, 10, 50],
        var_f.([ga01_his, ga05_his, ga10_his, ga50_his]),
        label=nothing,
        title="Mutation Change - Fitness Variance",
        size=(1024, 512),
    )
    savefig(mut_plt_f_v, "figs/$(quz)_mut_f_v_fig.png")

    mut_plt_i_m = plot(
        [1, 5, 10, 50],
        mean_i.([ga01_his, ga05_his, ga10_his, ga50_his]),
        label=nothing,
        title="Mutation Change - N.Iterations Mean",
        size=(1024, 512),
    )
    savefig(mut_plt_i_m, "figs/$(quz)_mut_i_m_fig.png")
    mut_plt_i_v = plot(
        [1, 5, 10, 50],
        var_i.([ga01_his, ga05_his, ga10_his, ga50_his]),
        label=nothing,
        title="Mutation Change - N.Iterations Variance",
        size=(1024, 512),
    )
    savefig(mut_plt_i_v, "figs/$(quz)_mut_i_v_fig.png")
end

let quz="uniform-0", n_pop=8, n_genes=ceil(Integer, log2(63)), uniform_p=0.0, n_repeat=100, stop_bybest=true
    ga01_his = SimpleGA[]
    ga05_his = SimpleGA[]
    ga10_his = SimpleGA[]
    ga50_his = SimpleGA[]

    for i in 1:n_repeat
        ga01 = SimpleGA(
            n_pop=n_pop,
            n_genes=n_genes,
            fitness=fitness,
            mutation_p=0.01,
            uniform_p=uniform_p,
        )
        optimize!(ga01, stop_bybest=stop_bybest)
        push!(ga01_his, ga01)

        ga05 = SimpleGA(
            n_pop=n_pop,
            n_genes=n_genes,
            fitness=fitness,
            mutation_p=0.05,
        )
        optimize!(ga05, stop_bybest=stop_bybest)
        push!(ga05_his, ga05)

        ga10 = SimpleGA(
            n_pop=n_pop,
            n_genes=n_genes,
            fitness=fitness,
            mutation_p=0.1,
        )
        optimize!(ga10, stop_bybest=stop_bybest)
        push!(ga10_his, ga10)

        ga50 = SimpleGA(
            n_pop=n_pop,
            n_genes=n_genes,
            fitness=fitness,
            mutation_p=0.5,
        )
        optimize!(ga50, stop_bybest=stop_bybest)
        push!(ga50_his, ga50)
    end

    mean_f(his) = mean(fitness.(broadcast(x -> x.best_chro, his)))
    var_f(his) = var(fitness.(broadcast(x -> x.best_chro, his)))

    mean_i(his) = mean(broadcast(x -> x.iter_n, his))
    var_i(his) = var(broadcast(x -> x.iter_n, his))

    fgm_c(his) = sum(broadcast(x -> str_toint(x.best_chro) == 5, his))

    mut_plt_gm = plot(
        [1, 5, 10, 50],
        fgm_c.([ga01_his, ga05_his, ga10_his, ga50_his]),
        label=nothing,
        title="Mutation Change - Global Minimum Count",
        size=(1024, 512),
    )
    savefig(mut_plt_gm, "figs/$(quz)_mut_gm_fig.png")

    mut_plt_f_m = plot(
        [1, 5, 10, 50],
        mean_f.([ga01_his, ga05_his, ga10_his, ga50_his]),
        label=nothing,
        title="Mutation Change - Fitness Mean",
        size=(1024, 512),
    )
    savefig(mut_plt_f_m, "figs/$(quz)_mut_f_m_fig.png")
    mut_plt_f_v = plot(
        [1, 5, 10, 50],
        var_f.([ga01_his, ga05_his, ga10_his, ga50_his]),
        label=nothing,
        title="Mutation Change - Fitness Variance",
        size=(1024, 512),
    )
    savefig(mut_plt_f_v, "figs/$(quz)_mut_f_v_fig.png")

    mut_plt_i_m = plot(
        [1, 5, 10, 50],
        mean_i.([ga01_his, ga05_his, ga10_his, ga50_his]),
        label=nothing,
        title="Mutation Change - N.Iterations Mean",
        size=(1024, 512),
    )
    savefig(mut_plt_i_m, "figs/$(quz)_mut_i_m_fig.png")
    mut_plt_i_v = plot(
        [1, 5, 10, 50],
        var_i.([ga01_his, ga05_his, ga10_his, ga50_his]),
        label=nothing,
        title="Mutation Change - N.Iterations Variance",
        size=(1024, 512),
    )
    savefig(mut_plt_i_v, "figs/$(quz)_mut_i_v_fig.png")
end
