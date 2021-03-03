module GA

    using Statistics, Random

    export
        SimpleGA,
        optimize!

    abstract type EvolutionaryAlgorithm end
    abstract type GeneticAlgorithm <: EvolutionaryAlgorithm end

    mutable struct SimpleGA <: GeneticAlgorithm
        n_iter::Integer
        fitness
        terminal_c::AbstractFloat
        crossover_p::AbstractFloat
        mutation_p::AbstractFloat
        uniform_p::AbstractFloat

        population::Vector{Vector{Bool}}
        best_chro::Vector{Bool}
        iter_n::Integer

        function SimpleGA(;
                n_pop::Integer=100,
                n_genes::Integer=10,
                n_iter::Integer=500,
                fitness,
                terminal_c::AbstractFloat=0.01,
                crossover_p::AbstractFloat=0.5,
                mutation_p::AbstractFloat=0.1,
                uniform_p::AbstractFloat=0.5)
            new(n_iter, fitness, terminal_c, crossover_p, mutation_p, uniform_p,
                [rand(Bool, n_genes) for i ∈ 1:n_pop], zeros(Bool, n_genes), 1)
        end
    end

    function optimize!(ga::GeneticAlgorithm; stop_bybest::Bool=false, stop_bystd::Bool=false)
        for i ∈ ga.iter_n:ga.n_iter
            ga.iter_n = i
            old_pop = shuffle(copy(ga.population))
            new_pop = []
            while !isempty(old_pop)
                ind1 = pop!(old_pop)
                ind2 = pop!(old_pop)
                n_ind1 = copy(ind1)
                n_ind2 = copy(ind2)
                # crossover
                if rand() < ga.crossover_p
                    if rand() < ga.uniform_p
                        for j ∈ 1:length(ind1)
                            if rand() < 0.5
                                n_ind1[j] = ind2[j]
                                n_ind2[j] = ind1[j]
                            end
                        end
                    else
                        c_point = rand(1:length(ind1))
                        n_ind1[c_point:end] = ind2[c_point:end]
                        n_ind2[c_point:end] = ind1[c_point:end]
                    end
                end
                # mutation
                if rand() < ga.mutation_p
                    m_point = rand(1:length(ind1))
                    n_ind1[m_point] = !n_ind1[m_point]
                    n_ind2[m_point] = !n_ind2[m_point]
                end
                push!(new_pop, n_ind1)
                push!(new_pop, n_ind2)
            end
            ga.population = sort([ga.population..., new_pop...], by=ga.fitness)[1:length(ga.population)]
            n_best_chro = ga.population[argmin(broadcast(ga.fitness, ga.population))]
            if ga.fitness(n_best_chro) < ga.fitness(ga.best_chro)
                ga.best_chro = n_best_chro
            elseif stop_bybest
                break
            end
            if stop_bystd && std(ga.fitness.(ga.population)) < ga.terminal_c
                break
            end
        end
    end
end