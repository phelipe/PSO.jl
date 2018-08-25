module PSO
    using StaticArrays 
    
    function make_constraints(::Type{Val{nothing}}, args, kwargs, verbose)
        verbose && println("No constraints given.")
        return x -> [0.0]
    end

    function make_constraints(eqs::Vector, args, kwargs, verbose)
        verbose && println("Converting ieqcons to a single constraint function.")
        return x -> [f(x, args...; kwargs...) for f in eqs]
    end

    function make_constraints(eqs::Function, args, kwargs, verbose)
        verbose && println("Single constraint function given in f_ieqcons.")
        return x -> eqs(x, args...; kwargs...)
    end

    make_constraints(eqs, args, kwargs, verbose) = make_constraints(Val{eqs}, args, kwargs, verbose)

    function update_position!(x, p, fx, fp, fs)
        i_update = (fx .< fp) .& fs
        p[i_update] = copy(x[i_update])
        fp[i_update] = fx[i_update]
    end

    function pso(func::Function, lb::Vector, ub::Vector, constraints, args, kwargs,
                 swarmsize, ω, ϕp, ϕg, maxiter, minstep, minfunc, verbose)
        assert(length(ub) == length(lb))
        assert(all(ub .> lb))

        obj = x -> func(x, args...; kwargs...)
        cons = make_constraints(constraints, args, kwargs, verbose)
        is_feasible = x -> all(cons(x) .>= 0)

        # Initialize the particle swarm
        S = swarmsize
        D = length(lb) # the number of dimensions each particle has
        ub = SVector{D}(ub)
        lb = SVector{D}(lb)
        vhigh = abs.(ub .- lb)
        vlow = -vhigh
       
        x = [@SVector rand(D) for x in 1:S] # particle positions
        v = [@SVector rand(D) for x in 1:S] # particle velocities
        p = [@SVector zeros(D) for x in 1:S] # best particle positions
        map!(x-> lb .+ x .* (ub .- lb), x, x) # particle positions
        map!(x-> vlow .+ x .* (vhigh .- vlow), v, v)# particle velocities
       
        fx = [obj(x[i]) for i in 1:S]  # current particle function values
        fs = [is_feasible(x[i]) for i = 1:S]  # feasibility of each particle
        fp = ones(S) * Inf  # best particle function values

        g = copy(x)  # best swarm position
        fg = ones(S) * Inf  # best swarm position starting value

        # Store particle's best position (if constraints are satisfied)
        update_position!(x, p, fx, fp, fs)

        # TODO: tenho que fazer o g ser um vetor com as melhores posições e fazer funções para 
        # que estes valores sejam escolhidos conforme a topologia escolhida, vou fazer de uma
        # maneira que seja possível escolher a quantidade de vizinhos.
        # Aqui que vai ficar inicialmente o lance das funções pois o vetor g vai ser diferente
        #dependendo de como o usuário quer.
        # Update swarm's best position
        i_min = indmin(fp)
        if fp[i_min] < fg
            g = copy(p[i_min])
            fg = fp[i_min]
        end

        # TODO: tirei o while e tudo abaixo dele daqui, colocar posteriormente
    end

    function pso(func, lb, ub; constraints=nothing, args=(), kwargs=Dict(), swarmsize=100,
                 omega=0.5, phip=0.5, phig=0.5, maxiter=100, minstep=1e-8, minfunc=1e-8,
                 verbose=false, particle_output=false)
        #g, fg, p, fp = pso(func, lb, ub, constraints, args, kwargs,
        #    swarmsize, omega, phip, phig, maxiter, minstep, minfunc, verbose)
        #return particle_output? (g, fg, p, fp) : (g, fg)
        # TODO: a parte acima está comentada temporariamente para desenvolvimento
        pso(func, lb, ub, constraints, args, kwargs,
            swarmsize, omega, phip, phig, maxiter, minstep, minfunc, verbose)
    end

    export pso;

end # module
