module PSO
    
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

    function findBestLocal(data::Vector, bests::Vector ,n::Integer) 
        order = sortperm(data,rev=true)
        order_size = length(order)
        value = Float64[]
        position = Integer[]
        for i = 1:order_size
            aux = mod.(collect(i-n:i+n),order_size)
            aux = map(x-> x==0 ? order_size : x, aux)
            position_best = order[aux]
            out = findmin(bests[position_best])
            push!(value,out[1])
            push!(position,position_best[out[2]])
        end
        value, position    
    end

    function findBestGlobal(data::Vector, bests::Vector ,n::Integer = 0)
        aux = findmin(bests)
        aux2 = ones(Integer,length(data))
        value = aux2 .* aux[1]
        position = aux2 .* aux[2]
        value, position
    end

    function pso(func::Function, lb::Vector, ub::Vector, constraints, args, kwargs,
                 swarmsize, ω, ϕp, ϕg, maxiter, minstep, minfunc, verbose, 
                 neighborhood, n)

        obj = x -> func(x, args...; kwargs...)
        cons = make_constraints(constraints, args, kwargs, verbose)
        is_feasible = x -> all(cons(x) .>= 0)

        # Initialize the particle swarm
        S = swarmsize
        D = length(lb) # the number of dimensions each particle has
        vhigh = abs.(ub .- lb)
        vlow = -vhigh
       
        x = [rand(D) for x in 1:S] # particle positions
        v = [rand(D) for x in 1:S] # particle velocities
        p = [zeros(D) for x in 1:S] # best particle positions
        map!(x-> lb .+ x .* (ub .- lb), x, x) # particle positions
        map!(x-> vlow .+ x .* (vhigh .- vlow), v, v)# particle velocities
       
        fx = [obj(x[i]) for i in 1:S]  # current particle function values
        fs = [is_feasible(x[i]) for i = 1:S]  # feasibility of each particle
        fp = ones(S) * Inf  # best particle function values

        g = copy(x)  # best swarm position
        fg = ones(S) * Inf  # best swarm position starting value

        # Store particle's best position (if constraints are satisfied)
        update_position!(x, p, fx, fp, fs)

        # Update swarm's best position
        fg, g = neighborhood(fx, fp, n)
        g = copy(p[g])

        # TODO: tirei o while e tudo abaixo dele daqui, colocar posteriormente
    end

    function pso(func, lb, ub; 
        neighborhood = findBestGlobal, n = 2, constraints=nothing, args=(), kwargs=Dict(), swarmsize=100,
                 omega=0.5, phip=0.5, phig=0.5, maxiter=100, minstep=1e-8, minfunc=1e-8,
                 verbose=false, particle_output=false)
        #g, fg, p, fp = pso(func, lb, ub, constraints, args, kwargs,
        #    swarmsize, omega, phip, phig, maxiter, minstep, minfunc, verbose)
        #return particle_output? (g, fg, p, fp) : (g, fg)
        # TODO: a parte acima está comentada temporariamente para desenvolvimento
        @assert iseven(n) "The value of 'n' must be even"
        @assert length(ub) == length(lb) "ub and lb must have same dimension"
        @assert all(ub .> lb) "Each value of 'ub' must be greater than 'lb'"
        @assert (swarmsize > n) " 'swarmsize' must be greater than 'n'"

        n= Integer(n/2)
        pso(func, lb, ub, constraints, args, kwargs,
        swarmsize, omega, phip, phig, maxiter, minstep, minfunc, verbose, neighborhood, n)
        
        
    end

    export pso, findBestGlobal, findBestLocal ;

end # module
