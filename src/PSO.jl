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
        verbose && println("Initialize the particle swarm.")
        S = swarmsize
        D = length(lb) # the number of dimensions each particle has
        vhigh = abs.(ub .- lb)
        vlow = -vhigh

        x = [rand(D) for x in 1:S] # particle positions
        v = [rand(D) for x in 1:S] # particle velocities
        p = [zeros(D) for x in 1:S] # best particle positions
        map!(x-> lb .+ x .* (ub .- lb), x, x) # particle positions
        map!(x-> vlow .+ x .* (vhigh .- vlow), v, v)# particle velocities

        fx = [obj(x[i]) for i = 1:S]  # current particle function values
        fs = [is_feasible(x[i]) for i = 1:S]  # feasibility of each particle
        fp = ones(S) * Inf  # best particle function values

        g = copy(x)  # best neighborhood swarm position
        fg = ones(S) * Inf  # best neighborhood swarm position starting value

        best_position = copy(x[1]) # best position
        best_value= Inf # best position value

        # Store particle's best position (if constraints are satisfied)
        verbose && println("Store particle's best position.")
        update_position!(x, p, fx, fp, fs)

        # Update swarm's best position
        verbose && println("Update swarm's best position.")
        fg, g = neighborhood(fx, fp, n)
        g = copy(p[g])

        it = 1
        it_best = 1
        verbose && println("Start iteration.")
        while it <= maxiter
        
        # Update the particles' velocities and positions
        v = ϕg*map((x,y)->x.*y,g.-x,[rand(D) for i= 1:S ])
        v .+=  ϕp*map((x,y)->x.*y,p.-x,[rand(D) for i= 1:S ])
        v .+= ω*v
        x .+= v
        
        # Correct for bound violations
        maskl = [x[i] .< lb for i = 1:S]
        masku = [x[i] .> ub for i = 1:S]
        mask = [ maskl[i] .| masku[i] for i = 1:S]
        x = map((x,y) -> x.*(.~(y)), x, mask)
        x .+= [lb.*maskl[i] for i = 1:S]
        x .+= [ub.*masku[i] for i=1:S]

        # Update objectives and constraints
        for i = 1:S
            fx[i] = obj(x[i])
            fs[i] = is_feasible(x[i])
        end

        # Store particle's best position (if constraints are satisfied)
        update_position!(x, p, fx, fp, fs)

        # Compare swarm's best position with global best position
        i_min = findmin(fp) 
        if i_min[1] < best_value
            it_best = it
            stepsize = √(sum((best_position .- p[i_min[2]]).^2))
            if abs.(best_value .- i_min[1]) <= minfunc
                verbose && println("\n Stopping search: Swarm best objective change less than $(minfunc)")
                return (best_position, best_value, p, fp)
            end
            if stepsize <= minstep
                verbose && println("Stopping search: Swarm best position change less than $(minstep)")
                return (best_position, best_value, p, fp)
            end
            best_position = copy(p[i_min[2]]) # best position
            best_value= i_min[1] # best position value    
        end

        verbose && (print("\r Iteration : $(it)/$(maxiter) | best-iteration $(it_best) | Best value: $(best_value) "))
        it += 1
        end
        println("\n Stopping search: maximum iterations reached --> $(maxiter)")
        is_feasible(best_position) || print("However, the optimization couldn't find a feasible design. Sorry")
        return (best_position, best_value, p, fp)

    end

    function pso(func, lb, ub; 
        neighborhood = findBestGlobal, n = 2, constraints=nothing, args=(), kwargs=Dict(), swarmsize=100,
                omega=0.5, phip=0.5, phig=0.5, maxiter=100, minstep=1e-8, minfunc=1e-8,
                verbose=false, particle_output=false)
       
        @assert iseven(n) "The value of 'n' must be even"
        @assert length(ub) == length(lb) "ub and lb must have same dimension"
        @assert all(ub .> lb) "Each value of 'ub' must be greater than 'lb'"
        @assert (swarmsize >= n) " 'swarmsize' must be greater than 'n'"

        n= Integer(n/2)
        g, fg, p, fp = pso(func, lb, ub, constraints, args, kwargs,
        swarmsize, omega, phip, phig, maxiter, minstep, minfunc, verbose, neighborhood, n)
        return particle_output ? (g, fg, p, fp) : (g, fg)

    end


    function psoInt(func::Function, lb::Vector, ub::Vector, constraints, args, kwargs,
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
        map!(x->round.(x),x,x)
        map!(x->round.(x),v,v)
        p = [zeros(D) for x in 1:S] # best particle positions
        map!(x-> lb .+ x .* (ub .- lb), x, x) # particle positions
        map!(x-> vlow .+ x .* (vhigh .- vlow), v, v)# particle velocities

        fx = [obj(x[i]) for i = 1:S]  # current particle function values
        fs = [is_feasible(x[i]) for i = 1:S]  # feasibility of each particle
        fp = ones(S) * Inf  # best particle function values

        g = copy(x)  # best neighborhood swarm position
        fg = ones(S) * Inf  # best neighborhood swarm position starting value

        best_position = copy(x[1]) # best position
        best_value= Inf # best position value

        # Store particle's best position (if constraints are satisfied)
        update_position!(x, p, fx, fp, fs)

        # Update swarm's best position
        fg, g = neighborhood(fx, fp, n)
        g = copy(p[g])

        it = 1
        it_best = 1
        while it <= maxiter
        
        # Update the particles' velocities and positions
        v = ϕg*map((x,y)->x.*y,g.-x,[rand(D) for i= 1:S ])
        v .+=  ϕp*map((x,y)->x.*y,p.-x,[rand(D) for i= 1:S ])
        v .+= ω*v
        map!(x->round.(x),v,v)
        x .+= v
        #map!(x->round.(x),x,x)
        
        # Correct for bound violations
        maskl = [x[i] .< lb for i = 1:S]
        masku = [x[i] .> ub for i = 1:S]
        mask = [ maskl[i] .| masku[i] for i = 1:S]
        x = map((x,y) -> x.*(.~(y)), x, mask)
        x .+= [lb.*maskl[i] for i = 1:S]
        x .+= [ub.*masku[i] for i=1:S]

        # Update objectives and constraints
        for i = 1:S
            fx[i] = obj(x[i])
            fs[i] = is_feasible(x[i])
        end

        # Store particle's best position (if constraints are satisfied)
        update_position!(x, p, fx, fp, fs)

        # Compare swarm's best position with global best position
        i_min = findmin(fp) 
        if i_min[1] < best_value
            it_best = it
            stepsize = √(sum((best_position .- p[i_min[2]]).^2))
            if abs.(best_value .- i_min[1]) <= minfunc
                verbose && println("\n Stopping search: Swarm best objective change less than $(minfunc)")
                return (best_position, best_value, p, fp)
            end
            if stepsize <= minstep
                verbose && println("Stopping search: Swarm best position change less than $(minstep)")
                return (best_position, best_value, p, fp)
            end
            best_position = copy(p[i_min[2]]) # best position
            best_value= i_min[1] # best position value    
        end

        verbose && (print("\r Iteration : $(it)/$(maxiter) | best-iteration $(it_best) | Best value: $(best_value)"))
        it += 1
        end
        verbose && println("\n Stopping search: maximum iterations reached --> $(maxiter)")
        is_feasible(best_position) || print("However, the optimization couldn't find a feasible design. Sorry")
        return (best_position, best_value, p, fp)

    end

    function psoInt(func, lb, ub; 
        neighborhood = findBestGlobal, n = 2, constraints=nothing, args=(), kwargs=Dict(), swarmsize=100,
                omega=0.5, phip=0.5, phig=0.5, maxiter=100, minstep=1e-8, minfunc=1e-8,
                verbose=false, particle_output=false)
 
        @assert iseven(n) "The value of 'n' must be even"
        @assert length(ub) == length(lb) "ub and lb must have same dimension"
        @assert all(ub .> lb) "Each value of 'ub' must be greater than 'lb'"
        @assert (swarmsize >= n) " 'swarmsize' must be greater than 'n'"

        n= Integer(n/2)
        g, fg, p, fp = psoInt(func, lb, ub, constraints, args, kwargs,
        swarmsize, omega, phip, phig, maxiter, minstep, minfunc, verbose, neighborhood, n)
        return particle_output ? (g, fg, p, fp) : (g, fg)

    end
    export pso, psoInt, findBestGlobal, findBestLocal ;

end # module
