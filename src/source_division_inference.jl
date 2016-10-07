import WCS


function fetch_catalog(rcf, stagedir)
    # note: this call to read_photoobj_files considers only primary detections.
    catalog = SDSSIO.read_photoobj_files([rcf,], stagedir)

    # we're ignoring really faint sources entirely...not even using them to
    # render the background
    isnt_faint(entry) = maximum(entry.star_fluxes) >= MIN_FLUX
    filter(isnt_faint, catalog)
end


@inline function in_box(entry::CatalogEntry, box::BoundingBox)
    box.ramin < entry.pos[1] < box.ramax &&
        box.decmin < entry.pos[2] < box.decmax
end


function load_catalog(box, rcfs, stagedir)
    num_fields = length(rcfs)
    if nodeid == 1
        nputs(nodeid, "$num_fields RCFs")
    end

    catalog = Vector{Tuple{CatalogEntry,RunCamcolField}}()
    cat_idx = 1
    tasks = Vector{Int64}()

    for i = 1:num_fields
        rcf_cat = fetch_catalog(rcfs[i], stagedir)
        for entry in rcf_cat
            push!(catalog, (entry, rcf))
            if in_box(entry, box)
                push!(tasks, cat_idx)
            end
            cat_idx = cat_idx + 1
        end
    end

    if nodeid == 1
        nputs(nodeid, "catalog size is $(length(catalog)), $(length(tasks)) tasks")
    end

    sync()

    catalog, tasks
end


function optimize_source(s::Int64, images::Garray, catalog::Garray,
                         catalog_offset::Garray, rcf_to_index::Array{Int64,3},
                         rcf_cache::Dict, rcf_cache_lock::SpinLock,
                         g_lock::SpinLock, stagedir::String, times::InferTiming)
    tid = Base.Threads.threadid()

    entry = catalog[s]
    t_box = BoundingBox(entry.pos[1] - 1e-8, entry.pos[1] + 1e-8,
                        entry.pos[2] - 1e-8, entry.pos[2] + 1e-8)
    surrounding_rcfs = get_overlapping_fields(t_box, stagedir)

    local_images = Vector{TiledImage}()
    local_catalog = Vector{CatalogEntry}()
    tic()
    for rcf in surrounding_rcfs
        lock(rcf_cache_lock)
        cached_imgs, ihandle, cached_cat, chandle, lru = get!(rcf_cache, rcf) do
            ntputs(nodeid, tid, "fetching $(rcf.run), $(rcf.camcol), $(rcf.field)")

            n = rcf_to_index[rcf.run, rcf.camcol, rcf.field]
            @assert n > 0

            tic()
            lock(g_lock)
            fimgs, ihandle = get(images, [n], [n])
            unlock(g_lock)
            times.ga_get = times.ga_get + toq()
            imgs = Vector{TiledImage}()
            for fimg in fimgs[1]
                img = Image(fimg)
                push!(imgs, TiledImage(img))
            end

            if n == 1
                s_a = 1
                tic()
                lock(g_lock)
                st, st_handle = get(catalog_offset, [n], [n])
                unlock(g_lock)
                times.ga_get = times.ga_get + toq()
                s_b = st[1]
            else
                tic()
                lock(g_lock)
                st, st_handle = get(catalog_offset, [n-1], [n])
                unlock(g_lock)
                times.ga_get = times.ga_get + toq()
                s_a = st[1]
                s_b = st[2]
            end
            tic()
            lock(g_lock)
            cat_entries, chandle = get(catalog, [s_a], [s_b])
            unlock(g_lock)
            times.ga_get = times.ga_get + toq()
            neighbors = [entry[1] for entry in cat_entries]

            imgs, ihandle, neighbors, chandle, time()
        end
        unlock(rcf_cache_lock)
        append!(local_images, cached_imgs)
        append!(local_catalog, cached_cat)
    end
    #ntputs(nodeid, tid, "fetched data to infer $s in $(toq()) secs")

    tic()
    i = findfirst(local_catalog, entry)
    neighbor_indexes = Infer.find_neighbors([i,], local_catalog, local_images)[1]
    neighbors = local_catalog[neighbor_indexes]
    #ntputs(nodeid, tid, "loaded neighbors of $s in $(toq()) secs")

    t0 = time()
    vs_opt = Infer.infer_source(local_images, neighbors, entry)
    runtime = time() - t0
    #ntputs(nodeid, tid, "ran inference for $s in $runtime secs")

    InferResult(entry.thing_id, entry.objid, entry.pos[1], entry.pos[2],
                vs_opt, runtime)
end


function optimize_sources(images, catalog, tasks, catalog_offset, task_offset,
            rcf_to_index, stagedir, timing)
    num_work_items = length(tasks)

    # inference results
    results = Vector{InferResult}()
    results_lock = SpinLock()

    # cache for RCF data; key is RCF, 
    rcf_cache = Dict{RunCamcolField,
                     Tuple{Vector{TiledImage},
                           GarrayMemoryHandle,
                           Vector{CatalogEntry},
                           GarrayMemoryHandle,
                           Float64}}()
    rcf_cache_lock = SpinLock()

    g_lock = SpinLock()

    # per-thread timing
    ttimes = Array(InferTiming, Base.Threads.nthreads())

    # create Dtree and get the initial allocation
    dt, isparent = Dtree(num_work_items, 0.4,
                         ceil(Int64, Base.Threads.nthreads() / 4))
    numwi, (startwi, endwi) = initwork(dt)
    rundt = runtree(dt)

    nputs(nodeid, "initially $numwi work items ($startwi-$endwi)")
    workitems, wi_handle = get(tasks, [startwi], [endwi])

    widx = 1
    wilock = SpinLock()

    function process_tasks()
        tid = Base.Threads.threadid()
        ttimes[tid] = InferTiming()
        times = ttimes[tid]

        if rundt && tid == 1
            ntputs(nodeid, tid, "running tree")
            while runtree(dt)
                Garbo.cpu_pause()
            end
        else
            while true
                lock(wilock)
                tic()
                if endwi == 0
                    ntputs(nodeid, tid, "out of work")
                    unlock(wilock)
                    times.sched_ovh = times.sched_ovh + toq()
                    break
                end
                if widx > numwi
                    ntputs(nodeid, tid, "consumed last work item; requesting more")
                    lock(g_lock)
                    numwi, (startwi, endwi) = getwork(dt)
                    unlock(g_lock)
                    times.sched_ovh = times.sched_ovh + toq()
                    ntputs(nodeid, tid, "got $numwi work items ($startwi-$endwi)")
                    if endwi > 0
                        tic()
                        lock(g_lock)
                        workitems, wi_handle = get(tasks, [startwi], [endwi])
                        unlock(g_lock)
                        times.ga_get = times.ga_get + toq()
                        widx = 1
                    end
                    unlock(wilock)
                    continue
                end
                taskidx = startwi+widx-1
                item = workitems[widx]
                widx = widx + 1
                times.sched_ovh = times.sched_ovh + toq()
                unlock(wilock)
                ntputs(nodeid, tid, "processing source $item")

                result = optimize_source(item, images, catalog, catalog_offset,
                                 rcf_to_index, rcf_cache, rcf_cache_lock,
                                 g_lock, stagedir, times)

                lock(results_lock)
                push!(results, result)
                unlock(results_lock)
            end
        end
    end

    tic()
    ccall(:jl_threading_run, Void, (Any,), Core.svec(process_tasks))
    #process_tasks()
    #ccall(:jl_threading_profile, Void, ())
    timing.opt_srcs = toq()

    if nodeid == 1
        nputs(nodeid, "complete")
    end
    tic()
    finalize(dt)
    timing.wait_done = toq()

    for tt in ttimes
        add_timing!(timing, tt)
    end

    return results
end


function set_thread_affinity(nid::Int, ppn::Int, tid::Int, nthreads::Int, show_cpu::Bool)
    cpu = (((nid - 1) % ppn) * nthreads)
    show_cpu && ntputs(nid, tid, "bound to $(cpu + tid)")
    mask = zeros(UInt8, 4096)
    mask[cpu + tid] = 1
    uvtid = ccall(:uv_thread_self, UInt64, ())
    ccall(:uv_thread_setaffinity, Int, (Ptr{Void}, Ptr{Void}, Ptr{Void}, Int64),
          pointer_from_objref(uvtid), mask, C_NULL, 4096)
end


function affinitize()
    ppn = try
        parse(Int, ENV["JULIA_EXCLUSIVE"])
    catch exc
        return
    end
    show_cpu = try
        parse(Bool, ENV["CELESTE_SHOW_AFFINITY"])
    catch exc
        false
    end
    function threadfun()
        set_thread_affinity(nodeid, ppn,
                            Base.Threads.threadid(), Base.Threads.nthreads(), show_cpu)
    end
    ccall(:jl_threading_run, Void, (Any,), Core.svec(threadfun))
end


function load_tasks(box::BoundingBox, rcfs::Vector{RunCamcolField}, stagedir::String)
end

"""
Fit the Celeste model to sources in a given ra, dec range,
based on data from specified fields
- box: a bounding box specifying a region of sky
"""
function divide_sources_and_infer(
                box::BoundingBox,
                stagedir::String;
                timing=InferTiming(),
                outdir=".")
    affinitize()

    tasks = load_tasks(box, rcfs, stagedir)

    # read the run-camcol-field triplets for this box
    rcfs = get_overlapping_fields(box, stagedir)

    # loads 25TB from disk for SDSS
    tic()
    images, catalog_offset, task_offset = load_images(box, rcfs, stagedir)
    timing.load_img = toq()

    try
        t = ENV["CELESTE_EXIT_AFTER_LOAD_IMAGES"]
        exit()
    catch exc
    end

    # loads 4TB from disk for SDSS
    tic()
    catalog, tasks = load_catalog(box, rcfs, catalog_offset, task_offset, stagedir)
    timing.load_cat = toq()

    try
        t = ENV["CELESTE_EXIT_AFTER_LOAD_CATALOG"]
        exit()
    catch exc
    end

    if length(tasks) > 0
        # create map from run, camcol, field to index into RCF array
        rcf_to_index = invert_rcf_array(rcfs)

        # optimization -- little disk access, cpu intensive
        results = optimize_sources(images, catalog, tasks,
                                   catalog_offset, task_offset,
                                   rcf_to_index, stagedir, timing)

        timing.num_srcs = length(results)

        tic()
        save_results(outdir, box, results)
        timing.write_results = toq()
    end

    finalize(tasks)
    finalize(catalog)
    finalize(task_offset)
    finalize(catalog_offset)
    finalize(images)
end

